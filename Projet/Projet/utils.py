import math
import numpy as np
import requests
import time
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

DEPOT_DICT = {"name": "Gare de Lyon", "lat": 48.8443, "lon": 2.3748}
# Dépôt format Tuple (Fichier 2 & 3)
DEPOT_COORDS = (48.8443, 2.3748)

N_CLUSTERS = 10
WORKERS = 10

# URLs
OVERPASS_URL = "http://overpass-api.de/api/interpreter"
OSRM_TABLE_URL = "http://router.project-osrm.org/table/v1/driving/"
OSRM_URL_TEMPLATE = "http://router.project-osrm.org/route/v1/driving/{lon1},{lat1};{lon2},{lat2}?overview=false"

# Noms de fichiers
FILE_RAW = "dataset_1_raw.csv"
FILE_PAIRWISE = "DATASET_FINAL.csv"
MODEL_FILE = "xgboost_TSP_model.json"
INPUT_PATTERN_CLUSTERS = "cluster_{}.csv"


# FONCTIONS OSRM / OVERPASS (Fichier 1)

def get_unique_places_from_osm():
    print("Récupération des lieux d'intérêt dans Paris depuis OpenStreetMap...")
    query = """
    [out:json][timeout:60];
    (
      node["tourism"](48.815,2.224,48.902,2.469);
      node["amenity"](48.815,2.224,48.902,2.469); 
      node["historic"](48.815,2.224,48.902,2.469);
      node["leisure"](48.815,2.224,48.902,2.469);
    );
    out center;
    """
    try: 
        response = requests.get(OVERPASS_URL, params={'data': query})
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException as e:
        print(f"Erreur lors de la requête vers l'API Overpass : {e}")
        return []

    unique_places = set()
    for element in data.get('elements', []):
        name = element.get('tags', {}).get('name')
        if name and 'lat' in element and 'lon' in element:
            unique_places.add((name, element['lat'], element['lon']))
    print(f"{len(unique_places)} lieux uniques trouvés dans Paris.")
    return list(unique_places)

def get_route_details(start_coords, end_coords):
    url = OSRM_URL_TEMPLATE.format(
        lon1=start_coords['lon'], lat1=start_coords['lat'],
        lon2=end_coords['lon'], lat2=end_coords['lat']
    )
    try: 
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get('code') == 'Ok':
                route = data['routes'][0]
                distance_km = round(route['distance'] / 1000, 2)
                duration_min = round(route['duration'] / 60, 2)
                return distance_km, duration_min
    except requests.exceptions.RequestException:
        return None, None
    return None, None

def process_place(place, depot):
    name, lat, lon = place
    destination = {"lat": lat, "lon": lon}
    depot_coords = {"lat": depot["lat"], "lon": depot["lon"]}
    distance_km, duration_min = get_route_details(depot_coords, destination)
    if distance_km is not None and duration_min is not None:
        return {
            "depot": depot["name"],
            "destination_name": name,
            "destination_lat": lat,
            "destination_lon": lon,
            "distance_km": distance_km,
            "duration_min": duration_min,
        }
    return None


# FONCTIONS MATRICES / OR-TOOLS 

def get_matrix_chunked(coords_list, cluster_id):
    n = len(coords_list)
    print(f"  [Cluster {cluster_id}] Calcul optimisé de la matrice ({n} points) via OSRM Table")
    full_matrix = np.zeros((n, n), dtype=int)
    CHUNK_SIZE = 50
    chunks = [range(i, min(i + CHUNK_SIZE, n)) for i in range(0, n, CHUNK_SIZE)]
    total_requests = len(chunks) ** 2
    pbar = tqdm(total=total_requests, desc=f"  Cluster {cluster_id} - Requêtes API", leave=False)

    for i_chunk in chunks:
        for j_chunk in chunks:
            src_coords = [coords_list[i] for i in i_chunk]
            dst_coords = [coords_list[j] for j in j_chunk]
            all_current_coords = src_coords + dst_coords
            coords_str = ";".join([f"{lon},{lat}" for lat, lon in all_current_coords])
            src_indices = ";".join(map(str, range(len(src_coords))))
            dst_indices = ";".join(map(str, range(len(src_coords), len(src_coords) + len(dst_coords))))
            url = f"{OSRM_TABLE_URL}{coords_str}?sources={src_indices}&destinations={dst_indices}&annotations=duration"
            try:
                time.sleep(0.5)
                response = requests.get(url, timeout=20)
                if response.status_code == 200:
                    data = response.json()
                    if 'durations' in data:
                        durations = data['durations']
                        for local_r, real_r in enumerate(i_chunk):
                            for local_c, real_c in enumerate(j_chunk):
                                val = durations[local_r][local_c]
                                if val is None: val = 99999
                                full_matrix[real_r][real_c] = int(val * 10)
            except Exception as e:
                print(f"Erreur API : {e}")
            pbar.update(1)
    pbar.close()
    return full_matrix.tolist()

def solve_tsp(distance_matrix):
    if not distance_matrix: return None
    manager = pywrapcp.RoutingIndexManager(len(distance_matrix), 1, 0)
    routing = pywrapcp.RoutingModel(manager)
    def distance_callback(from_index, to_index):
        return distance_matrix[manager.IndexToNode(from_index)][manager.IndexToNode(to_index)]
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_parameters.time_limit.seconds = 30
    solution = routing.SolveWithParameters(search_parameters)
    if solution:
        index = routing.Start(0)
        route = []
        while not routing.IsEnd(index):
            route.append(manager.IndexToNode(index))
            index = solution.Value(routing.NextVar(index))
        return route
    return None


# FONCTIONS MATHS / FEATURES 

def calculate_haversine(lat1, lon1, lat2, lon2):
    R = 6371
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def calculate_bearing(lat1, lon1, lat2, lon2):
    dlon = math.radians(lon2 - lon1)
    lat1 = math.radians(lat1)
    lat2 = math.radians(lat2)
    y = math.sin(dlon) * math.cos(lat2)
    x = (math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon))
    bearing = math.degrees(math.atan2(y, x))
    return (bearing + 360) % 360

def compute_density(df_cluster):
    coords = df_cluster[['destination_lat', 'destination_lon']].values
    if len(coords) < 6: return [0] * len(coords)
    nbrs = NearestNeighbors(n_neighbors=6, algorithm='ball_tree').fit(coords)
    distances, _ = nbrs.kneighbors(coords)
    density_scores = np.mean(distances[:, 1:], axis=1) * 100
    return density_scores