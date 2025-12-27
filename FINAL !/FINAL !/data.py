import pandas as pd
import os
import numpy as np
import utils
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from concurrent.futures import ThreadPoolExecutor, as_completed

def run_step_1_extraction():
    """Correspond à ton fichier 1_data_extraction.py"""
    print("\n--- ÉTAPE 1 : EXTRACTION ---")
    if os.path.exists(utils.FILE_RAW):
        print(f"Fichier {utils.FILE_RAW} existe déjà. On passe.")
        return

    osm_places = utils.get_unique_places_from_osm()
    if osm_places:
        dataset_rows = []
        print("\nCalcul des distances et durées (traitement parallèle)...")
        with ThreadPoolExecutor(max_workers=utils.WORKERS) as executor:
            futures = {
                executor.submit(utils.process_place, place, utils.DEPOT_DICT): place
                for place in osm_places
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="Calcul des routes"):
                result = future.result()
                if result:
                    dataset_rows.append(result)
        
        df = pd.DataFrame(dataset_rows)
        df.to_csv(utils.FILE_RAW, index=False, encoding="utf-8-sig")
        print(f"\nDataset complet enregistré : {utils.FILE_RAW}")

def run_step_2_clustering():
    """Correspond à ton fichier 2_data_clustering_and_order.py"""
    print("\n--- ÉTAPE 2 : CLUSTERING & VÉRITÉ TERRAIN ---")
    try:
        df = pd.read_csv(utils.FILE_RAW)
        df = df.dropna(subset=['destination_lat', 'destination_lon'])
    except:
        print("Erreur chargement fichier raw."); return

    scaler = StandardScaler()
    coords_scaled = scaler.fit_transform(df[['destination_lat', 'destination_lon']])
    kmeans = KMeans(n_clusters=utils.N_CLUSTERS, n_init=10, random_state=42)
    df['cluster'] = kmeans.fit_predict(coords_scaled) + 1

    for c_id in range(1, utils.N_CLUSTERS + 1):
        out_name = utils.INPUT_PATTERN_CLUSTERS.format(c_id)
        if os.path.exists(out_name): continue

        df_c = df[df['cluster'] == c_id].copy().reset_index(drop=True)
        if len(df_c) < 2: continue

        print(f"\nTraitement du cluster {c_id}/{utils.N_CLUSTERS} ({len(df_c)} points)")
        coords = [utils.DEPOT_COORDS] + [(r['destination_lat'], r['destination_lon']) for _, r in df_c.iterrows()]
        
        matrix = utils.get_matrix_chunked(coords, c_id)
        order = utils.solve_tsp(matrix)

        if order:
            res = []
            rank = 1
            for idx in order:
                if idx == 0: continue
                row = df_c.iloc[idx - 1].to_dict()
                row['TARGET_ORDER'] = rank
                res.append(row)
                rank += 1
            pd.DataFrame(res).to_csv(out_name, index=False)
            print(f"Fichier généré : {out_name}")

def run_step_3_pairwise():
    """Correspond à ton fichier 3_data_final.py"""
    print("\n--- ÉTAPE 3 : GÉNÉRATION PAIRWISE ---")
    if os.path.exists(utils.FILE_PAIRWISE):
        print(f"Fichier {utils.FILE_PAIRWISE} existe déjà. Fin.")
        return

    all_pairs = []
    for c_id in range(1, utils.N_CLUSTERS + 1):
        filename = utils.INPUT_PATTERN_CLUSTERS.format(c_id)
        if not os.path.exists(filename): continue

        print(f"Traitement du fichier {filename}")
        df = pd.read_csv(filename)
        center_lat = df['destination_lat'].mean()
        center_lon = df['destination_lon'].mean()
        df['density'] = utils.compute_density(df)
        
        lats = df['destination_lat'].values
        lons = df['destination_lon'].values
        
        dist_depot, angle_depot, dist_center = [], [], []
        for i in range(len(df)):
            dist_depot.append(utils.calculate_haversine(utils.DEPOT_COORDS[0], utils.DEPOT_COORDS[1], lats[i], lons[i]))
            angle_depot.append(utils.calculate_bearing(utils.DEPOT_COORDS[0], utils.DEPOT_COORDS[1], lats[i], lons[i]))
            dist_center.append(utils.calculate_haversine(center_lat, center_lon, lats[i], lons[i]))
        
        df['dist_depot'] = dist_depot
        df['angle_depot'] = angle_depot
        df['dist_center'] = dist_center

        records = df.to_dict('records')
        for i in tqdm(range(len(records)), desc=f"Cluster {c_id}", leave=False):
            row_a = records[i]
            for j in range(len(records)):
                if i == j: continue
                row_b = records[j]
                
                diff_angle = row_a['angle_depot'] - row_b['angle_depot']
                if diff_angle > 180: diff_angle -= 360
                if diff_angle < -180: diff_angle += 360
                
                dist_ab = utils.calculate_haversine(row_a['destination_lat'], row_a['destination_lon'], row_b['destination_lat'], row_b['destination_lon'])
                
                pair_data = {
                    'cluster_id': c_id,
                    'dist_between_a_b': dist_ab,
                    'diff_angle_depot': diff_angle,
                    'diff_dist_depot': row_a['dist_depot'] - row_b['dist_depot'],
                    'diff_dist_center': row_a['dist_center'] - row_b['dist_center'],
                    'diff_density': row_a['density'] - row_b['density'],
                    'dist_depot_a': row_a['dist_depot'],
                    'dist_depot_b': row_b['dist_depot'],
                    'angle_depot_a': row_a['angle_depot'],
                    'angle_depot_b': row_b['angle_depot'],
                    'TARGET_A_BEFORE_B': 1 if row_a['TARGET_ORDER'] < row_b['TARGET_ORDER'] else 0
                }
                all_pairs.append(pair_data)

    df_final = pd.DataFrame(all_pairs)
    cols_float = [c for c in df_final.columns if 'TARGET' not in c and 'cluster' not in c]
    df_final[cols_float] = df_final[cols_float].astype('float32')
    df_final.to_csv(utils.FILE_PAIRWISE, index=False)
    print(f"Dataset généré : {utils.FILE_PAIRWISE}")