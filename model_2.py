import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import time
from tqdm import tqdm
from xgboost.callback import TrainingCallback

class TqdmCallback(TrainingCallback):
    def __init__(self):
        self.bar = tqdm(total=7000, desc="Entraînement XGBoost")

    def after_iteration(self, model, epoch, evals_log):
        self.bar.update(1)
        return False

    def after_training(self, model):
        self.bar.close()
        return model

def run_simple_split_analysis():
    """
    Effectue une analyse avec un split simple : entraînement sur 80% des données,
    test sur 20%. Utilise XGBoost pour prédire l'ordre des points A et B.
    Génère un graphique d'importance des features.
    """
    print(f"\n{'═'*50}")
    print("ANALYSE SIMPLE SPLIT")
    print("Objectif : Entraîner sur Train, Tester sur Test.")
    print(f"{'═'*50}")
    
    # Chargement des données
    try:
        df = pd.read_csv("DATASET_FINAL.csv")
    except FileNotFoundError:
        print("Erreur : fichier DATASET_FINAL.csv introuvable."); return

    # Définition des features et de la target
    FEATURES = [
        'dist_between_a_b', 'diff_angle_depot', 'diff_dist_depot', 'diff_dist_center',
        'diff_density', 'dist_depot_a', 'dist_depot_b', 'angle_depot_a', 'angle_depot_b'
    ]
    TARGET = 'TARGET_A_BEFORE_B'
    X = df[FEATURES]
    y = df[TARGET]

    # Split des données : 80% train, 20% test
    print("Division des données : 80% pour l'entraînement, 20% pour le test")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Configuration du modèle XGBoost
    model = xgb.XGBClassifier(
        n_estimators=7000,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        n_jobs=-1,
        random_state=42
    )

    # Entraînement du modèle avec barre de progression
    print("Entraînement en cours sur X_train...")
    start_time = time.time()
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    params = model.get_params()
    del params['n_estimators']
    
    bst = xgb.train(
        params, dtrain,
        num_boost_round=7000,
        evals=[(dtest, 'test')],
        early_stopping_rounds=50,
        verbose_eval=False,
        callbacks=[TqdmCallback()]
    )
    
    model._Booster = bst
    duration = time.time() - start_time
    best_iter = bst.best_iteration + 1
    
    # Évaluation sur le test set
    print(f"Arrêt optimal trouvé à {best_iter} arbres.")
    print("Prédictions sur X_test...")
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    
    print(f"\nRÉSULTATS DU SPLIT UNIQUE :")
    print(f"Accuracy (Test) : {acc:.2%}")
    print(f"Temps d'entraînement : {duration:.1f} sec")
    
    # Génération du graphique d'importance des features
    feature_importance = pd.DataFrame({'Feature': FEATURES, 'Importance': model.feature_importances_}).sort_values(by='Importance', ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance, palette='rocket')
    plt.title("Importance des variables (Split Simple)")
    plt.tight_layout()
    plt.savefig("feature_importance_simple_split.png")
    print("Graphique généré : feature_importance_simple_split.png")
    print("Fin de l'analyse simple.\n")

if __name__ == "__main__":
    run_simple_split_analysis()