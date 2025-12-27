import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import time
from tqdm import tqdm
import utils

def train_model_cv():
    print(f"\nEntraînement du modèle avec 5 folds (early stopping activé).")
    
    try:
        df = pd.read_csv(utils.FILE_PAIRWISE)
        print(f"Dataset chargé : {len(df)} lignes.")
    except FileNotFoundError:
        print("Erreur : fichier introuvable."); return

    FEATURES = [
        'dist_between_a_b', 'diff_angle_depot', 'diff_dist_depot', 'diff_dist_center',
        'diff_density', 'dist_depot_a', 'dist_depot_b', 'angle_depot_a', 'angle_depot_b'
    ]
    TARGET = 'TARGET_A_BEFORE_B'

    X = df[FEATURES]
    y = df[TARGET]

    base_model = xgb.XGBClassifier(
        n_estimators=7000,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        n_jobs=-1,
        random_state=42,
        early_stopping_rounds=50
    )

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    best_iterations = []
    start_time = time.time()

    for fold_id, (train_idx, test_idx) in enumerate(tqdm(kfold.split(X, y), total=5, desc="Progression")):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        base_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        best_iter = base_model.best_iteration + 1
        best_iterations.append(best_iter)
        preds = base_model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        scores.append(acc)
        tqdm.write(f"Fold {fold_id + 1}: arrêt à {best_iter} arbres | Accuracy = {acc:.2%}")

    scores = np.array(scores)
    training_time = time.time() - start_time

    print("\n" + "=" * 45)
    print("Résultats de la validation croisée")
    print("=" * 45)
    print(f"Nombre moyen d'arbres : {int(np.mean(best_iterations))}")
    print(f"Accuracy moyenne      : {scores.mean():.2%}")
    print(f"Stabilité (±2σ)        : {scores.std() * 2:.2%}")
    print(f"Temps total            : {training_time:.1f} secondes")
    print("=" * 45)

    optimal_trees = int(np.mean(best_iterations))
    print(f"\nEntraînement final avec {optimal_trees} arbres.")

    final_model = xgb.XGBClassifier(
        n_estimators=optimal_trees,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        n_jobs=-1,
        random_state=42
    )
    final_model.fit(X, y)

    feature_importance = pd.DataFrame({'Feature': FEATURES, 'Importance': final_model.feature_importances_}).sort_values(by='Importance', ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance, palette='viridis')
    plt.title("Importance des variables (modèle final)")
    plt.tight_layout()
    plt.savefig("feature_importance.png")

    final_model.save_model(utils.MODEL_FILE)
    print(f"Modèle sauvegardé : {utils.MODEL_FILE}")