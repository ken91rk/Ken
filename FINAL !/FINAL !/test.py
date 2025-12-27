import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import os
import utils

def test_performance():
    print("[MODEL] Chargement des données pour le TEST...")
    if not os.path.exists(utils.FILE_PAIRWISE):
        print(f"Fichier {utils.FILE_PAIRWISE} manquant.")
        return

    df = pd.read_csv(utils.FILE_PAIRWISE)
    features = ['dist_between_a_b', 'diff_angle_depot', 'diff_dist_depot', 'diff_dist_center', 'diff_density', 'dist_depot_a', 'dist_depot_b', 'angle_depot_a', 'angle_depot_b']
    target = 'TARGET_A_BEFORE_B'

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = xgb.XGBClassifier()
    try:
        model.load_model(utils.MODEL_FILE)
    except FileNotFoundError:
        print(f"Modèle {utils.MODEL_FILE} introuvable.")
        return

    print("[ANALYSE TRAIN] Performances sur le train")
    y_pred_train = model.predict(X_train)
    acc_train = accuracy_score(y_train, y_pred_train)
    print(f"Accuracy (Train) : {acc_train:.2%}")
    print("\n--- Rapport de Classification (Train) ---")
    print(classification_report(y_train, y_pred_train))

    cm_train = confusion_matrix(y_train, y_pred_train)
    disp_train = ConfusionMatrixDisplay(confusion_matrix=cm_train, display_labels=["B avant A", "A avant B"])
    plt.figure(figsize=(6,6))
    disp_train.plot(cmap='Blues', values_format='d')
    plt.title(f"Matrice de Confusion (Train set)\nAccuracy: {acc_train:.2%}")
    plt.savefig("analyse_confusion_matrix_train.png")
    plt.close()

    print("[EVAL TEST] Performances sur le test")
    y_pred_test = model.predict(X_test)
    acc_test = accuracy_score(y_test, y_pred_test)
    print(f"Accuracy (Test) : {acc_test:.2%}")
    print("\n--- Rapport de Classification (Test) ---")
    print(classification_report(y_test, y_pred_test))

    cm_test = confusion_matrix(y_test, y_pred_test)
    disp_test = ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=["B avant A", "A avant B"])
    plt.figure(figsize=(6,6))
    disp_test.plot(cmap='Blues', values_format='d')
    plt.title(f"Matrice de Confusion (Test set)\nAccuracy: {acc_test:.2%}")
    plt.savefig("analyse_confusion_matrix_test.png")
    plt.close()

    print("Test terminé.")