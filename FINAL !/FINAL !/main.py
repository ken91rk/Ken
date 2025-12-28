import data
import model
import test
import model_2
import time

def main():
    start_time = time.time()
    print("Lancement du projet")
    print("==========================================================")
    
    # 1. GÉNÉRATION DONNÉES
    # Génère DATASET_FINAL.csv nécessaire pour la suite
    data.run_step_1_extraction()
    data.run_step_2_clustering()
    data.run_step_3_pairwise()
    
    # 2. ENTRAÎNEMENT (Cross-Validation)
    model.train_model_cv()
    
    # 3. ENTRAÎNEMENT (ANALYSE SIMPLE SPLIT)
    model_2.run_simple_split_analysis()
    
    # 4. TEST FINAL
    test.test_performance()
    
    total_time = time.time() - start_time
    print(f"\nPROCESSUS TERMINÉ en {total_time:.0f} secondes.")

if __name__ == "__main__":
    main()