import data
import model
import test
import time

def main():
    start_time = time.time()
    print("Lanceement du projet")
    print("==========================================================")
    
    # 1. GÉNÉRATION DONNÉES
    data.run_step_1_extraction()
    data.run_step_2_clustering()
    data.run_step_3_pairwise()
    
    # 2. ENTRAÎNEMENT
    model.train_model_cv()
    
    # 3. TEST
    test.test_performance()
    
    print("PROCESSUS TERMINÉ.")

if __name__ == "__main__":
    main()