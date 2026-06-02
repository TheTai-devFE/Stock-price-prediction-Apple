import os
import sys
from src.spark_pipeline.spark_processor import SparkProcessor
from src.model_trainer import ModelTrainer

def run_spark_pipeline():
    print("=== APPLE STOCK PRICE PREDICTION PIPELINE: SPARK & DATA LAKE ===")
    
    # Initialize Spark Processor
    processor = SparkProcessor()
    
    # 1. Bootstrapping Data Lake (simulating loading historical HDFS files)
    import glob
    historical_files = glob.glob("data/data_lake/part_historical_*.json")
    if not historical_files:
        print("[*] Data Lake is empty. Bootstrapping historical data from MongoDB aapl_raw...")
        success = processor.initialize_data_lake()
        if not success:
            print("[!] Bootstrapping failed. Exiting.")
            return
    else:
        print(f"[*] Found {len(historical_files)} historical partition files in Data Lake.")
        
    # 2. Run Spark processing (ETL & Feature Engineering) on Data Lake partitions
    print("\n--- Apache Spark Stage 3 & 5: ETL & Feature Engineering ---")
    df_spark_features = processor.process_data_lake()
    
    if df_spark_features.empty:
        print("[!] Spark processing returned empty DataFrame. Exiting.")
        return
        
    # 3. Model Training & Evaluation (using Spark processed features)
    print("\n--- Stage 6 & 7: Model Training & Evaluation (Spark Data) ---")
    # Store reports in a separate directory to avoid overwriting old pipeline reports
    trainer = ModelTrainer(report_path="reports/models_spark")
    
    # Prepare data
    X_train, X_test, y_train, y_test = trainer.prepare_data(df_spark_features, target_col='close')
    
    # 1. Linear Regression
    trainer.train_linear_regression(X_train, y_train)
    
    # 2. XGBoost
    trainer.train_xgboost(X_train, y_train)
    
    # 3. LightGBM
    trainer.train_lightgbm(X_train, y_train)
    
    # 4. CatBoost
    trainer.train_catboost(X_train, y_train)
    
    # 5. ARIMA + LSTM Hybrid
    trainer.train_arima_lstm(X_train, y_train)

    # Evaluate all models and save markdown report
    df_metrics = trainer.evaluate_all(X_test, y_test)
    trainer.save_metrics_report(df_metrics)

    print("\n[SUCCESS] Spark Pipeline execution completed successfully.")


if __name__ == "__main__":
    run_spark_pipeline()
