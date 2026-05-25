from src.data_provider import DataProvider
from src.etl_processor import ETLProcessor
from src.eda_visualizer import EDAVisualizer
from src.feature_engineer import FeatureEngineer
from src.model_trainer import ModelTrainer
from src.utils import export_table_image
import os

def run_pipeline():
    print("=== APPLE STOCK PRICE PREDICTION PIPELINE: STAGES 1-7 ===")
    
    # Configuration
    TICKER = "AAPL"
    START_DATE = "2010-01-01"
    COLLECTION_RAW = "aapl_raw"
    COLLECTION_CLEANED = "aapl_cleaned"
    COLLECTION_FEATURES = "aapl_features"

    # Initialize components
    provider = DataProvider()
    etl = ETLProcessor()

    # --- Stage 1 & 2: Collection & Storage ---
    print("\n--- Giai đoạn 1 & 2: Data Collection & Storage (MongoDB) ---")
    raw_data = provider.fetch_stock_data(TICKER, START_DATE)
    
    if not raw_data.empty:
        print(f"[*] Raw data head:\n{raw_data.head()}")
        provider.save_data(raw_data, COLLECTION_RAW)
        
        # --- Export Raw Table for Report ---
        export_table_image(
            raw_data, 
            "AAPL Raw Data (MongoDB Storage)", 
            "reports/raw_data_table.png"
        )
    else:
        print("[!] Failed to fetch data. Exiting.")
        return

    # --- Stage 3: ETL ---
    print("\n--- Giai đoạn 3: ETL (Extract, Transform, Load) ---")
    # Load raw from MongoDB for cleaning
    df_to_clean = provider.load_data(COLLECTION_RAW)
    
    # Clean data
    cleaned_data = etl.clean_data(df_to_clean)
    
    # Save cleaned data to MongoDB
    provider.save_data(cleaned_data, COLLECTION_CLEANED)
    
    # --- Stage 4: EDA ---
    print("\n--- Giai đoạn 4: EDA (Exploratory Data Analysis) ---")
    visualizer = EDAVisualizer()
    visualizer.plot_price_trend(cleaned_data, TICKER)
    visualizer.plot_volume_trend(cleaned_data, TICKER)
    visualizer.plot_correlation_heatmap(cleaned_data)

    # --- Stage 5: Feature Engineering ---
    print("\n--- Giai đoạn 5: Feature Engineering ---")
    engineer = FeatureEngineer()
    featured_data = engineer.process_all_features(cleaned_data)
    
    # Save final features for modeling to MongoDB
    provider.save_data(featured_data, COLLECTION_FEATURES)
    
    # --- Stage 6 & 7: Model Training & Evaluation ---
    print("\n--- Giai đoạn 6 & 7: Model Training & Evaluation ---")
    # Load featured data from MongoDB
    df_for_training = provider.load_data(COLLECTION_FEATURES)
    
    trainer = ModelTrainer()
    
    # Prepare data
    X_train, X_test, y_train, y_test = trainer.prepare_data(df_for_training, target_col='close')
    
    # 1. Linear Regression
    trainer.train_linear_regression(X_train, y_train)
    trainer.evaluate_model('LinearRegression', X_test, y_test)
    
    # 2. XGBoost
    trainer.train_xgboost(X_train, y_train)
    trainer.evaluate_model('XGBoost', X_test, y_test)
    
    # 3. LightGBM
    trainer.train_lightgbm(X_train, y_train)
    trainer.evaluate_model('LightGBM', X_test, y_test)
    
    # 4. CatBoost
    trainer.train_catboost(X_train, y_train)
    trainer.evaluate_model('CatBoost', X_test, y_test)
    
    # 5. ARIMA + LSTM Hybrid
    trainer.train_arima_lstm(X_train, y_train)
    trainer.evaluate_model('ARIMA_LSTM', X_test, y_test)

    print("\n[SUCCESS] Stages 1-7 are completed successfully.")

if __name__ == "__main__":
    run_pipeline()
