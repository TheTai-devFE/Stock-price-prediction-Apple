import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
from statsmodels.tsa.arima.model import ARIMA
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Suppress TF logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

class ModelTrainer:
    """
    Class for Stage 6 & 7: Model Training & Evaluation.
    Supports Linear Regression and XGBoost for stock price prediction.
    """

    def __init__(self, report_path: str = "reports/models"):
        self.report_path = report_path
        os.makedirs(self.report_path, exist_ok=True)
        self.models = {}
        self.results = {}

    def prepare_data(self, df: pd.DataFrame, target_col: str = 'close', test_size: float = 0.2):
        """
        Prepare features and target for training.
        Uses time-series splitting (no shuffle).
        """
        print(f"[*] Preparing data for training. Target: {target_col}")
        
        # Sort by date if 'date' column exists
        if 'date' in df.columns:
            df = df.sort_values('date')
            df = df.drop(columns=['date'])

        X = df.drop(columns=[target_col])
        y = df[target_col]

        # Time-series split (take the last 20% for testing)
        split_idx = int(len(df) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        return X_train, X_test, y_train, y_test

    def train_linear_regression(self, X_train, y_train):
        print("[*] Training Linear Regression model...")
        model = LinearRegression()
        model.fit(X_train, y_train)
        self.models['LinearRegression'] = model
        return model

    def train_xgboost(self, X_train, y_train):
        print("[*] Training XGBoost model...")
        model = XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=5, n_jobs=-1)
        model.fit(X_train, y_train)
        self.models['XGBoost'] = model
        return model

    def train_lightgbm(self, X_train, y_train):
        print("[*] Training LightGBM model...")
        model = LGBMRegressor(n_estimators=1000, learning_rate=0.05, max_depth=5, n_jobs=-1, random_state=42, verbose=-1)
        model.fit(X_train, y_train)
        self.models['LightGBM'] = model
        return model

    def train_catboost(self, X_train, y_train):
        print("[*] Training CatBoost model...")
        model = CatBoostRegressor(iterations=1000, learning_rate=0.05, depth=5, random_seed=42, verbose=0)
        model.fit(X_train, y_train)
        self.models['CatBoost'] = model
        return model

    def train_arima_lstm(self, X_train, y_train):
        print("[*] Training ARIMA + LSTM Hybrid model...")
        
        # 1. Fit ARIMA (1, 1, 1) on closing prices
        print("[*] Fitting ARIMA(1, 1, 1) part...")
        arima_model = ARIMA(y_train, order=(1, 1, 1))
        arima_fitted = arima_model.fit()
        
        # Get in-sample residuals directly using statsmodels resid attribute
        train_residuals = arima_fitted.resid
        # Drop any NaNs
        train_residuals = train_residuals[~np.isnan(train_residuals)]
        
        # Save residuals to temporary npy file for subprocess
        os.makedirs("scratch", exist_ok=True)
        np.save("scratch/residuals.npy", train_residuals.values)
        
        # 2. Run LSTM in an isolated subprocess to avoid thread deadlocks
        print("[*] Fitting LSTM part in clean subprocess to avoid OpenMP deadlock...")
        import subprocess
        result = subprocess.run(["venv/bin/python3", "src/train_lstm_helper.py"], capture_output=True, text=True)
        if result.returncode != 0:
            print("[!] Subprocess failed!")
            print(result.stderr)
            raise RuntimeError("LSTM training subprocess failed!")
            
        # Load trained Keras model
        lstm_model = tf.keras.models.load_model("scratch/lstm_model.keras")
        
        self.models['ARIMA_LSTM'] = {
            'arima': arima_fitted,
            'lstm': lstm_model
        }
        print("[+] ARIMA + LSTM Hybrid model trained successfully.")
        return self.models['ARIMA_LSTM']

    def evaluate_model(self, model_name: str, X_test, y_test):
        """
        Evaluate model and store results.
        """
        if model_name not in self.models:
            print(f"[!] Model {model_name} not found.")
            return

        model = self.models[model_name]
        if model_name == 'ARIMA_LSTM':
            arima_model = model['arima']
            lstm_model = model['lstm']
            
            # Predict ARIMA part
            arima_test_preds = arima_model.predict(start=len(arima_model.fittedvalues), end=len(arima_model.fittedvalues) + len(y_test) - 1).values
            
            # Calculate rolling input sequences for LSTM using actual residuals
            train_residuals = arima_model.resid
            train_residuals = train_residuals[~np.isnan(train_residuals)]
            
            res_values = train_residuals.values if hasattr(train_residuals, 'values') else train_residuals
            full_residuals = np.concatenate([res_values[-10:], (y_test.values - arima_test_preds)])
            
            X_seq_test = []
            for i in range(len(full_residuals) - 10):
                X_seq_test.append(full_residuals[i:(i + 10)])
            X_seq_test = np.array(X_seq_test)
            X_seq_test = np.reshape(X_seq_test, (X_seq_test.shape[0], X_seq_test.shape[1], 1))
            
            # Save test sequences to temporary npy file for subprocess
            np.save("scratch/X_seq_test.npy", X_seq_test)
            
            # Predict in an isolated subprocess to avoid thread deadlocks
            import subprocess
            result = subprocess.run(["venv/bin/python3", "src/predict_lstm_helper.py"], capture_output=True, text=True)
            if result.returncode != 0:
                print("[!] Prediction subprocess failed!")
                print(result.stderr)
                raise RuntimeError("LSTM prediction subprocess failed!")
                
            # Load predicted residuals
            lstm_test_preds = np.load("scratch/lstm_test_preds.npy")
            predictions = arima_test_preds + lstm_test_preds
        else:
            predictions = model.predict(X_test)

        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        print(f"\n--- Evaluation Results for {model_name} ---")
        print(f"MSE: {mse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R2 Score: {r2:.4f}")

        self.results[model_name] = {
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'predictions': predictions
        }

        # Plot Actual vs Predicted
        self.plot_predictions(model_name, y_test, predictions)
        return self.results[model_name]

    def plot_predictions(self, model_name: str, y_test, predictions):
        plt.figure(figsize=(12, 6))
        plt.plot(y_test.values, label='Actual Price', color='blue', alpha=0.7)
        plt.plot(predictions, label='Predicted Price', color='red', alpha=0.7, linestyle='--')
        plt.title(f'Actual vs Predicted Stock Price - {model_name}')
        plt.xlabel('Time Steps (Test Data)')
        plt.ylabel('Price (USD)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        save_path = os.path.join(self.report_path, f"{model_name.lower()}_prediction.png")
        plt.savefig(save_path)
        print(f"[*] Saved prediction plot to: {save_path}")
        plt.close()
