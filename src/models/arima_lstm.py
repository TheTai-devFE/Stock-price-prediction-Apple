import os
import subprocess
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from src.models.base_model import BaseModel

# Suppress TF logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

class ArimaLstmModel(BaseModel):
    """
    ARIMA + LSTM Hybrid model for time series forecasting.
    First models the linear patterns using ARIMA, then models the residuals using LSTM.
    """

    def __init__(self):
        super().__init__("ARIMA_LSTM")
        self.model = {
            'arima': None,
            'lstm': None
        }
        self.scaler = MinMaxScaler(feature_range=(-1, 1))

    def __getitem__(self, key):
        """
        Support dictionary-like access for backwards compatibility.
        """
        return self.model[key]

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        print("[*] Training ARIMA + LSTM Hybrid model...")
        
        # 1. Fit ARIMA (1, 0, 1) since the target y_train is already differenced (Price Difference)
        print("[*] Fitting ARIMA(1, 0, 1) part...")
        arima_model = ARIMA(y_train, order=(1, 0, 1))
        arima_fitted = arima_model.fit()
        
        # Get in-sample residuals directly using statsmodels resid attribute
        train_residuals = arima_fitted.resid
        # Drop any NaNs
        train_residuals = train_residuals[~np.isnan(train_residuals)]
        
        # Scale residuals to [-1, 1] using MinMaxScaler
        residuals_reshaped = train_residuals.values.reshape(-1, 1)
        scaled_residuals = self.scaler.fit_transform(residuals_reshaped).flatten()
        
        # Save scaled residuals to temporary npy file for subprocess training
        os.makedirs("scratch", exist_ok=True)
        np.save("scratch/residuals.npy", scaled_residuals)
        
        # 2. Run LSTM in an isolated subprocess to avoid thread deadlocks with OpenMP
        print("[*] Fitting LSTM part in clean subprocess to avoid OpenMP deadlock...")
        result = subprocess.run(["venv/bin/python3", "src/train_lstm_helper.py"], capture_output=True, text=True)
        if result.returncode != 0:
            print("[!] Subprocess failed!")
            print(result.stderr)
            raise RuntimeError("LSTM training subprocess failed!")
            
        # Load trained Keras model
        lstm_model = tf.keras.models.load_model("scratch/lstm_model.keras")
        
        self.model = {
            'arima': arima_fitted,
            'lstm': lstm_model
        }
        print("[+] ARIMA + LSTM Hybrid model trained successfully.")
        return self

    def predict(self, X: pd.DataFrame, y: pd.Series = None) -> np.ndarray:
        if self.model['arima'] is None:
            raise RuntimeError("Model must be fitted before prediction.")
            
        arima_model = self.model['arima']
        n_steps = len(X)
        
        # Predict ARIMA part
        start_idx = len(arima_model.fittedvalues)
        end_idx = start_idx + n_steps - 1
        arima_preds = arima_model.predict(start=start_idx, end=end_idx).values
        
        # Get historical residuals for LSTM context
        train_residuals = arima_model.resid
        train_residuals = train_residuals[~np.isnan(train_residuals)]
        
        # Scale train residuals
        res_values = train_residuals.values if hasattr(train_residuals, 'values') else train_residuals
        scaled_res_values = self.scaler.transform(res_values.reshape(-1, 1)).flatten()
        
        # Determine LSTM sequence input
        if y is not None:
            # If target y is provided, calculate actual residuals for sequence inputs (evaluation mode)
            y_values = y.values if hasattr(y, 'values') else y
            test_residuals = y_values - arima_preds
            scaled_test_residuals = self.scaler.transform(test_residuals.reshape(-1, 1)).flatten()
            
            full_residuals = np.concatenate([scaled_res_values[-10:], scaled_test_residuals])
            
            X_seq_test = []
            for i in range(len(full_residuals) - 10):
                X_seq_test.append(full_residuals[i:(i + 10)])
            X_seq_test = np.array(X_seq_test)
            X_seq_test = np.reshape(X_seq_test, (X_seq_test.shape[0], X_seq_test.shape[1], 1))
        else:
            # If target y is not provided, perform self-rolling prediction of residuals
            current_seq = list(scaled_res_values[-10:])
            lstm_preds = []
            for _ in range(n_steps):
                X_seq_step = np.array(current_seq[-10:]).reshape(1, 10, 1)
                os.makedirs("scratch", exist_ok=True)
                np.save("scratch/X_seq_test.npy", X_seq_step)
                
                # Execute LSTM prediction in safe subprocess
                subprocess.run(["venv/bin/python3", "src/predict_lstm_helper.py"], capture_output=True, text=True)
                step_pred = np.load("scratch/lstm_test_preds.npy")[0]
                lstm_preds.append(step_pred)
                current_seq.append(step_pred)
                
            lstm_preds_unscaled = self.scaler.inverse_transform(np.array(lstm_preds).reshape(-1, 1)).flatten()
            return arima_preds + lstm_preds_unscaled
            
        # Save sequence inputs for safe subprocess prediction
        os.makedirs("scratch", exist_ok=True)
        np.save("scratch/X_seq_test.npy", X_seq_test)
        
        # Execute LSTM prediction in safe subprocess
        result = subprocess.run(["venv/bin/python3", "src/predict_lstm_helper.py"], capture_output=True, text=True)
        if result.returncode != 0:
            print("[!] Prediction subprocess failed!")
            print(result.stderr)
            raise RuntimeError("LSTM prediction subprocess failed!")
            
        # Load predicted scaled residuals and inverse scale them
        lstm_test_preds_scaled = np.load("scratch/lstm_test_preds.npy")
        lstm_test_preds = self.scaler.inverse_transform(lstm_test_preds_scaled.reshape(-1, 1)).flatten()
        
        predictions = arima_preds + lstm_test_preds
        return predictions
