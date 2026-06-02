import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.models import (
    LinearRegressionModel,
    XGBoostModel,
    LightGBMModel,
    CatBoostModel,
    ArimaLstmModel
)

class ModelTrainer:
    """
    Orchestrator for Model Training & Evaluation.
    Maintains backwards compatibility while referencing modularized model components.
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

        leakage_cols = ['open', 'high', 'low', 'adj_close']
        cols_to_drop = [c for c in leakage_cols if c in df.columns] + [target_col]
        X = df.drop(columns=cols_to_drop)
        # Sort columns alphabetically to ensure identical feature ordering across pipelines
        X = X.reindex(sorted(X.columns), axis=1)
        
        # Target transformation: Price Difference (y_t = close_t - close_{t-1})
        y = df[target_col] - df['lag_1']

        # Time-series split (take the last 20% for testing)
        split_idx = int(len(df) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        return X_train, X_test, y_train, y_test

    def train_linear_regression(self, X_train, y_train):
        model = LinearRegressionModel()
        model.fit(X_train, y_train)
        self.models['LinearRegression'] = model
        return model

    def train_xgboost(self, X_train, y_train):
        # Time-series validation split (last 10% of train data for early stopping)
        split_idx = int(len(X_train) * 0.9)
        X_tr, X_val = X_train.iloc[:split_idx], X_train.iloc[split_idx:]
        y_tr, y_val = y_train.iloc[:split_idx], y_train.iloc[split_idx:]
        
        model = XGBoostModel()
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], early_stopping_rounds=30)
        self.models['XGBoost'] = model
        return model

    def train_lightgbm(self, X_train, y_train):
        split_idx = int(len(X_train) * 0.9)
        X_tr, X_val = X_train.iloc[:split_idx], X_train.iloc[split_idx:]
        y_tr, y_val = y_train.iloc[:split_idx], y_train.iloc[split_idx:]
        
        model = LightGBMModel()
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], early_stopping_rounds=30)
        self.models['LightGBM'] = model
        return model

    def train_catboost(self, X_train, y_train):
        split_idx = int(len(X_train) * 0.9)
        X_tr, X_val = X_train.iloc[:split_idx], X_train.iloc[split_idx:]
        y_tr, y_val = y_train.iloc[:split_idx], y_train.iloc[split_idx:]
        
        model = CatBoostModel()
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], early_stopping_rounds=30)
        self.models['CatBoost'] = model
        return model

    def train_arima_lstm(self, X_train, y_train):
        model = ArimaLstmModel()
        model.fit(X_train, y_train)
        self.models['ARIMA_LSTM'] = model
        return model

    def train_all(self, X_train, y_train):
        """
        Train all 5 models sequentially.
        """
        print("\n--- Training All Models ---")
        self.train_linear_regression(X_train, y_train)
        self.train_xgboost(X_train, y_train)
        self.train_lightgbm(X_train, y_train)
        self.train_catboost(X_train, y_train)
        self.train_arima_lstm(X_train, y_train)
        print("[+] All models trained successfully.")

    def evaluate_model(self, model_name: str, X_test, y_test):
        """
        Evaluate model and store results.
        """
        if model_name not in self.models:
            print(f"[!] Model {model_name} not found. Fitting standard version...")
            if model_name == 'LinearRegression':
                # Automatic training if not already trained
                raise ValueError("Model must be trained before calling evaluate_model.")
            
        model = self.models[model_name]
        
        # Perform evaluation using the model class method
        eval_res = model.evaluate(X_test, y_test)
        
        predictions_diff = eval_res['predictions']
        
        # Reconstruct absolute price from price difference
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        # X_test contains 'lag_1' (yesterday's close price)
        lag_1_values = X_test['lag_1'].values
        
        predictions_abs = predictions_diff + lag_1_values
        y_test_abs = y_test.values + lag_1_values
        
        mse = mean_squared_error(y_test_abs, predictions_abs)
        mae = mean_absolute_error(y_test_abs, predictions_abs)
        r2 = r2_score(y_test_abs, predictions_abs)

        print(f"\n--- Evaluation Results for {model_name} (Reconstructed Absolute Price) ---")
        print(f"MSE: {mse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R2 Score: {r2:.4f}")

        self.results[model_name] = {
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'predictions': predictions_abs
        }

        # Plot Actual vs Predicted (Using absolute prices)
        # For plot labels consistency, we convert y_test_abs back to pandas Series
        y_test_abs_series = pd.Series(y_test_abs, index=y_test.index)
        self.plot_predictions(model_name, y_test_abs_series, predictions_abs)
        return self.results[model_name]

    def evaluate_all(self, X_test, y_test) -> pd.DataFrame:
        """
        Evaluate all trained models and return a summary DataFrame of the metrics.
        """
        print("\n--- Evaluating All Models ---")
        metrics_list = []
        for name in self.models.keys():
            res = self.evaluate_model(name, X_test, y_test)
            metrics_list.append({
                "Model": name,
                "MSE": res['mse'],
                "MAE": res['mae'],
                "R2 Score": res['r2']
            })
        
        df_metrics = pd.DataFrame(metrics_list)
        return df_metrics

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

    def save_metrics_report(self, df_metrics: pd.DataFrame, filename: str = "latest_metrics.md"):
        """
        Save the evaluation metrics of all models to a Markdown file.
        """
        filepath = os.path.join(self.report_path, filename)
        
        # Ensure the directory exists
        os.makedirs(self.report_path, exist_ok=True)
        
        # Write markdown contents
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("# 🍎 BÁO CÁO CẬP NHẬT ĐỘ ĐO ĐÁNH GIÁ MÔ HÌNH MỚI NHẤT\n\n")
            f.write(f"Thời gian ghi nhận: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("Dưới đây là các chỉ số đánh giá độ chính xác thu được từ lượt huấn luyện mới nhất:\n\n")
            
            # Format DataFrame as Markdown table
            df_formatted = df_metrics.copy()
            df_formatted['MSE'] = df_formatted['MSE'].map('{:,.6f}'.format)
            df_formatted['MAE'] = df_formatted['MAE'].map('{:,.6f}'.format)
            df_formatted['R2 Score'] = df_formatted['R2 Score'].map('{:,.6f}'.format)
            
            # Generate markdown table manually to avoid 'tabulate' dependency
            headers = list(df_formatted.columns)
            markdown_table = []
            markdown_table.append("| " + " | ".join(headers) + " |")
            markdown_table.append("| " + " | ".join(["---"] * len(headers)) + " |")
            for _, row in df_formatted.iterrows():
                markdown_table.append("| " + " | ".join([str(val) for val in row]) + " |")
            f.write("\n".join(markdown_table))
            f.write("\n\n---\n*Báo cáo được xuất tự động bởi hệ thống ModelTrainer.*")
        print(f"[+] Saved latest metrics report to: {filepath}")

