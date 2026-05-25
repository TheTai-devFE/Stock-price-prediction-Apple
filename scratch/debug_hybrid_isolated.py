import pandas as pd
import numpy as np
from pymongo import MongoClient
from statsmodels.tsa.arima.model import ARIMA
import time
import subprocess
import os

print("[*] Starting Isolated Hybrid Model diagnostic pipeline...")

# 1. Connect to MongoDB and fetch features
client = MongoClient("mongodb://localhost:27017/")
db = client["apple_stock_db"]
collection = db["aapl_features"]
cursor = collection.find({}, {"_id": 0})
df = pd.DataFrame(list(cursor))

if df.empty:
    print("[!] MongoDB collection 'aapl_features' is empty!")
    exit(1)

if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
elif 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

y = df['close']
train_size = int(len(y) * 0.8)
y_train = y.iloc[:train_size]

# 2. Fit ARIMA in main process
print("[*] Main Process: Fitting ARIMA(1, 1, 1)...")
start_arima = time.time()
arima_model = ARIMA(y_train, order=(1, 1, 1))
arima_fitted = arima_model.fit()
train_residuals = arima_fitted.resid
train_residuals = train_residuals[~np.isnan(train_residuals)]
end_arima = time.time()
print(f"[+] Main Process: ARIMA fitted in {end_arima - start_arima:.2f} seconds!")

# Save residuals to temporary npy file
os.makedirs("scratch", exist_ok=True)
residuals_path = "scratch/residuals.npy"
np.save(residuals_path, train_residuals.values)
print(f"[*] Main Process: Saved residuals to {residuals_path}")

# 3. Launch LSTM training in isolated subprocess
print("[*] Main Process: Launching LSTM training in clean subprocess to avoid OpenMP deadlock...")
start_lstm = time.time()
result = subprocess.run(["venv/bin/python3", "scratch/train_lstm_helper.py"], capture_output=True, text=True)
end_lstm = time.time()

print("\n--- Subprocess Output ---")
print(result.stdout)
if result.stderr:
    print("[!] Subprocess Standard Error:")
    print(result.stderr)
print("-------------------------\n")

if result.returncode == 0:
    print(f"[+] Diagnostic complete: Hybrid Model successfully trained with Process Isolation in {end_lstm - start_arima:.2f} seconds total!")
else:
    print(f"[!] Subprocess exited with non-zero code: {result.returncode}")
