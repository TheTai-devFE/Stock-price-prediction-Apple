import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import pandas as pd
import numpy as np
from pymongo import MongoClient
from statsmodels.tsa.arima.model import ARIMA
import tensorflow as tf

tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import time

print("[*] Debugging Hybrid Model training process...")

# 1. Connect to MongoDB and fetch features data
client = MongoClient("mongodb://localhost:27017/")
db = client["apple_stock_db"]
collection = db["aapl_features"]

cursor = collection.find({}, {"_id": 0})
df = pd.DataFrame(list(cursor))
if df.empty:
    print("[!] MongoDB collection 'aapl_features' is empty!")
    exit(1)

# Sort by Date if available
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
elif 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

# Select closing prices
if 'close' in df.columns:
    y = df['close']
else:
    print("[!] No 'close' column found!")
    exit(1)

# 2. Train-test split
train_size = int(len(y) * 0.8)
y_train = y.iloc[:train_size]
y_test = y.iloc[train_size:]

print(f"[*] Data loaded: Total: {len(y)}, Train: {len(y_train)}, Test: {len(y_test)}")

# 3. Fit ARIMA
print("[*] Fitting ARIMA(1, 1, 1)...")
start_arima = time.time()
arima_model = ARIMA(y_train, order=(1, 1, 1))
arima_fitted = arima_model.fit()
end_arima = time.time()
print(f"[+] ARIMA fitted in {end_arima - start_arima:.2f} seconds!")

# 4. Get residuals
train_residuals = arima_fitted.resid
train_residuals = train_residuals[~np.isnan(train_residuals)]
print(f"[*] Residuals shape: {train_residuals.shape}")

# 5. Create Sequences
seq_length = 10
X_seq, y_seq = [], []
res_values = train_residuals.values if hasattr(train_residuals, 'values') else train_residuals
for i in range(len(res_values) - seq_length):
    X_seq.append(res_values[i:(i + seq_length)])
    y_seq.append(res_values[i + seq_length])

X_seq = np.array(X_seq)
y_seq = np.array(y_seq)
X_seq = np.reshape(X_seq, (X_seq.shape[0], X_seq.shape[1], 1))

# Slice to first 100 samples for diagnostic testing
X_seq = X_seq[:100]
y_seq = y_seq[:100]

print(f"[*] Sliced LSTM Input Shape: {X_seq.shape}, Output Shape: {y_seq.shape}")
print(f"[*] X_seq stats - Min: {X_seq.min():.4f}, Max: {X_seq.max():.4f}, Mean: {X_seq.mean():.4f}, NaN count: {np.isnan(X_seq).sum()}")
print(f"[*] y_seq stats - Min: {y_seq.min():.4f}, Max: {y_seq.max():.4f}, Mean: {y_seq.mean():.4f}, NaN count: {np.isnan(y_seq).sum()}")

# 6. Fit LSTM
tf.config.set_visible_devices([], 'GPU')
model = Sequential([
    tf.keras.Input(shape=(seq_length, 1)) if hasattr(tf.keras, 'Input') else tf.keras.layers.InputLayer(input_shape=(seq_length, 1)),
    LSTM(16, activation='tanh'),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse', run_eagerly=True)

print("[*] Starting LSTM training (8 epochs, batch_size=512)...")
start_lstm = time.time()
model.fit(X_seq, y_seq, epochs=8, batch_size=512, verbose=1)
end_lstm = time.time()
print(f"[+] LSTM trained in {end_lstm - start_lstm:.2f} seconds!")
