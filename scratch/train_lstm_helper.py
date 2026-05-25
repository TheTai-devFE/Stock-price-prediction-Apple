import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import time

print("[*] Subprocess: Khoi tao va bat dau huan luyen LSTM tren residuals...")

# 1. Load residuals
residuals_path = "scratch/residuals.npy"
if not os.path.exists(residuals_path):
    print(f"[!] Khong tim thay file residuals tai {residuals_path}!")
    exit(1)
    
res_values = np.load(residuals_path)
seq_length = 10
X_seq, y_seq = [], []
for i in range(len(res_values) - seq_length):
    X_seq.append(res_values[i:(i + seq_length)])
    y_seq.append(res_values[i + seq_length])

X_seq = np.array(X_seq)
y_seq = np.array(y_seq)
X_seq = np.reshape(X_seq, (X_seq.shape[0], X_seq.shape[1], 1))

print(f"[*] Subprocess: Dataset LSTM shape: {X_seq.shape}")

# 2. Xay dung va huan luyen model LSTM
tf.config.set_visible_devices([], 'GPU')
model = Sequential([
    tf.keras.layers.Input(shape=(seq_length, 1)),
    LSTM(16, activation='tanh'),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')

start = time.time()
model.fit(X_seq, y_seq, epochs=8, batch_size=512, verbose=1)
end = time.time()

# 3. Save model
model.save("scratch/lstm_model.keras")
print(f"[+] Subprocess: LSTM trained in {end-start:.2f} seconds and saved successfully!")
