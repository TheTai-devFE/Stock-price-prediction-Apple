import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load residuals
residuals_path = "scratch/residuals.npy"
if not os.path.exists(residuals_path):
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

# Build and train LSTM
tf.config.set_visible_devices([], 'GPU')
model = Sequential([
    tf.keras.layers.Input(shape=(seq_length, 1)),
    LSTM(16, activation='tanh'),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X_seq, y_seq, epochs=8, batch_size=512, verbose=0)

# Save trained model
model.save("scratch/lstm_model.keras")
