import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import time

print("[*] TensorFlow version:", tf.__version__)

# Force CPU only
tf.config.set_visible_devices([], 'GPU')

X = np.random.randn(100, 10, 1)
y = np.random.randn(100, 1)

print("[*] Creating model...")
model = Sequential([
    LSTM(16, activation='relu', input_shape=(10, 1)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')

print("[*] Training started...")
start = time.time()
model.fit(X, y, epochs=5, batch_size=32, verbose=1)
end = time.time()
print(f"[+] Training finished in {end-start:.2f} seconds!")
