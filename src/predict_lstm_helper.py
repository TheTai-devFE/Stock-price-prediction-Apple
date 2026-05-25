import os
import numpy as np
import tensorflow as tf

# Load sequences
input_path = "scratch/X_seq_test.npy"
if not os.path.exists(input_path):
    exit(1)
    
X_seq_test = np.load(input_path)

# Load trained model
model_path = "scratch/lstm_model.keras"
if not os.path.exists(model_path):
    exit(1)
    
tf.config.set_visible_devices([], 'GPU')
model = tf.keras.models.load_model(model_path)

# Make prediction
preds = model.predict(X_seq_test, verbose=0).flatten()

# Save predicted residuals
np.save("scratch/lstm_test_preds.npy", preds)
