import numpy as np
import pandas as pd
import time
from fnn import microns

print("Starting FNN model test...")

# Time the model loading process
print("Loading model...")
start_time = time.time()
model, ids = microns.scan(session=8, scan_idx=5, verbose=True)
load_time = time.time() - start_time
print(f"Model loaded in {load_time:.2f} seconds")

# Print basic model information
print(f"Model type: {type(model)}")
print(f"Number of neurons in the model: {len(ids)}")

# Create a simple test stimulus (black-gray-white sequence)
print("Creating test stimulus...")
frames = np.concatenate([
    np.full(shape=[30, 144, 256], dtype="uint8", fill_value=0),   # 1 second of black
    np.full(shape=[30, 144, 256], dtype="uint8", fill_value=128), # 1 second of gray
    np.full(shape=[30, 144, 256], dtype="uint8", fill_value=255), # 1 second of white
])
print(f"Stimulus shape: {frames.shape}")

# Predict neural responses
print("Predicting neural responses...")
start_time = time.time()
responses = model.predict(stimuli=frames)
predict_time = time.time() - start_time
print(f"Prediction completed in {predict_time:.2f} seconds")

# Print response information
print(f"Response shape: {responses.shape}")
print(f"Response summary statistics:")
print(f"  Min: {responses.min():.4f}")
print(f"  Max: {responses.max():.4f}")
print(f"  Mean: {responses.mean():.4f}")
print(f"  Std: {responses.std():.4f}")

# Print neuron ID information
if isinstance(ids, pd.DataFrame):
    print(f"Neuron ID dataframe columns: {ids.columns.tolist()}")
    print(f"First 5 neuron IDs:\n{ids.head(5)}")
else:
    print(f"Neuron ID type: {type(ids)}")

print("Test completed successfully!")
