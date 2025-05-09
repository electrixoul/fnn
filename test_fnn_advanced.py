import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from fnn import microns

print("Starting FNN advanced model testing...")

# Time the model loading process
print("Loading model...")
start_time = time.time()
model, ids = microns.scan(session=8, scan_idx=5, verbose=True)
load_time = time.time() - start_time
print(f"Model loaded in {load_time:.2f} seconds")

# Print basic model information
print(f"Model type: {type(model)}")
print(f"Number of neurons in the model: {len(ids)}")

# Create different test stimuli to showcase the model's ability to generalize
print("\nCreating test stimuli...")

# 1. Black-gray-white sequence (as before)
frames_basic = np.concatenate([
    np.full(shape=[30, 144, 256], dtype="uint8", fill_value=0),    # 1 second of black
    np.full(shape=[30, 144, 256], dtype="uint8", fill_value=128),  # 1 second of gray
    np.full(shape=[30, 144, 256], dtype="uint8", fill_value=255),  # 1 second of white
])

# 2. Moving bar stimulus (mimicking a Gabor-like pattern)
bar_frames = []
for i in range(60):
    frame = np.zeros((144, 256), dtype=np.uint8)
    # Create a vertical bar that moves from left to right
    bar_pos = int((i / 60) * 256)
    bar_width = 20
    if bar_pos < 256:
        frame[:, max(0, bar_pos-bar_width//2):min(256, bar_pos+bar_width//2)] = 255
    bar_frames.append(frame)
frames_bar = np.array(bar_frames)

# 3. Circular pattern (mimicking expanding/contracting stimulus)
circle_frames = []
for i in range(60):
    frame = np.zeros((144, 256), dtype=np.uint8)
    # Create a circle that expands and contracts
    radius = int(20 + 40 * np.sin(i / 60 * 2 * np.pi))
    cy, cx = 72, 128  # center
    y, x = np.ogrid[-cy:144-cy, -cx:256-cx]
    mask = x*x + y*y <= radius*radius
    frame[mask] = 255
    circle_frames.append(frame)
frames_circle = np.array(circle_frames)

print(f"Basic stimulus shape: {frames_basic.shape}")
print(f"Moving bar stimulus shape: {frames_bar.shape}")
print(f"Circle stimulus shape: {frames_circle.shape}")

# Run predictions for each stimulus type
print("\nPredicting neural responses to different stimuli...")

start_time = time.time()
responses_basic = model.predict(stimuli=frames_basic)
bar_time = time.time() - start_time
print(f"Basic stimulus prediction completed in {bar_time:.2f} seconds")

start_time = time.time()
responses_bar = model.predict(stimuli=frames_bar)
bar_time = time.time() - start_time
print(f"Bar stimulus prediction completed in {bar_time:.2f} seconds")

start_time = time.time()
responses_circle = model.predict(stimuli=frames_circle)
circle_time = time.time() - start_time
print(f"Circle stimulus prediction completed in {circle_time:.2f} seconds")

# Print response information
print("\nResponse summary statistics:")
print(f"Basic stimulus: Mean: {responses_basic.mean():.4f}, Std: {responses_basic.std():.4f}")
print(f"Bar stimulus: Mean: {responses_bar.mean():.4f}, Std: {responses_bar.std():.4f}")
print(f"Circle stimulus: Mean: {responses_circle.mean():.4f}, Std: {responses_circle.std():.4f}")

# Select a few neurons to visualize their responses
selected_neurons = [100, 500, 1000, 2000, 3000]
print(f"\nVisualizing responses for selected neurons: {selected_neurons}")

# Create a figure to visualize the responses
fig, axes = plt.subplots(len(selected_neurons), 3, figsize=(15, 3*len(selected_neurons)))
fig.tight_layout(pad=3.0)

stimulus_names = ["Black-Gray-White", "Moving Bar", "Expanding Circle"]
responses_list = [responses_basic, responses_bar, responses_circle]

for i, neuron_idx in enumerate(selected_neurons):
    for j, (stim_name, responses) in enumerate(zip(stimulus_names, responses_list)):
        axes[i, j].plot(responses[:, neuron_idx])
        axes[i, j].set_title(f"Neuron {neuron_idx} - {stim_name}")
        axes[i, j].set_xlabel("Time (frames)")
        axes[i, j].set_ylabel("Activity")
        
plt.savefig("fnn_responses.png")
print("Visualizations saved to 'fnn_responses.png'")

# Extract and analyze readout parameters
print("\nExtracting model readout parameters...")

# Access readout parameters (the functional 'barcode' from the paper)
if hasattr(model, 'readout') and hasattr(model.readout, 'feature'):
    readout_params = model.readout

    # Check if we can access the readout positions (reflecting receptive field positions)
    if hasattr(readout_params, 'position') and hasattr(readout_params.position, 'mean'):
        positions = readout_params.position.mean
        print(f"Readout position shape: {positions.shape}")
        
        # Visualize the spatial distribution of receptive fields
        plt.figure(figsize=(10, 8))
        rf_positions = positions.detach().cpu().numpy()
        if len(rf_positions.shape) > 1 and rf_positions.shape[1] >= 2:
            plt.scatter(rf_positions[:, 0], rf_positions[:, 1], s=3, alpha=0.5)
            plt.title("Spatial Distribution of Receptive Fields")
            plt.xlabel("X Position")
            plt.ylabel("Y Position")
            plt.savefig("receptive_fields.png")
            print("Receptive field visualization saved to 'receptive_fields.png'")
        else:
            print("Readout positions not in expected format for visualization")
    else:
        print("Could not access readout positions")

print("\nTest completed successfully!")
