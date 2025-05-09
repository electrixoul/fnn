import numpy as np
import matplotlib.pyplot as plt
import sys

def visualize_receptive_field(sta, neuron_idx, file_path=None):
    """
    Visualize the receptive field and save to file.
    
    Args:
        sta: Spike-triggered average image
        neuron_idx: Index of the analyzed neuron
        file_path: Path to save the visualization
    """
    plt.figure(figsize=(10, 6))
    
    # Normalize STA for better visualization
    sta_normalized = (sta - np.mean(sta)) / np.std(sta)
    
    # Plot the receptive field
    plt.imshow(sta_normalized, cmap='coolwarm')
    plt.colorbar(label='Normalized STA Value')
    plt.title(f'Receptive Field Feature Image for Neuron {neuron_idx}')
    plt.xlabel('Horizontal Position (pixels)')
    plt.ylabel('Vertical Position (pixels)')
    
    if file_path:
        plt.savefig(file_path)
    
    plt.show()

# Load the results
print("Loading results...")
results = np.load('receptive_fields/all_results.npy', allow_pickle=True).item()
print(f"Loaded {len(results)} neuron results")

# Get list of processed neurons
neurons = list(results.keys())
print(f"Available neurons: {neurons[:10]}...")

# Either show the specified neuron or the first one
if len(sys.argv) > 1:
    neuron_idx = int(sys.argv[1])
    if neuron_idx in results:
        print(f"Visualizing neuron {neuron_idx}")
        visualize_receptive_field(results[neuron_idx], neuron_idx)
    else:
        print(f"Neuron {neuron_idx} not found in results")
        print(f"Available neurons: {neurons[:10]}...")
else:
    # Show the first neuron
    neuron_idx = neurons[0]
    print(f"Visualizing first neuron: {neuron_idx}")
    visualize_receptive_field(results[neuron_idx], neuron_idx)

# Save a sample to disk
output_path = f'receptive_fields/quick_sample_{neuron_idx}.png'
visualize_receptive_field(results[neuron_idx], neuron_idx, output_path)
print(f"Sample saved to {output_path}")
