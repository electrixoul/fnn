import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from fnn import microns

def create_checkerboard(frames=60, size=(144, 256), square_size=16, alternate=True):
    """
    Create a checkerboard stimulus with optional alternating pattern over time.
    
    Args:
        frames: Number of frames to generate
        size: Size of each frame (height, width)
        square_size: Size of each checkerboard square
        alternate: Whether to alternate the pattern on each frame
        
    Returns:
        Numpy array of shape [frames, height, width] with checkerboard pattern
    """
    h, w = size
    pattern = np.zeros((frames, h, w), dtype=np.uint8)
    
    for f in range(frames):
        # Create base checkerboard
        phase_offset = f % 2 if alternate else 0
        for i in range(0, h, square_size):
            for j in range(0, w, square_size):
                if ((i // square_size) + (j // square_size) + phase_offset) % 2 == 0:
                    pattern[f, i:i+square_size, j:j+square_size] = 255
    
    return pattern

def create_radial_grating(frames=60, size=(144, 256), center=None, 
                          max_radius=None, num_segments=8, rotation=True):
    """
    Create a radial grating pattern that can rotate over time.
    
    Args:
        frames: Number of frames to generate
        size: Size of each frame (height, width)
        center: Center of the radial pattern (defaults to image center)
        max_radius: Maximum radius of the pattern (defaults to full frame)
        num_segments: Number of light/dark segments in the pattern
        rotation: Whether to rotate the pattern over time
        
    Returns:
        Numpy array of shape [frames, height, width] with radial pattern
    """
    h, w = size
    if center is None:
        center = (h // 2, w // 2)
    if max_radius is None:
        max_radius = max(h, w)
    
    cy, cx = center
    pattern = np.zeros((frames, h, w), dtype=np.uint8)
    
    for f in range(frames):
        # Create angle grid relative to center
        y, x = np.ogrid[:h, :w]
        angles = np.arctan2(y - cy, x - cx)
        
        # Add rotation if requested
        if rotation:
            rotation_angle = 2 * np.pi * f / frames
            angles += rotation_angle
        
        # Create radial segments
        segment_size = 2 * np.pi / num_segments
        radial_mask = np.mod(angles, 2 * segment_size) < segment_size
        
        # Create distance mask to make it circular
        dist_from_center = np.sqrt((y - cy)**2 + (x - cx)**2)
        dist_mask = dist_from_center <= max_radius
        
        # Combine masks
        pattern[f][np.logical_and(radial_mask, dist_mask)] = 255
    
    return pattern

def analyze_neuron_clusters(responses, ids, num_clusters=3):
    """
    Cluster neurons based on their response patterns and analyze each cluster.
    Uses a simple clustering approach based on activity levels.
    
    Args:
        responses: Neuron responses of shape [frames, neurons]
        ids: Neuron ID dataframe
        num_clusters: Number of clusters to create
        
    Returns:
        Dictionary of neuron clusters with their indices and statistics
    """
    # Calculate summary statistics for each neuron
    mean_activity = responses.mean(axis=0)
    max_activity = responses.max(axis=0)
    std_activity = responses.std(axis=0)
    
    # Simple clustering based on mean activity
    # For a more advanced approach, methods like K-means could be used
    sorted_indices = np.argsort(mean_activity)
    cluster_size = len(sorted_indices) // num_clusters
    
    clusters = {}
    for i in range(num_clusters):
        start_idx = i * cluster_size
        end_idx = (i + 1) * cluster_size if i < num_clusters - 1 else len(sorted_indices)
        
        cluster_indices = sorted_indices[start_idx:end_idx]
        clusters[f"cluster_{i+1}"] = {
            "indices": cluster_indices,
            "size": len(cluster_indices),
            "mean_activity": mean_activity[cluster_indices].mean(),
            "max_activity": max_activity[cluster_indices].mean(),
            "std_activity": std_activity[cluster_indices].mean(),
        }
    
    return clusters

# Main execution
if __name__ == "__main__":
    print("Starting FNN pattern analysis...")
    
    # Load model
    print("Loading model...")
    start_time = time.time()
    model, ids = microns.scan(session=8, scan_idx=5)
    load_time = time.time() - start_time
    print(f"Model loaded in {load_time:.2f} seconds")
    
    # Create stimulus patterns
    print("\nGenerating stimulus patterns...")
    checkerboard = create_checkerboard(frames=60)
    radial_grating = create_radial_grating(frames=60)
    
    print(f"Checkerboard shape: {checkerboard.shape}")
    print(f"Radial grating shape: {radial_grating.shape}")
    
    # Save example frames to visualize the stimuli
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(checkerboard[0], cmap='gray')
    plt.title("Checkerboard Stimulus")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(radial_grating[0], cmap='gray')
    plt.title("Radial Grating Stimulus")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig("stimulus_patterns.png")
    print("Saved stimulus patterns visualization to 'stimulus_patterns.png'")
    
    # Run model prediction for each pattern
    print("\nPredicting neural responses...")
    
    start_time = time.time()
    responses_checker = model.predict(stimuli=checkerboard)
    checker_time = time.time() - start_time
    print(f"Checkerboard prediction completed in {checker_time:.2f} seconds")
    
    start_time = time.time()
    responses_radial = model.predict(stimuli=radial_grating)
    radial_time = time.time() - start_time
    print(f"Radial grating prediction completed in {radial_time:.2f} seconds")
    
    # Print response statistics
    print("\nResponse summary statistics:")
    print(f"Checkerboard: Mean: {responses_checker.mean():.4f}, Max: {responses_checker.max():.4f}, Std: {responses_checker.std():.4f}")
    print(f"Radial grating: Mean: {responses_radial.mean():.4f}, Max: {responses_radial.max():.4f}, Std: {responses_radial.std():.4f}")
    
    # Analyze neural populations by clustering
    print("\nClustering neurons based on response patterns...")
    checker_clusters = analyze_neuron_clusters(responses_checker, ids)
    radial_clusters = analyze_neuron_clusters(responses_radial, ids)
    
    # Print cluster statistics
    print("\nCheckerboard response clusters:")
    for name, cluster in checker_clusters.items():
        print(f"  {name}: {cluster['size']} neurons, mean activity: {cluster['mean_activity']:.4f}")
    
    print("\nRadial grating response clusters:")
    for name, cluster in radial_clusters.items():
        print(f"  {name}: {cluster['size']} neurons, mean activity: {cluster['mean_activity']:.4f}")
    
    # Visualize population responses over time
    print("\nVisualizing population responses...")
    
    plt.figure(figsize=(15, 8))
    
    # Plot average responses for each cluster
    plt.subplot(2, 1, 1)
    for i, (name, cluster) in enumerate(checker_clusters.items()):
        indices = cluster["indices"]
        mean_response = responses_checker[:, indices].mean(axis=1)
        plt.plot(mean_response, label=f"{name} ({cluster['size']} neurons)")
    
    plt.title("Population Responses to Checkerboard")
    plt.xlabel("Time (frames)")
    plt.ylabel("Mean Activity")
    plt.legend()
    
    plt.subplot(2, 1, 2)
    for i, (name, cluster) in enumerate(radial_clusters.items()):
        indices = cluster["indices"]
        mean_response = responses_radial[:, indices].mean(axis=1)
        plt.plot(mean_response, label=f"{name} ({cluster['size']} neurons)")
    
    plt.title("Population Responses to Radial Grating")
    plt.xlabel("Time (frames)")
    plt.ylabel("Mean Activity")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("population_responses.png")
    print("Saved population responses visualization to 'population_responses.png'")
    
    # Visualize response similarity matrix
    print("\nAnalyzing response similarity across neurons...")
    
    # Select a random subset of neurons for visualization (too many would be unreadable)
    n_neurons = 100
    random_indices = np.random.choice(len(ids), size=n_neurons, replace=False)
    
    # Calculate correlation matrices
    corr_checker = np.corrcoef(responses_checker[:, random_indices].T)
    corr_radial = np.corrcoef(responses_radial[:, random_indices].T)
    
    plt.figure(figsize=(15, 7))
    
    plt.subplot(1, 2, 1)
    plt.imshow(corr_checker, cmap='viridis', vmin=-1, vmax=1)
    plt.colorbar(label='Correlation')
    plt.title("Neuron Response Correlations (Checkerboard)")
    plt.xlabel("Neuron Index")
    plt.ylabel("Neuron Index")
    
    plt.subplot(1, 2, 2)
    plt.imshow(corr_radial, cmap='viridis', vmin=-1, vmax=1)
    plt.colorbar(label='Correlation')
    plt.title("Neuron Response Correlations (Radial Grating)")
    plt.xlabel("Neuron Index")
    plt.ylabel("Neuron Index")
    
    plt.tight_layout()
    plt.savefig("response_correlations.png")
    print("Saved response correlation visualization to 'response_correlations.png'")
    
    print("\nPattern analysis completed successfully!")
