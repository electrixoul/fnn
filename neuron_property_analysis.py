import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from fnn import microns

def calculate_neuron_metrics(responses, stimuli_types):
    """
    Calculate various metrics for each neuron based on their responses to different stimuli.
    
    Args:
        responses: Dict where keys are stimuli types and values are response arrays of shape [frames, neurons]
        stimuli_types: List of stimuli types to analyze
        
    Returns:
        DataFrame with neuron metrics
    """
    n_neurons = responses[stimuli_types[0]].shape[1]
    metrics = pd.DataFrame(index=range(n_neurons))
    
    # Calculate basic metrics for each stimulus type
    for stim_type in stimuli_types:
        resp = responses[stim_type]
        metrics[f'{stim_type}_mean'] = resp.mean(axis=0)
        metrics[f'{stim_type}_max'] = resp.max(axis=0)
        metrics[f'{stim_type}_std'] = resp.std(axis=0)
        
        # Time to peak (frame index of maximum response)
        metrics[f'{stim_type}_peak_time'] = np.argmax(resp, axis=0)
        
        # Calculate response onset speed (slope from start to first peak)
        peak_times = np.argmax(resp, axis=0)
        slopes = np.zeros(n_neurons)
        for i in range(n_neurons):
            if peak_times[i] > 0:  # Avoid division by zero
                slopes[i] = resp[peak_times[i], i] / peak_times[i]
        metrics[f'{stim_type}_onset_speed'] = slopes
        
        # Calculate adaptation index (end response / max response)
        # Values close to 1 mean little adaptation, values close to 0 mean strong adaptation
        last_resp = resp[-1, :]
        max_resp = resp.max(axis=0)
        # Avoid division by zero
        adaptation = np.where(max_resp > 0, last_resp / np.maximum(max_resp, 1e-10), 1.0)
        metrics[f'{stim_type}_adaptation'] = adaptation
    
    # Calculate cross-stimulus selectivity (max of one stimulus / mean of all stimuli)
    max_responses = np.array([metrics[f'{stim_type}_max'] for stim_type in stimuli_types])
    mean_across_stim = max_responses.mean(axis=0)
    max_across_stim = max_responses.max(axis=0)
    
    # Get most selective stimulus for each neuron
    most_selective_stim_idx = np.argmax(max_responses, axis=0)
    metrics['preferred_stimulus'] = [stimuli_types[i] for i in most_selective_stim_idx]
    
    # Calculate selectivity index (ratio of max to mean)
    # Higher values mean more selective neurons
    metrics['selectivity_index'] = np.where(mean_across_stim > 0, 
                                          max_across_stim / np.maximum(mean_across_stim, 1e-10), 
                                          1.0)
    
    # Calculate temporal dynamics metric (std of response over time / mean response)
    # Higher values mean more temporally dynamic neurons
    temporal_variability = np.array([metrics[f'{stim_type}_std'] for stim_type in stimuli_types])
    mean_responses = np.array([metrics[f'{stim_type}_mean'] for stim_type in stimuli_types])
    
    # Average across stimuli
    avg_temp_var = temporal_variability.mean(axis=0)
    avg_mean_resp = mean_responses.mean(axis=0)
    
    # Calculate coefficient of variation over time
    metrics['temporal_dynamics'] = np.where(avg_mean_resp > 0, 
                                          avg_temp_var / np.maximum(avg_mean_resp, 1e-10), 
                                          0.0)
    
    return metrics

def create_stimuli(size=(144, 256)):
    """
    Create a dictionary of different stimuli for testing.
    
    Returns:
        Dictionary with stimuli names as keys and stimulus arrays as values
    """
    # Create basic brightness sequence (black-gray-white)
    basic_sequence = np.zeros((90, size[0], size[1]), dtype=np.uint8)
    basic_sequence[0:30] = 0      # Black
    basic_sequence[30:60] = 128   # Gray
    basic_sequence[60:90] = 255   # White
    
    # Create moving bar stimulus
    moving_bar = np.zeros((60, size[0], size[1]), dtype=np.uint8)
    bar_width = 20
    for f in range(60):
        pos = int(f * (size[1] - bar_width) / 59)  # Smoothly move from left to right
        moving_bar[f, :, pos:pos+bar_width] = 255
    
    # Create expanding circle stimulus
    expanding_circle = np.zeros((60, size[0], size[1]), dtype=np.uint8)
    center_y, center_x = size[0] // 2, size[1] // 2
    
    for f in range(60):
        # Radius varies from 20 to 60 pixels following a sine pattern
        radius = 20 + 20 * np.sin(f / 60 * 2 * np.pi)
        y_grid, x_grid = np.ogrid[:size[0], :size[1]]
        dist_from_center = np.sqrt((y_grid - center_y)**2 + (x_grid - center_x)**2)
        circle_mask = dist_from_center <= radius
        expanding_circle[f][circle_mask] = 255
    
    # Create checkerboard stimulus
    checkerboard = np.zeros((60, size[0], size[1]), dtype=np.uint8)
    square_size = 16
    
    for f in range(60):
        phase_offset = f % 2  # Alternate phase every frame
        for i in range(0, size[0], square_size):
            for j in range(0, size[1], square_size):
                if ((i // square_size) + (j // square_size) + phase_offset) % 2 == 0:
                    checkerboard[f, i:i+square_size, j:j+square_size] = 255
    
    # Create radial grating stimulus
    radial_grating = np.zeros((60, size[0], size[1]), dtype=np.uint8)
    center_y, center_x = size[0] // 2, size[1] // 2
    num_segments = 8
    
    for f in range(60):
        # Create angle grid relative to center
        y, x = np.ogrid[:size[0], :size[1]]
        angles = np.arctan2(y - center_y, x - center_x)
        
        # Add rotation
        rotation_angle = 2 * np.pi * f / 60
        angles += rotation_angle
        
        # Create radial segments
        segment_size = 2 * np.pi / num_segments
        radial_mask = np.mod(angles, 2 * segment_size) < segment_size
        
        # Set white pixels
        radial_grating[f][radial_mask] = 255
    
    return {
        'basic_sequence': basic_sequence,
        'moving_bar': moving_bar,
        'expanding_circle': expanding_circle,
        'checkerboard': checkerboard,
        'radial_grating': radial_grating
    }

def analyze_neuron_representations(model, ids, responses, stimuli_types):
    """
    Analyze the relationships between neuron properties and response patterns.
    
    Args:
        model: The FNN model
        ids: Neuron ID dataframe
        responses: Dict with stimulus types as keys and response arrays as values
        stimuli_types: List of stimulus types
        
    Returns:
        None (generates plots and prints statistics)
    """
    print("Analyzing neuron property relationships...")
    
    # Get neuron metrics
    metrics = calculate_neuron_metrics(responses, stimuli_types)
    
    # Add spatial position information if available
    if hasattr(model, 'readout') and hasattr(model.readout, 'positions'):
        positions = model.readout.positions
        if positions is not None:
            metrics['x_pos'] = positions[:, 0]
            metrics['y_pos'] = positions[:, 1]
            
            # Calculate distance from center
            metrics['distance_from_center'] = np.sqrt(metrics['x_pos']**2 + metrics['y_pos']**2)
    
    # Plot distribution of neuron preferences
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    preference_counts = metrics['preferred_stimulus'].value_counts()
    preference_counts.plot(kind='bar', color='skyblue')
    plt.title('Distribution of Preferred Stimuli')
    plt.xlabel('Stimulus Type')
    plt.ylabel('Number of Neurons')
    plt.xticks(rotation=45)
    
    plt.subplot(1, 2, 2)
    plt.hist(metrics['selectivity_index'], bins=50, color='orange', alpha=0.7)
    plt.title('Distribution of Selectivity Index')
    plt.xlabel('Selectivity Index (higher = more selective)')
    plt.ylabel('Number of Neurons')
    
    plt.tight_layout()
    plt.savefig("neuron_preferences.png")
    
    # Analyze relationship between spatial position and stimulus preference
    if 'x_pos' in metrics.columns and 'y_pos' in metrics.columns:
        plt.figure(figsize=(10, 8))
        
        # Color by preferred stimulus
        stim_categories = metrics['preferred_stimulus'].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(stim_categories)))
        
        for i, stim in enumerate(stim_categories):
            subset = metrics[metrics['preferred_stimulus'] == stim]
            plt.scatter(subset['x_pos'], subset['y_pos'], c=[colors[i]], 
                       label=stim, alpha=0.5, s=10)
        
        plt.title('Spatial Distribution of Neuron Preferences')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig("spatial_preferences.png")
        
        # Analyze relationship between distance from center and selectivity
        plt.figure(figsize=(10, 6))
        
        # Create hexbin plot
        plt.hexbin(metrics['distance_from_center'], metrics['selectivity_index'], 
                 gridsize=30, cmap='viridis', mincnt=1)
        plt.colorbar(label='Neuron Count')
        
        # Add trend line
        from scipy import stats
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            metrics['distance_from_center'], metrics['selectivity_index'])
        
        x_vals = np.linspace(metrics['distance_from_center'].min(), 
                           metrics['distance_from_center'].max(), 100)
        plt.plot(x_vals, intercept + slope * x_vals, 'r-', 
               label=f'Trend: r={r_value:.2f}, p={p_value:.4f}')
        
        plt.title('Relationship Between Distance from Center and Selectivity')
        plt.xlabel('Distance from Center')
        plt.ylabel('Selectivity Index')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig("center_distance_vs_selectivity.png")
    
    # Analyze temporal dynamics across stimuli
    plt.figure(figsize=(15, 10))
    
    for i, stim_type in enumerate(stimuli_types):
        plt.subplot(2, 3, i+1)
        
        # Get peak times and adaptation metrics for this stimulus
        peak_times = metrics[f'{stim_type}_peak_time']
        adaptation = metrics[f'{stim_type}_adaptation']
        
        # Create scatter plot
        plt.scatter(peak_times, adaptation, alpha=0.5, s=5)
        
        plt.title(f'Temporal Dynamics: {stim_type}')
        plt.xlabel('Time to Peak (frames)')
        plt.ylabel('Adaptation Index')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("temporal_dynamics.png")
    
    # Dimensionality reduction on response patterns
    print("\nPerforming dimensionality reduction on response patterns...")
    
    # Prepare data for dimensionality reduction
    # Concatenate mean responses to each stimulus type
    features = np.column_stack([metrics[f'{stim_type}_mean'] for stim_type in stimuli_types])
    
    # Apply PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(features)
    
    plt.figure(figsize=(10, 8))
    
    # Color by preferred stimulus
    stim_categories = metrics['preferred_stimulus'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(stim_categories)))
    
    for i, stim in enumerate(stim_categories):
        subset_idx = metrics['preferred_stimulus'] == stim
        plt.scatter(pca_result[subset_idx, 0], pca_result[subset_idx, 1], 
                   c=[colors[i]], label=stim, alpha=0.7, s=15)
    
    plt.title('PCA of Neuron Response Patterns')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("response_pattern_pca.png")
    
    # Print key insights
    print("\nKey statistical insights:")
    print(f"Total neurons analyzed: {len(metrics)}")
    
    print("\nStimulus preference distribution:")
    preference_pct = preference_counts / len(metrics) * 100
    for stim, count in preference_counts.items():
        print(f"  {stim}: {count} neurons ({preference_pct[stim]:.1f}%)")
    
    print("\nSelectivity statistics:")
    print(f"  Mean selectivity index: {metrics['selectivity_index'].mean():.3f}")
    print(f"  Max selectivity index: {metrics['selectivity_index'].max():.3f}")
    print(f"  Min selectivity index: {metrics['selectivity_index'].min():.3f}")
    
    # If position data is available
    if 'distance_from_center' in metrics.columns:
        print("\nSpatial organization:")
        print(f"  Correlation between distance from center and selectivity: r={r_value:.3f}, p={p_value:.4f}")
        
        # Group by preferred stimulus and get average positions
        avg_positions = metrics.groupby('preferred_stimulus')[['x_pos', 'y_pos']].mean()
        print("\nAverage positions by preferred stimulus:")
        print(avg_positions)

# Main execution
if __name__ == "__main__":
    print("Starting FNN neuron property analysis...")
    
    # Load model
    print("Loading model...")
    start_time = time.time()
    model, ids = microns.scan(session=8, scan_idx=5)
    load_time = time.time() - start_time
    print(f"Model loaded in {load_time:.2f} seconds")
    
    # Create stimuli
    print("\nGenerating stimuli...")
    stimuli = create_stimuli()
    stimuli_types = list(stimuli.keys())
    print(f"Created {len(stimuli_types)} different stimulus types")
    
    # Run model predictions
    print("\nPredicting neural responses...")
    responses = {}
    
    for stim_type, stimulus in stimuli.items():
        start_time = time.time()
        responses[stim_type] = model.predict(stimuli=stimulus)
        pred_time = time.time() - start_time
        print(f"  {stim_type}: {len(stimulus)} frames, predicted in {pred_time:.2f} seconds")
    
    # Analyze neuron properties and response patterns
    analyze_neuron_representations(model, ids, responses, stimuli_types)
    
    print("\nNeuron property analysis completed successfully!")
    print("Generated visualization files:")
    print("  - neuron_preferences.png")
    print("  - spatial_preferences.png")
    print("  - center_distance_vs_selectivity.png")
    print("  - temporal_dynamics.png")
    print("  - response_pattern_pca.png")
