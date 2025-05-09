import numpy as np
import matplotlib.pyplot as plt
import time
import torch
import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from tqdm import tqdm
from fnn import microns

# Create output directory if it doesn't exist
output_dir = "receptive_fields"
os.makedirs(output_dir, exist_ok=True)

def generate_white_noise(n_frames=1000, height=144, width=256, contrast=50):
    """
    Generate white noise stimuli with controlled contrast.
    
    Args:
        n_frames: Number of frames to generate
        height: Height of each frame
        width: Width of each frame
        contrast: Standard deviation of the Gaussian noise (0-255)
        
    Returns:
        Numpy array of shape [n_frames, height, width] with white noise
    """
    # Generate Gaussian noise with mean 128 and specified contrast (standard deviation)
    noise = np.random.normal(128, contrast, size=(n_frames, height, width))
    # Clip to valid pixel range
    noise = np.clip(noise, 0, 255).astype(np.uint8)
    return noise

def compute_reverse_correlation_batch(model, neuron_indices, n_frames=500, batch_size=100, contrast=50):
    """
    Compute reverse correlation (spike-triggered average) for a batch of neurons.
    
    Args:
        model: FNN model to use for prediction
        neuron_indices: Indices of neurons to analyze
        n_frames: Total number of frames to use
        batch_size: Number of frames to process in each batch
        contrast: Contrast level for white noise (0-255)
        
    Returns:
        dict: Dictionary mapping neuron indices to their STA results
    """
    results = {}
    n_batches = n_frames // batch_size
    
    # Initialize arrays to hold weighted sums and total responses
    weighted_sums = {idx: np.zeros((144, 256)) for idx in neuron_indices}
    total_responses = {idx: 0 for idx in neuron_indices}
    
    for i in range(n_batches):
        # Generate white noise stimuli (shared across all neurons)
        stimuli = generate_white_noise(n_frames=batch_size, contrast=contrast)
        
        # Get model response for all neurons
        responses = model.predict(stimuli=stimuli)
        
        # Process each neuron's response using vectorized operations
        for idx in neuron_indices:
            neuron_response = responses[:, idx]
            
            # Vectorized calculation of weighted sum
            weighted_sum = np.zeros((144, 256))
            for j, response in enumerate(neuron_response):
                weighted_sum += response * stimuli[j]
            
            weighted_sums[idx] += weighted_sum
            total_responses[idx] += neuron_response.sum()
    
    # Normalize by the total response to get the STAs
    for idx in neuron_indices:
        if total_responses[idx] > 0:
            results[idx] = weighted_sums[idx] / total_responses[idx]
        else:
            results[idx] = weighted_sums[idx]
    
    return results

def process_neuron_batch(model_path, neuron_indices, n_frames=300, batch_size=50, contrast=50, silent=False):
    """
    Process a batch of neurons in a separate process.
    
    Args:
        model_path: Path to save/load the model
        neuron_indices: Indices of neurons to analyze
        n_frames: Total number of frames to use
        batch_size: Number of frames to process in each batch
        contrast: Contrast level for white noise (0-255)
        silent: Whether to suppress logging
        
    Returns:
        dict: Dictionary mapping neuron indices to their STA results
    """
    # Load model in this process - using CPU only as MPS has issues with grid_sample border padding
    if not silent:
        print(f"Loading model for processing {len(neuron_indices)} neurons...")
    
    model, ids = microns.scan(session=8, scan_idx=5, cuda=False, verbose=False)
    
    if not silent:
        print(f"Model loaded, starting computation for {len(neuron_indices)} neurons...")
    
    # Using CPU for computation as MPS has compatibility issues with grid_sample
    # Compute STAs for the batch of neurons
    result = compute_reverse_correlation_batch(model, neuron_indices, n_frames, batch_size, contrast)
    
    if not silent:
        print(f"Completed computation for {len(neuron_indices)} neurons")
    
    return result

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
        plt.close()  # Close the plot to free memory
    else:
        plt.show()

def analyze_neuron_properties(model, ids, neuron_indices):
    """
    Analyze and return properties of selected neurons.
    
    Args:
        model: FNN model
        ids: Neuron ID dataframe
        neuron_indices: Indices of neurons to analyze
        
    Returns:
        dict: Dictionary with neuron properties
    """
    properties = {}
    
    # Check if we can access readout parameters
    if hasattr(model, 'readout') and hasattr(model.readout, 'position'):
        positions = model.readout.position.mean
        if positions is not None:
            positions = positions.detach().cpu().numpy()
            for idx in neuron_indices:
                pos = positions[idx]
                properties[idx] = {
                    'position_x': float(pos[0]),
                    'position_y': float(pos[1])
                }
                
                # Add metadata from dataframe
                if isinstance(ids, np.ndarray):
                    properties[idx]['unit_id'] = ids[idx]
                elif hasattr(ids, 'iloc'):
                    neuron_data = ids.iloc[idx] if idx < len(ids) else None
                    if neuron_data is not None:
                        for column, value in neuron_data.items():
                            properties[idx][column] = value
    
    return properties

def main():
    print("Starting FNN Reverse Correlation Analysis with Improved Progress Monitoring...")
    total_start_time = time.time()
    
    # Time the model loading process
    print("Loading model...")
    start_time = time.time()
    model, ids = microns.scan(session=8, scan_idx=5, verbose=True)
    load_time = time.time() - start_time
    print(f"Model loaded in {load_time:.2f} seconds")
    
    # We'll use CPU instead of MPS since grid_sample has compatibility issues with MPS
    print("Using CPU for computation since MPS has issues with grid_sample border padding")
    
    # Print basic model information
    n_total_neurons = len(ids)
    print(f"Total neurons in the model: {n_total_neurons}")
    
    # Randomly select 1000 neurons for analysis
    np.random.seed(42)  # For reproducibility
    n_neurons = 1000
    selected_neurons = np.random.choice(n_total_neurons, size=n_neurons, replace=False)
    print(f"Randomly selected {n_neurons} neurons for analysis")
    
    # Check for existing results and load them if available
    all_results = {}
    results_file = f"{output_dir}/all_results.npy"
    if os.path.exists(results_file):
        print("Found existing results, loading...")
        try:
            all_results = np.load(results_file, allow_pickle=True).item()
            print(f"Loaded {len(all_results)} existing neuron results")
        except Exception as e:
            print(f"Error loading existing results: {e}")
    
    # Analyze and save properties of selected neurons if not already done
    properties_file = f"{output_dir}/neuron_properties.npy"
    if not os.path.exists(properties_file):
        print("Extracting neuron properties...")
        properties = analyze_neuron_properties(model, ids, selected_neurons)
        np.save(properties_file, properties)
        print(f"Neuron properties saved to {properties_file}")
    else:
        print(f"Neuron properties already exist at {properties_file}")
    
    # Determine neurons that still need processing
    neurons_to_process = [n for n in selected_neurons if n not in all_results]
    print(f"Need to process {len(neurons_to_process)} out of {n_neurons} neurons")
    
    if not neurons_to_process:
        print("All neurons already processed, skipping to visualization")
    else:
        # Adjust parameters for better progress tracking
        n_frames = 200     # Reduced for faster computation
        batch_size = 50    # Smaller batches for better monitoring
        contrast = 50      # Standard deviation of the white noise
        
        # Use smaller batches for better progress visualization
        neurons_per_batch = 10  # Process 10 neurons at a time for clear progress
        neuron_batches = [neurons_to_process[i:i+neurons_per_batch] 
                         for i in range(0, len(neurons_to_process), neurons_per_batch)]
        
        print(f"Split {len(neurons_to_process)} neurons into {len(neuron_batches)} smaller batches")
        print(f"Each batch will process {neurons_per_batch} neurons with {n_frames} frames per neuron")
        
        # Create a temporary path to save/load the model if needed
        model_path = "temp_model.pt"
    
        # Start sequential computation with clear progress updates
        print(f"Starting computation with {n_frames} frames per neuron...")
        compute_start_time = time.time()
        
        # Process each batch and show progress
        for batch_idx, batch in enumerate(neuron_batches):
            try:
                batch_start = time.time()
                print(f"Processing batch {batch_idx+1}/{len(neuron_batches)} with {len(batch)} neurons...")
                print(f"Batch neurons: {batch}")
                
                # Process this batch
                print(f"Starting computation for {len(batch)} neurons...")
                batch_results = process_neuron_batch(model_path, batch, n_frames, batch_size, contrast)
                
                # Update overall results
                print(f"Computation finished, updating results dictionary...")
                all_results.update(batch_results)
                
                # Save intermediate results after each batch
                print(f"Saving batch results to {results_file}...")
                np.save(results_file, all_results)
                print(f"Results saved with {len(all_results)} total neurons.")
                
                batch_time = time.time() - batch_start
                print(f"Batch {batch_idx+1}/{len(neuron_batches)} completed in {batch_time:.2f} seconds")
                
                # Estimate remaining time
                elapsed = time.time() - compute_start_time
                batches_completed = batch_idx + 1
                batches_remaining = len(neuron_batches) - batches_completed
                if batches_completed > 0:
                    avg_time_per_batch = elapsed / batches_completed
                    est_remaining_time = avg_time_per_batch * batches_remaining
                    print(f"Progress: {batches_completed}/{len(neuron_batches)} batches " +
                          f"({batches_completed*100/len(neuron_batches):.1f}%)")
                    print(f"Estimated remaining time: {est_remaining_time/60:.1f} minutes")
            except Exception as e:
                print(f"ERROR processing batch {batch_idx+1}: {str(e)}")
                import traceback
                traceback.print_exc()
                print("Saving current results and continuing...")
                try:
                    # Try to save what we have
                    np.save(results_file, all_results)
                    print(f"Saved {len(all_results)} results successfully despite error.")
                except:
                    print("Could not save results after error.")
        
        rc_time = time.time() - compute_start_time
        print(f"Computation completed in {rc_time:.2f} seconds")
        print(f"Average time per neuron: {rc_time/len(neurons_to_process):.2f} seconds")
    
    # Visualize and save receptive fields for neurons that don't have visualizations yet
    print("Generating and saving visualizations...")
    viz_start_time = time.time()
    
    # Check which neurons already have visualizations
    visualization_needed = []
    for neuron_idx in selected_neurons:
        output_file = f"{output_dir}/neuron_{neuron_idx}_receptive_field.png"
        if not os.path.exists(output_file) and neuron_idx in all_results:
            visualization_needed.append(neuron_idx)
    
    print(f"Generating visualizations for {len(visualization_needed)} neurons...")
    
    for i, neuron_idx in enumerate(tqdm(visualization_needed, desc="Saving visualizations")):
        # Save the raw STA data if not already saved
        sta_file = f"{output_dir}/neuron_{neuron_idx}_sta.npy"
        if not os.path.exists(sta_file):
            np.save(sta_file, all_results[neuron_idx])
        
        # Generate and save visualization
        output_file = f"{output_dir}/neuron_{neuron_idx}_receptive_field.png"
        visualize_receptive_field(all_results[neuron_idx], neuron_idx, file_path=output_file)
        
        # Print progress periodically
        if (i+1) % 50 == 0:
            print(f"Processed {i+1}/{len(visualization_needed)} visualizations")
    
    viz_time = time.time() - viz_start_time
    print(f"Visualization generation completed in {viz_time:.2f} seconds")
    
    # Generate a summary visualization of a sample of receptive fields if not already created
    summary_file = f"{output_dir}/sample_receptive_fields.png"
    if not os.path.exists(summary_file):
        print("Generating summary visualization...")
        plt.figure(figsize=(15, 15))
        
        sample_size = min(25, len(selected_neurons))  # Show up to 25 neurons in the summary
        sample_indices = np.random.choice(list(all_results.keys()), size=sample_size, replace=False)
        
        for i, idx in enumerate(sample_indices):
            plt.subplot(5, 5, i+1)
            sta_normalized = (all_results[idx] - np.mean(all_results[idx])) / np.std(all_results[idx])
            plt.imshow(sta_normalized, cmap='coolwarm')
            plt.title(f"Neuron {idx}")
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(summary_file)
        print(f"Summary visualization saved to {summary_file}")
    else:
        print(f"Summary visualization already exists at {summary_file}")
    
    # Calculate total execution time
    total_time = time.time() - total_start_time
    print(f"Total execution time: {total_time:.2f} seconds")
    
    print(f"All results saved to '{output_dir}' directory")
    print("Receptive field analysis completed successfully!")

if __name__ == "__main__":
    main()
