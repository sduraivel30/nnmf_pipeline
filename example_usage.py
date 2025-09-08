"""
Example usage script for NNMF Pipeline
Demonstrates how to use the pipeline with sample data.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import the main pipeline
from src.main_pipeline import NNMFPipeline


def generate_sample_data():
    """Generate synthetic electrode data for demonstration."""
    np.random.seed(42)
    
    # Parameters
    n_electrodes = 50
    n_stories = 3
    story_lengths = np.array([200, 300, 250])  # samples per story
    total_time = np.sum(story_lengths)
    
    # Generate synthetic electrode data with some structure
    # Create some underlying components
    n_true_components = 4
    W_true = np.random.exponential(1, (n_electrodes, n_true_components))
    
    # Generate temporal patterns for each story
    electrode_data = np.zeros((n_electrodes, total_time))
    current_pos = 0
    
    for story_idx, story_len in enumerate(story_lengths):
        # Generate story-specific temporal patterns
        H_story = np.random.exponential(1, (n_true_components, story_len))
        
        # Add some smooth temporal structure
        for comp in range(n_true_components):
            t = np.linspace(0, 4*np.pi, story_len)
            H_story[comp] *= (1 + 0.5 * np.sin(t + comp))
        
        # Generate electrode data for this story
        story_data = W_true @ H_story
        
        # Add noise
        story_data += 0.3 * np.random.randn(n_electrodes, story_len)
        
        # Store in full data matrix
        electrode_data[:, current_pos:current_pos + story_len] = story_data
        current_pos += story_len
    
    # Generate subject IDs (3 subjects with different numbers of electrodes)
    subject_ids = (['Subject_A'] * 20 + 
                   ['Subject_B'] * 15 + 
                   ['Subject_C'] * 15)
    
    return electrode_data, story_lengths, subject_ids


def main():
    """Main example script."""
    print("NNMF Pipeline Example")
    print("=" * 50)
    
    # Generate sample data
    print("Generating synthetic electrode data...")
    electrode_data, story_lengths, subject_ids = generate_sample_data()
    
    print(f"Data shape: {electrode_data.shape}")
    print(f"Stories: {len(story_lengths)} (lengths: {story_lengths})")
    print(f"Subjects: {len(set(subject_ids))}")
    
    # Initialize pipeline
    print("\\nInitializing NNMF pipeline...")
    pipeline = NNMFPipeline(
        n_permutations=100,  # Reduced for faster demo
        n_jobs=2,           # Use 2 cores for demo
        random_state=42
    )
    
    # Run complete analysis
    print("\\nRunning complete NNMF analysis...")
    results = pipeline.run_complete_analysis(
        electrode_data=electrode_data,
        story_lengths=story_lengths,
        subject_ids=subject_ids,
        max_components=8,     # Reduced for faster demo
        cv_splits=3,          # Reduced for faster demo
        init_reps=5,          # Reduced for faster demo
        test_frac=0.2,
        save_results=True,
        output_dir=\"results/demo_run\"
    )
    
    # Display some results
    print(\"\\n\" + \"=\"*50)
    print(\"ANALYSIS RESULTS SUMMARY\")
    print(\"=\"*50)
    
    # Model summary
    final_model = results['final_model']
    print(f\"\\nOptimal Model:\")
    print(f\"  Components: {final_model['n_components']}\")
    print(f\"  Alpha: {final_model['alpha']:.4f}\")
    print(f\"  Variance explained: {final_model['variance_explained']:.3f}\")
    
    # Statistical summary
    stats = results['statistics']
    n_electrodes = electrode_data.shape[0]
    print(f\"\\nElectrode Reliability:\")
    print(f\"  Cross-run significant: {stats['cross_run']['n_significant']}/{n_electrodes}\")
    print(f\"  Permutation significant: {stats['cross_electrode']['n_significant_electrodes_perm']}/{n_electrodes}\")
    print(f\"  2nd-level correlation: {stats['cross_electrode']['second_level_correlation']:.3f}\")
    
    # Component summary
    components = results['components']
    print(f\"\\nComponent Analysis:\")
    for i in range(final_model['n_components']):
        comp_key = f'component_{i+1}'
        if comp_key in components:
            comp = components[comp_key]
            print(f\"  Component {i+1}: sparsity={comp.get('spatial_sparsity', 'N/A'):.3f}\")
    
    # Create some plots
    create_example_plots(results)
    
    print(\"\\nExample completed successfully!\")
    print(f\"Results saved to: results/demo_run/\")


def create_example_plots(results):
    \"\"\"Create example visualization plots.\"\"\"\n    print(\"\\nCreating example plots...\")\n    \n    fig, axes = plt.subplots(2, 3, figsize=(15, 10))\n    \n    # Plot 1: Cross-run correlations\n    stats = results['statistics']\n    valid_corrs = stats['cross_run']['valid_correlations']\n    threshold = stats['cross_run']['shuffle_threshold']\n    \n    axes[0, 0].hist(valid_corrs, bins=15, alpha=0.7, color='blue')\n    axes[0, 0].axvline(threshold, color='red', linestyle='--', linewidth=2)\n    axes[0, 0].set_xlabel('Cross-run Correlation')\n    axes[0, 0].set_ylabel('Count')\n    axes[0, 0].set_title('Cross-run Correlations')\n    axes[0, 0].grid(True)\n    \n    # Plot 2: Spatial components (W matrix)\n    W = results['final_model']['W']\n    im1 = axes[0, 1].imshow(W.T, aspect='auto', cmap='viridis')\n    axes[0, 1].set_xlabel('Electrodes')\n    axes[0, 1].set_ylabel('Components')\n    axes[0, 1].set_title('Spatial Components (W)')\n    plt.colorbar(im1, ax=axes[0, 1])\n    \n    # Plot 3: Temporal components (H matrix)\n    H = results['final_model']['H']\n    im2 = axes[0, 2].imshow(H, aspect='auto', cmap='viridis')\n    axes[0, 2].set_xlabel('Time')\n    axes[0, 2].set_ylabel('Components')\n    axes[0, 2].set_title('Temporal Components (H)')\n    plt.colorbar(im2, ax=axes[0, 2])\n    \n    # Plot 4: Cross-validation results\n    cv = results['cross_validation']\n    component_range = cv['component_range']\n    mean_errors = np.mean(cv['mean_recon_errors'], axis=1)\n    \n    axes[1, 0].plot(component_range, mean_errors, 'b-o')\n    axes[1, 0].axvline(cv['optimal_components'], color='red', linestyle='--')\n    axes[1, 0].set_xlabel('Number of Components')\n    axes[1, 0].set_ylabel('Mean Reconstruction Error')\n    axes[1, 0].set_title('Cross-validation Results')\n    axes[1, 0].grid(True)\n    \n    # Plot 5: Electrode pair counts\n    pair_counts = stats['cross_electrode']['electrode_pair_counts']\n    thresholds = stats['cross_electrode']['null_thresholds']\n    \n    sorted_indices = np.argsort(pair_counts)[::-1]\n    sorted_counts = pair_counts[sorted_indices]\n    sorted_thresholds = thresholds[sorted_indices]\n    \n    axes[1, 1].plot(sorted_counts, 'b-', linewidth=2, label='Observed')\n    axes[1, 1].plot(sorted_thresholds, 'r--', linewidth=2, label='Threshold')\n    axes[1, 1].set_xlabel('Electrodes (sorted)')\n    axes[1, 1].set_ylabel('Significant Pair Count')\n    axes[1, 1].set_title('Electrode Significance')\n    axes[1, 1].legend()\n    axes[1, 1].grid(True)\n    \n    # Plot 6: Reconstruction quality\n    original_data = results['data_splits']['run1'] + results['data_splits']['run2']\n    reconstruction = W @ H\n    \n    # Show first few electrodes\n    time_slice = slice(0, 200)  # First 200 time points\n    electrode_slice = slice(0, 5)  # First 5 electrodes\n    \n    axes[1, 2].plot(original_data[electrode_slice, time_slice].T, alpha=0.7, label='Original')\n    axes[1, 2].plot(reconstruction[electrode_slice, time_slice].T, '--', alpha=0.7, label='Reconstruction')\n    axes[1, 2].set_xlabel('Time')\n    axes[1, 2].set_ylabel('Signal')\n    axes[1, 2].set_title('Reconstruction Quality')\n    axes[1, 2].legend()\n    axes[1, 2].grid(True)\n    \n    plt.tight_layout()\n    \n    # Save plot\n    output_dir = Path(\"results/demo_run\")\n    output_dir.mkdir(parents=True, exist_ok=True)\n    plt.savefig(output_dir / \"analysis_summary.png\", dpi=150, bbox_inches='tight')\n    plt.show()\n    \n    print(\"Plots saved to results/demo_run/analysis_summary.png\")\n\n\nif __name__ == \"__main__\":\n    main()