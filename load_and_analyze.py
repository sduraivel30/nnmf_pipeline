"""
Example Usage - Load Data and Run NNMF Analysis
Demonstrates loading preprocessed data and running the pipeline.
"""

import numpy as np
from pathlib import Path
import logging
from src.data_loader import load_story_data_for_nnmf
from src.main_pipeline import NNMFPipeline

# Setup logging
logging.basicConfig(level=logging.INFO)

def main():
    """Example of loading real data and running NNMF analysis."""
    print("NNMF Pipeline - Real Data Example")
    print("=" * 50)
    
    # Configuration
    config = {
        'data_dir': '/path/to/your/data/directory',  # <--- UPDATE THIS
        'story_name': 'YourStoryName',               # <--- UPDATE THIS
        'epoch': [-0.5, 180.0],                      # Time epoch in seconds
        'subject_list': None,                        # None = all subjects; or e.g. ['subj1', 'subj2']
        'summary_stats_file': '',                    # Path to langloc file if available
        'include_significant': False,
        # NNMF parameters
        'max_components': 15,
        'cv_splits': 5,
        'init_reps': 10,
        'test_frac': 0.2,
        'n_permutations': 1000,
        'save_results': True,
        'output_dir': 'results/real_data_analysis'
    }
    
    print("\nStep 1: Loading story data across subjects...")
    try:
        electrode_data, story_lengths, subject_ids = load_story_data_for_nnmf(
            data_dir=config['data_dir'],
            story_name=config['story_name'],
            epoch=config['epoch'],
            subject_list=config['subject_list'],
            summary_stats_file=config['summary_stats_file'],
            include_significant=config['include_significant'],
            verbose=True
        )
        print("\nLoaded data successfully:")
        print(f"  Electrode data shape: {electrode_data.shape}")
        print(f"  Story lengths: {story_lengths}")
        print(f"  Number of subjects: {len(set(subject_ids))}")
        print(f"  Subject distribution: {dict((subj, subject_ids.count(subj)) for subj in set(subject_ids))}")
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        print("\nUsing synthetic data for demonstration...")
        electrode_data, story_lengths, subject_ids = generate_synthetic_data()
    
    print("\nStep 2: Initializing NNMF Pipeline...")
    pipeline = NNMFPipeline(
        n_permutations=config['n_permutations'],
        n_jobs=-1,  # All cores
        random_state=42
    )
    
    print("\nStep 3: Running complete NNMF analysis...")
    try:
        results = pipeline.run_complete_analysis(
            electrode_data=electrode_data,
            story_lengths=story_lengths,
            subject_ids=subject_ids,
            max_components=config['max_components'],
            cv_splits=config['cv_splits'],
            init_reps=config['init_reps'],
            test_frac=config['test_frac'],
            save_results=config['save_results'],
            output_dir=config['output_dir']
        )
        # Step 4: Results summary
        print("\n" + "="*60)
        print("DETAILED RESULTS ANALYSIS")
        print("="*60)
        analyze_results(results, electrode_data.shape[0])
        # Step 5: Visualizations
        print("\nStep 5: Creating detailed visualizations...")
        create_detailed_visualizations(results)
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()

def generate_synthetic_data():
    """Generate synthetic data as fallback when real data isn't available."""
    np.random.seed(42)
    n_electrodes = 64
    total_time = 2000
    story_lengths = np.array([total_time])  # Single story
    subject_distribution = [16, 20, 14, 14]  # Electrodes per subject
    subject_ids = []
    for i, count in enumerate(subject_distribution):
        subject_ids.extend([f'Subject_{i+1}'] * count)
    n_true_components = 6
    W_true = np.random.exponential(0.8, (n_electrodes, n_true_components))
    current_electrode = 0
    for subj_idx, count in enumerate(subject_distribution):
        subject_pattern = np.random.exponential(1, (count, n_true_components))
        subject_pattern[:, subj_idx] *= 2
        W_true[current_electrode:current_electrode + count] = subject_pattern
        current_electrode += count
    H_true = np.random.exponential(0.5, (n_true_components, total_time))
    for comp in range(n_true_components):
        t = np.linspace(0, 8 * np.pi, total_time)
        H_true[comp] *= (1 + 0.4 * np.sin(t * (comp + 1) / n_true_components))
        trend = np.exp(-0.5 * (np.linspace(-2, 2, total_time) ** 2))
        H_true[comp] *= (1 + 0.3 * trend)
    electrode_data = W_true @ H_true
    electrode_data += 0.2 * np.random.randn(n_electrodes, total_time)
    current_electrode = 0
    for subj_idx, count in enumerate(subject_distribution):
        common_noise = 0.1 * np.random.randn(1, total_time)
        electrode_data[current_electrode:current_electrode + count] += common_noise
        current_electrode += count
    line_noise = 0.05 * np.sin(2 * np.pi * 60 * np.linspace(0, 10, total_time))
    electrode_data += line_noise[np.newaxis, :]
    print("Generated synthetic data with realistic structure")
    return electrode_data, story_lengths, subject_ids

def analyze_results(results: dict, n_electrodes: int):
    """Provide detailed analysis of results."""
    final_model = results['final_model']
    cv_results = results['cross_validation']
    stats = results['statistics']
    print(f"\n1. MODEL PERFORMANCE:")
    print(f"   Optimal components: {final_model['n_components']}")
    print(f"   Optimal alpha: {final_model['alpha']:.4f}")
    print(f"   Variance explained: {final_model['variance_explained']:.3f}")
    print(f"   Reconstruction error: {final_model['reconstruction_error']:.6f}")
    print(f"   Training time: {final_model.get('training_time', 'N/A')} seconds")
    print(f"\n2. CROSS-VALIDATION RESULTS:")
    mean_cv_error = np.mean(cv_results['mean_recon_errors'])
    mean_reliability = np.mean(cv_results['mean_reliability'])
    print(f"   Mean CV reconstruction error: {mean_cv_error:.6f}")
    print(f"   Mean CV reliability: {mean_reliability:.3f}")
    print(f"   Component range tested: {cv_results['component_range']}")
    print(f"   Alpha range tested: {cv_results['alpha_range']}")
    print(f"\n3. ELECTRODE RELIABILITY:")
    cross_run = stats['cross_run']
    cross_electrode = stats['cross_electrode']
    print(f"   Total electrodes: {n_electrodes}")
    print(f"   Cross-run significant: {cross_run['n_significant']}/{n_electrodes} ({100*cross_run['n_significant']/n_electrodes:.1f}%)")
    print(f"   Cross-run mean correlation: {cross_run['mean_correlation']:.3f}")
    print(f"   Cross-run shuffle threshold: {cross_run['shuffle_threshold']:.3f}")
    print(f"\n4. ELECTRODE PAIR ANALYSIS:")
    print(f"   Valid electrode pairs: {cross_electrode['n_valid_pairs']}")
    print(f"   Significant pairs: {cross_electrode['n_significant_pairs']}")
    print(f"   2nd-level correlation: {cross_electrode['second_level_correlation']:.3f}")
    print(f"   Permutation significant electrodes: {cross_electrode['n_significant_electrodes_perm']}/{n_electrodes} ({100*cross_electrode['n_significant_electrodes_perm']/n_electrodes:.1f}%)")
    if 'components' in results:
        print(f"\n5. COMPONENT ANALYSIS:")
        components = results['components']
        for i in range(final_model['n_components']):
            comp_key = f'component_{i+1}'
            if comp_key in components:
                comp = components[comp_key]
                print(f"   Component {i+1}:")
                print(f"     Spatial sparsity: {comp.get('spatial_sparsity', 'N/A'):.3f}")
                print(f"     Temporal sparsity: {comp.get('temporal_sparsity', 'N/A'):.3f}")
                print(f"     Variance contribution: {comp.get('variance_contribution', 'N/A'):.3f}")
    subject_counts = {}
    for subj_id in results['statistics'].get('subject_ids', []):
        subject_counts[subj_id] = subject_counts.get(subj_id, 0) + 1
    print(f"\n6. SUBJECT ANALYSIS:")
    print(f"   Number of subjects: {len(subject_counts)}")
    for subj, count in subject_counts.items():
        print(f"     {subj}: {count} electrodes")

def create_detailed_visualizations(results: dict):
    """Create comprehensive visualizations of results."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.style.use('default')
    sns.set_palette("husl")
    fig = plt.figure(figsize=(20, 15))
    # Plot 1: Cross-validation heatmap
    ax1 = plt.subplot(3, 4, (1, 2))
    cv_results = results['cross_validation']
    im = ax1.imshow(cv_results['composite_score'], aspect='auto', cmap='viridis')
    ax1.set_title('Cross-Validation: Composite Score\n(Lower = Better)', fontsize=12)
    ax1.set_xlabel('Alpha Index')
    ax1.set_ylabel('Component Index')
    plt.colorbar(im, ax=ax1)
    opt_k_idx = cv_results['component_range'].index(cv_results['optimal_components'])
    opt_a_idx = cv_results['alpha_range'].index(cv_results['optimal_alpha'])
    ax1.scatter(opt_a_idx, opt_k_idx, color='red', s=100, marker='*', label='Optimal')
    ax1.legend()
    # Plot 2: Component spatial patterns
    ax2 = plt.subplot(3, 4, (3, 4))
    W = results['final_model']['W']
    im2 = ax2.imshow(W.T, aspect='auto', cmap='RdBu_r')
    ax2.set_title(f'Spatial Components (W)\n{W.shape[1]} components', fontsize=12)
    ax2.set_xlabel('Electrodes')
    ax2.set_ylabel('Components')
    plt.colorbar(im2, ax=ax2)
    # Plot 3: Temporal patterns sample
    ax3 = plt.subplot(3, 4, 5)
    H = results['final_model']['H']
    time_sample = slice(0, min(500, H.shape[1]))
    for i in range(min(3, H.shape[0])):
        ax3.plot(H[i, time_sample], label=f'Comp {i+1}', alpha=0.8)
    ax3.set_title('Temporal Components (H)\nFirst 3 Components', fontsize=12)
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Activation')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    # Plot 4: Cross-run correlations
    ax4 = plt.subplot(3, 4, 6)
    stats = results['statistics']
    valid_corrs = stats['cross_run']['valid_correlations']
    threshold = stats['cross_run']['shuffle_threshold']
    ax4.hist(valid_corrs, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax4.axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold ({threshold:.3f})')
    ax4.set_title('Cross-run Electrode Correlations', fontsize=12)
    ax4.set_xlabel('Correlation')
    ax4.set_ylabel('Count')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    # Plot 5: Electrode significance
    ax5 = plt.subplot(3, 4, 7)
    pair_counts = stats['cross_electrode']['electrode_pair_counts']
    thresholds = stats['cross_electrode']['null_thresholds']
    sorted_indices = np.argsort(pair_counts)[::-1]
    sorted_counts = pair_counts[sorted_indices]
    sorted_thresholds = thresholds[sorted_indices]
    x_range = range(len(sorted_counts))
    ax5.plot(x_range, sorted_counts, 'b-', linewidth=2, label='Observed', alpha=0.8)
    ax5.plot(x_range, sorted_thresholds, 'r--', linewidth=2, label='95% Threshold', alpha=0.8)
    ax5.fill_between(x_range, sorted_counts, sorted_thresholds,
                     where=(sorted_counts > sorted_thresholds),
                     color='green', alpha=0.3, label='Significant')
    ax5.set_title('Electrode Pair Significance', fontsize=12)
    ax5.set_xlabel('Electrodes (sorted by count)')
    ax5.set_ylabel('Significant Pair Count')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    # Plot 6: Reconstruction quality
    ax6 = plt.subplot(3, 4, 8)
    original_data = results['data_splits']['run1'] + results['data_splits']['run2']
    reconstruction = W @ H
    n_sample_electrodes = 5
    time_slice = slice(0, min(200, original_data.shape[1]))
    electrode_slice = slice(0, n_sample_electrodes)
    time_points = np.arange(time_slice.stop - time_slice.start)
    for i in range(n_sample_electrodes):
        ax6.plot(time_points, original_data[i, time_slice],
                alpha=0.7, linewidth=1, label=f'Orig {i+1}' if i < 2 else "")
        ax6.plot(time_points, reconstruction[i, time_slice],
                '--', alpha=0.7, linewidth=1, label=f'Recon {i+1}' if i < 2 else "")
    ax6.set_title(f'Reconstruction Quality\n(Sample of {n_sample_electrodes} electrodes)', fontsize=12)
    ax6.set_xlabel('Time')
    ax6.set_ylabel('Signal')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    # Plot 7: Component variance (if available)
    if 'components' in results:
        components = results['components']
        ax7 = plt.subplot(3, 4, 9)
        comp_vars = []
        comp_labels = []
        for i in range(W.shape[1]):
            comp_key = f'component_{i+1}'
            if comp_key in components:
                comp_vars.append(components[comp_key].get('variance_contribution', 0))
                comp_labels.append(f'C{i+1}')
        if comp_vars:
            ax7.bar(comp_labels, comp_vars, color='lightcoral', alpha=0.8)
            ax7.set_title('Component Variance Contributions', fontsize=12)
            ax7.set_xlabel('Component')
            ax7.set_ylabel('Variance Explained')
            ax7.grid(True, alpha=0.3)
    # Plot 8: Subject distribution
    ax8 = plt.subplot(3, 4, 10)
    subject_ids = results['statistics'].get('subject_ids', [])
    if subject_ids:
        subject_counts = {}
        for subj in subject_ids:
            subject_counts[subj] = subject_counts.get(subj, 0) + 1
        subjects = list(subject_counts.keys())
        counts = list(subject_counts.values())
        bars = ax8.bar(subjects, counts, color='lightgreen', alpha=0.8)
        ax8.set_title('Electrode Count by Subject', fontsize=12)
        ax8.set_xlabel('Subject')
        ax8.set_ylabel('Number of Electrodes')
        ax8.tick_params(axis='x', rotation=45)
        for bar, count in zip(bars, counts):
            ax8.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    str(count), ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    output_dir = Path("results/real_data_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "comprehensive_analysis.png", dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Comprehensive analysis plot saved to {output_dir / 'comprehensive_analysis.png'}")

if __name__ == "__main__":
    main()
