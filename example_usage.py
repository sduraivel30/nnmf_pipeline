import numpy as np
from pathlib import Path
import logging
from src import NNMFPipeline
import matplotlib.pyplot as plt
import seaborn as sns

# Setup logging
logging.basicConfig(level=logging.INFO)


def generate_synthetic_data():
    """Generate synthetic electrode data with realistic structure."""
    np.random.seed(42)
    n_electrodes = 128
    total_time = 10000
    story_lengths = np.array([total_time], dtype=int)  # Single story
    subject_distribution = [32, 40, 28, 28]  # Electrodes per subject

    subject_ids = []
    for i, count in enumerate(subject_distribution):
        subject_ids.extend([f'Subject_{i+1}'] * count)

    n_true_components = 7
    W_true = np.random.exponential(0.8, (n_electrodes, n_true_components))

    current_electrode = 0
    for subj_idx, count in enumerate(subject_distribution):
        subject_pattern = np.random.exponential(1, (count, n_true_components))
        subject_pattern[:, subj_idx] *= 2  # Enhance one component per subject
        W_true[current_electrode:current_electrode + count] = subject_pattern
        current_electrode += count

    H_true = np.random.exponential(0.5, (n_true_components, total_time))
    for comp in range(n_true_components):
        t = np.linspace(0, 8 * np.pi, total_time)
        H_true[comp] *= (1 + 0.4 * np.sin(t * (comp + 1) / n_true_components))
        trend = np.exp(-0.5 * (np.linspace(-2, 2, total_time) ** 2))
        H_true[comp] *= (1 + 0.3 * trend)

    electrode_data = W_true @ H_true
    electrode_data += 0.2 * np.random.randn(n_electrodes, total_time)  # Gaussian noise

    current_electrode = 0
    for subj_idx, count in enumerate(subject_distribution):
        common_noise = 0.1 * np.random.randn(1, total_time)
        electrode_data[current_electrode:current_electrode + count] += common_noise
        current_electrode += count

    # Temporal artifacts (line noise)
    line_noise = 0.05 * np.sin(2 * np.pi * 60 * np.linspace(0, 10, total_time))  # 60Hz
    electrode_data += line_noise[np.newaxis, :]

    print("Generated synthetic data with realistic structure")
    plt.figure(figsize=(10, 6))
    plt.imshow(electrode_data, aspect='auto', cmap='viridis')
    plt.colorbar(label='Amplitude')
    plt.title('Synthetic Electrode Data (Electrodes × Time)')
    plt.xlabel('Time')
    plt.ylabel('Electrodes')
    plt.show()

    return electrode_data, story_lengths, subject_ids, W_true, H_true


def preprocess_data_for_nmf(
    data,
    method='baseline_shift',
    preserve_dynamics=True,
    baseline_percentile=10.0
):
    """Preprocess z-scored HG data for NNMF input, ensuring non-negativity."""
    data = data.copy()
    print(f"Input data stats: min={data.min():.3f}, max={data.max():.3f}")

    if method == 'baseline_shift':
        min_val = data.min()
        if min_val < 0:
            data -= min_val
        print(f"Applied baseline shift: added {-min_val:.3f}")

    elif method == 'percentile_shift':
        baseline = np.percentile(data, baseline_percentile, axis=1, keepdims=True)
        data -= baseline
        data = np.clip(data, 0, None)
        print(f"Applied percentile shift: used {baseline_percentile}th percentile baseline")

    elif method == 'rectified_linear':
        negatives = np.mean(data < 0)
        data = np.maximum(data, 0)
        print(f"Applied ReLU (set negative values to zero), zeroed {negatives*100:.1f}% of data")

    elif method == 'exponential':
        data = np.exp(data)
        print("Applied exponential transform")

    elif method == 'softplus':
        data = np.log(1 + np.exp(data))
        print("Applied softplus transform")

    elif method == 'square_plus':
        pos_mask = data >= 0
        data = data ** 2 + 0.1 * pos_mask
        print("Applied square plus shift transform")

    else:
        raise ValueError("Unknown preprocessing method")

    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    if np.any(data < 0):
        data = np.clip(data, 0, None)
    assert np.all(data >= 0), "Data must be non-negative after preprocessing"

    if preserve_dynamics:
        data += 1e-8

    print(f"Output data stats: min={data.min():.6f}, max={data.max():.3f}")
    return data


def plot_reconstruction_vs_components(results, W_true=None, H_true=None):
    """Plot train/test reconstruction and compare true vs predicted components at optimal k."""
    import matplotlib.pyplot as plt

    cv = results['cross_validation']
    optimal_k = results['final_model']['n_components']
    opt_alpha = results['final_model']['alpha']

    comp_range = cv['component_range']
    alpha_idx = cv['alpha_range'].index(opt_alpha)

    mean_train_costs = results.get('train_costs')  # optional field, add in your pipeline if possible
    mean_test_errors = cv['mean_recon_errors'][:, alpha_idx]

    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    if mean_train_costs is not None:
        plt.plot(comp_range, mean_train_costs[:, alpha_idx], label='Training Reconstruction Cost', marker='o')
    plt.plot(comp_range, mean_test_errors, label='Test Reconstruction Error', marker='x')
    plt.axvline(optimal_k, color='r', linestyle='--', label=f'Optimal Components: {optimal_k}')
    plt.xlabel("Number of Components")
    plt.ylabel("Reconstruction Error")
    plt.title("Reconstruction Error vs Number of Components")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    if W_true is not None and H_true is not None:
        # For the optimal component, plot the true and predicted temporal activities
        predicted_H = results['final_model']['H'][:optimal_k, :]
        true_H = H_true[:optimal_k, :]
        t = np.arange(min(true_H.shape[1], predicted_H.shape[1]))

        for i in range(min(optimal_k, 3)):  # plotting first 3 components
            plt.plot(t, true_H[i, t], label=f'True Comp {i+1}', alpha=0.6)
            plt.plot(t, predicted_H[i, t], linestyle='--', label=f'Predicted Comp {i+1}', alpha=0.8)
        plt.xlabel("Time")
        plt.ylabel("Component Activation")
        plt.title("True vs Predicted Temporal Components")
        plt.legend()
        plt.grid(True)
    else:
        plt.text(0.5, 0.5, "True components not available for comparison",
                 horizontalalignment='center', verticalalignment='center')
        plt.axis('off')

    plt.tight_layout()
    plt.show()

def analyze_results(results, n_electrodes):
    """Print a comprehensive summary of NNMF analysis results."""
    final_model = results.get('final_model', {})
    cv = results.get('cross_validation', {})
    stats = results.get('statistics', {})

    print("\n===== NNMF Analysis Summary =====")
    print(f"Optimal number of components: {final_model.get('n_components', 'N/A')}")
    print(f"Optimal alpha (regularization): {final_model.get('alpha', 'N/A'):.4f}")
    print(f"Variance explained by model: {final_model.get('variance_explained', 'N/A'):.3f}")

    mean_recon_err = cv.get('mean_recon_errors', None)
    mean_reliability = cv.get('mean_reliability', None)
    if mean_recon_err is not None and mean_reliability is not None:
        avg_recon_err = mean_recon_err.mean()
        avg_rel = mean_reliability.mean()
        print(f"Average CV reconstruction error: {avg_recon_err:.6f}")
        print(f"Average CV reliability: {avg_rel:.3f}")

    if stats:
        n_sig = stats.get('cross_run', {}).get('n_significant', 0)
        n_sig_perm = stats.get('cross_electrode', {}).get('n_significant_electrodes', 0)
        corr = stats.get('cross_electrode', {}).get('second_level_correlation', 0)
        print(f"Significant electrodes (Cross-run): {n_sig} / {n_electrodes} ({100 * n_sig / n_electrodes:.1f}%)")
        print(f"Significant electrodes (Permutation test): {n_sig_perm} / {n_electrodes} ({100 * n_sig_perm / n_electrodes:.1f}%)")
        print(f"Second-level correlation: {corr:.3f}")

    if 'components' in results:
        print("\nComponents spatial sparsity:")
        for i in range(final_model.get('n_components', 0)):
            key = f"component_{i+1}"
            comp = results['components'].get(key, {})
            sparsity = comp.get('spatial_sparsity', float('nan'))
            print(f"  Component {i+1}: sparsity = {sparsity:.3f}")

    subjects = stats.get('subject_ids', [])
    subj_counts = {}
    for s in subjects:
        subj_counts[s] = subj_counts.get(s, 0) + 1
    print(f"\nNumber of subjects: {len(subj_counts)}")
    for s, c in subj_counts.items():
        print(f"  {s}: {c} electrodes")

    print("=" * 40)


def create_detailed_visualizations(results):
    """Generate heatmaps and component plots summarizing NNMF results."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    cv = results.get('cross_validation', {})
    final_model = results.get('final_model', {})

    sns.set(style='whitegrid')
    plt.figure(figsize=(20, 15))

    # 1. CV composite score heatmap
    plt.subplot(3, 3, 1)
    composite = cv.get('composite_score', None)
    if composite is not None:
        sns.heatmap(composite, annot=False, cmap='viridis',
                    xticklabels=np.round(cv.get('alpha_range', []), decimals=3),
                    yticklabels=cv.get('component_range', []))
        plt.title("CV Composite Score (Lower is Better)")
        plt.xlabel("Alpha")
        plt.ylabel("Number of Components")

    # 2. Reconstruction error heatmap
    plt.subplot(3, 3, 2)
    recon_err = cv.get('mean_recon_errors', None)
    if recon_err is not None:
        sns.heatmap(recon_err, annot=False, cmap='magma',
                    xticklabels=np.round(cv.get('alpha_range', []), decimals=3),
                    yticklabels=cv.get('component_range', []))
        plt.title("Mean CV Reconstruction Error")
        plt.xlabel("Alpha")
        plt.ylabel("Number of Components")

    # 3. Reliability heatmap
    plt.subplot(3, 3, 3)
    rel = cv.get('mean_reliability', None)
    if rel is not None:
        sns.heatmap(rel, annot=False, cmap='coolwarm',
                    xticklabels=np.round(cv.get('alpha_range', []), decimals=3),
                    yticklabels=cv.get('component_range', []))
        plt.title("Mean CV Reliability")
        plt.xlabel("Alpha")
        plt.ylabel("Number of Components")

    # 4. Spatial components heatmap
    plt.subplot(3,3,4)
    W = final_model.get('W', None)
    if W is not None:
        sns.heatmap(W.T, cmap='coolwarm', center=0)
        plt.title(f"Spatial Components (W), {W.shape[1]} components")
        plt.xlabel("Electrode")
        plt.ylabel("Component")

    # 5. Sample temporal components
    plt.subplot(3, 3, 5)
    H = final_model.get('H', None)
    if H is not None:
        for i in range(min(3, H.shape[0])):
            plt.plot(H[i, :500], label=f"Comp {i+1}")
        plt.title("Temporal Components: First 3")
        plt.xlabel("Time")
        plt.ylabel("Activation")
        plt.legend()

    plt.tight_layout()
    plt.show()

def run():
    electrode_data, story_lengths, subject_ids, W_true, H_true = generate_synthetic_data()
    electrode_data = preprocess_data_for_nmf(electrode_data, method='baseline_shift', preserve_dynamics=True)

    pipeline = NNMFPipeline(n_permutations=50, n_jobs=-1, random_state=42)
    results = pipeline.run_complete_analysis(
        electrode_data=electrode_data,
        story_lengths=story_lengths,
        subject_ids=subject_ids,
        max_components=10,
        cv_splits=3,
        init_reps=5,
        test_frac=0.2
    )

    # Optionally attach ground truth to results for plotting
    results['components_true'] = {'W': W_true, 'H': H_true}

    # Print summary stats
    analyze_results(results, electrode_data.shape[0])

    # Plot standard detailed visuals from pipeline
    create_detailed_visualizations(results)

    # Custom plots for reconstruction & true-vs-estimated components
    plot_reconstruction_vs_components(results, W_true, H_true)

    print("Analysis complete.")


if __name__ == "__main__":
    run()
