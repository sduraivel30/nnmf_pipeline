import numpy as np
from typing import Dict, Any
from .nnmf_solver import ConstrainedNNMFSolver
# Assuming time splitting handled internally in this function or else implement here as needed
from .data_splitting import split_stories_into_runs  
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class CrossValidator:
    """Cross-validation for NNMF hyperparameter search with time-stratified splits."""

    def __init__(self, n_jobs: int = -1):
        self.n_jobs = n_jobs

    def run_cross_validation(
        self,
        electrode_data: np.ndarray,
        story_lengths: np.ndarray,
        max_components: int,
        cv_splits: int,
        init_reps: int,
        test_frac: float,
        constraint_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run cross-validation search stratified across the time dimension.

        Args:
            electrode_data: [n_electrodes × total_time] full electrode data
            story_lengths: [n_stories] story length array
            max_components: Maximum number of components to test
            cv_splits: Number of CV splits
            init_reps: Number of random initializations per fit
            test_frac: Fraction of time points held out for testing
            constraint_config: Constraint configuration dict

        Returns:
            cv_results: Cross-validation results dict
        """
        n_electrodes, total_time = electrode_data.shape
        n_stories = len(story_lengths)
        component_range = list(range(2, max_components + 1))
        alpha_range = constraint_config['alpha_range']

        # Initialize results storage
        cv_results = np.full((len(component_range), len(alpha_range), cv_splits), np.inf)
        recon_errors = np.full((len(component_range), len(alpha_range), cv_splits), np.inf)
        split_reliability = np.zeros((len(component_range), len(alpha_range), cv_splits))

        logger.info(f"Starting time-stratified cross-validation: {len(component_range)} components × {len(alpha_range)} alphas × {cv_splits} splits")

        all_time_indices = np.arange(total_time)

        for split in range(cv_splits):
            logger.info(f"CV Split {split + 1}/{cv_splits}")

            # Shuffle and split time indices
            np.random.shuffle(all_time_indices)
            n_test = int(total_time * test_frac)
            test_idx = np.sort(all_time_indices[:n_test])
            train_idx = np.sort(all_time_indices[n_test:])

            train_data = electrode_data[:, train_idx]
            test_data = electrode_data[:, test_idx]

            for k_idx, k in enumerate(tqdm(component_range, desc=f"Split {split+1} Components")):
                for a_idx, alpha in enumerate(alpha_range):
                    try:
                        current_config = constraint_config.copy()
                        current_config['regularization_strength'] = alpha

                        solver = ConstrainedNNMFSolver(current_config)

                        # Fit NNMF on train data
                        W_train, H_train, costs, _ = solver.fit_constrained_nnmf(
                            train_data, k, init_reps
                        )

                        # Project test data temporally
                        W_test, H_test = solver.project_to_test_data(test_data, W_train)

                        test_recon = W_test @ H_test
                        recon_error = np.sum((test_data - test_recon) ** 2) / test_data.size

                        # Split train_data temporally into halves for reliability
                        split_point = train_data.shape[1] // 2
                        run1_train = train_data[:, :split_point]
                        run2_train = train_data[:, split_point:]

                        reliability = solver.compute_split_half_reliability(
                            W_train, H_train, run1_train, run2_train
                        )

                        # Store results
                        cv_results[k_idx, a_idx, split] = costs[-1]
                        recon_errors[k_idx, a_idx, split] = recon_error
                        split_reliability[k_idx, a_idx, split] = reliability

                    except Exception as e:
                        logger.warning(f"CV failed for k={k}, alpha={alpha:.4f}, split={split}: {str(e)}")
                        cv_results[k_idx, a_idx, split] = np.inf
                        recon_errors[k_idx, a_idx, split] = np.inf
                        split_reliability[k_idx, a_idx, split] = 0.0

        mean_recon_errors = np.mean(recon_errors, axis=2)
        mean_reliability = np.mean(split_reliability, axis=2)

        reliability_term = mean_reliability + 1e-8  # avoid division by zero
        composite_score = mean_recon_errors / reliability_term

        valid_mask = np.isfinite(composite_score)
        if not np.any(valid_mask):
            raise RuntimeError("All cross-validation attempts failed")

        flat_min_idx = np.argmin(composite_score[valid_mask])
        valid_indices = np.argwhere(valid_mask)
        opt_k_idx, opt_a_idx = valid_indices[flat_min_idx]

        cv_dict = {
            'optimal_components': component_range[opt_k_idx],
            'optimal_alpha': alpha_range[opt_a_idx],
            'cv_results': cv_results,
            'recon_errors': recon_errors,
            'split_reliability': split_reliability,
            'mean_recon_errors': mean_recon_errors,
            'mean_reliability': mean_reliability,
            'composite_score': composite_score,
            'component_range': component_range,
            'alpha_range': alpha_range
        }

        logger.info(f"Time-stratified cross-validation complete. Optimal: k={cv_dict['optimal_components']}, alpha={cv_dict['optimal_alpha']:.4f}")

        return cv_dict

    def plot_cv_results(self, cv_results: Dict[str, Any]) -> None:
        """Plot cross-validation results."""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        im1 = axes[0, 0].imshow(
            cv_results['mean_recon_errors'],
            aspect='auto',
            cmap='viridis'
        )
        axes[0, 0].set_title('Mean Reconstruction Errors')
        axes[0, 0].set_xlabel('Alpha Index')
        axes[0, 0].set_ylabel('Component Index')
        plt.colorbar(im1, ax=axes[0, 0])

        im2 = axes[0, 1].imshow(
            cv_results['mean_reliability'],
            aspect='auto',
            cmap='viridis'
        )
        axes[0, 1].set_title('Mean Split-Half Reliability')
        axes[0, 1].set_xlabel('Alpha Index')
        axes[0, 1].set_ylabel('Component Index')
        plt.colorbar(im2, ax=axes[0, 1])

        im3 = axes[1, 0].imshow(
            cv_results['composite_score'],
            aspect='auto',
            cmap='viridis'
        )
        axes[1, 0].set_title('Composite Score (Lower = Better)')
        axes[1, 0].set_xlabel('Alpha Index')
        axes[1, 0].set_ylabel('Component Index')
        plt.colorbar(im3, ax=axes[1, 0])

        component_range = cv_results['component_range']
        alpha_range = cv_results['alpha_range']
        if not isinstance(alpha_range, list):
            alpha_range = alpha_range.tolist()
        opt_alpha_idx = alpha_range.index(cv_results['optimal_alpha'])

        axes[1, 1].plot(
            component_range,
            cv_results['mean_recon_errors'][:, opt_alpha_idx],
            'b-o', label='Reconstruction Error'
        )
        axes[1, 1].set_xlabel('Number of Components')
        axes[1, 1].set_ylabel('Mean Reconstruction Error')
        axes[1, 1].set_title(f'Optimal Alpha = {cv_results["optimal_alpha"]:.4f}')
        axes[1, 1].axvline(
            cv_results['optimal_components'],
            color='red',
            linestyle='--',
            label='Optimal K'
        )
        axes[1, 1].legend()
        axes[1, 1].grid(True)

        plt.tight_layout()
        plt.show()
