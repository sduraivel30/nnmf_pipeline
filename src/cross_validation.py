"""
Cross-Validation Module - Norman-Haignere Methodology
Electrode-stratified cross-validation for NNMF hyperparameter optimization.
"""

import numpy as np
from typing import Dict, Any, Tuple
from .nnmf_solver import ConstrainedNNMFSolver
from .data_splitting import create_electrode_stratified_split, split_stories_into_runs
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class CrossValidator:
    """Cross-validation for NNMF hyperparameter search."""
    
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
        Run cross-validation search over (k, alpha) parameter space.
        
        Args:
            electrode_data: [n_electrodes × total_time] full electrode data
            story_lengths: [n_stories] story length array
            max_components: Maximum number of components to test
            cv_splits: Number of CV splits
            init_reps: Number of random initializations per fit
            test_frac: Fraction of data held out for testing
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
        
        logger.info(f\"Starting cross-validation: {len(component_range)} components × {len(alpha_range)} alphas × {cv_splits} splits\")\n        \n        # Cross-validation loop\n        for split in range(cv_splits):\n            logger.info(f\"CV Split {split + 1}/{cv_splits}\")\n            \n            # Create electrode-stratified train/test split\n            train_idx, test_idx = create_electrode_stratified_split(n_electrodes, test_frac)\n            \n            train_data = electrode_data[train_idx, :]\n            test_data = electrode_data[test_idx, :]\n            \n            # Test each component count\n            for k_idx, k in enumerate(tqdm(component_range, desc=f\"Split {split+1} Components\")):\n                \n                # Test each alpha value\n                for a_idx, alpha in enumerate(alpha_range):\n                    \n                    try:\n                        # Configure constraints for this alpha\n                        current_config = constraint_config.copy()\n                        current_config['regularization_strength'] = alpha\n                        \n                        # Initialize solver\n                        solver = ConstrainedNNMFSolver(current_config)\n                        \n                        # Fit NNMF on training data\n                        W_train, H_train, costs, _ = solver.fit_constrained_nnmf(\n                            train_data, k, init_reps\n                        )\n                        \n                        # Evaluate on test data using Norman-Haignere projection\n                        W_test, H_test = solver.project_to_test_data(test_data, W_train)\n                        \n                        # Compute reconstruction error on test set\n                        test_recon = W_test @ H_test\n                        recon_error = np.sum((test_data - test_recon) ** 2) / test_data.size\n                        \n                        # Compute split-half reliability using story splits\n                        run1_train, run2_train, _ = split_stories_into_runs(\n                            train_data, story_lengths\n                        )\n                        \n                        reliability = solver.compute_split_half_reliability(\n                            W_train, H_train, run1_train, run2_train\n                        )\n                        \n                        # Store results\n                        cv_results[k_idx, a_idx, split] = costs[-1]\n                        recon_errors[k_idx, a_idx, split] = recon_error\n                        split_reliability[k_idx, a_idx, split] = reliability\n                        \n                    except Exception as e:\n                        logger.warning(f\"CV failed for k={k}, alpha={alpha:.4f}, split={split}: {str(e)}\")\n                        cv_results[k_idx, a_idx, split] = np.inf\n                        recon_errors[k_idx, a_idx, split] = np.inf\n                        split_reliability[k_idx, a_idx, split] = 0.0\n        \n        # Find optimal parameters\n        mean_recon_errors = np.mean(recon_errors, axis=2)\n        mean_reliability = np.mean(split_reliability, axis=2)\n        \n        # Composite score: minimize reconstruction error, maximize reliability\n        # Handle division by zero\n        reliability_term = mean_reliability + 1e-8\n        composite_score = mean_recon_errors / reliability_term\n        \n        # Find minimum (excluding infinite values)\n        valid_mask = np.isfinite(composite_score)\n        if not np.any(valid_mask):\n            raise RuntimeError(\"All cross-validation attempts failed\")\n            \n        min_idx = np.unravel_index(\n            np.argmin(composite_score[valid_mask]), \n            composite_score.shape\n        )\n        opt_k_idx, opt_a_idx = min_idx\n        \n        # Compile results\n        cv_dict = {\n            'optimal_components': component_range[opt_k_idx],\n            'optimal_alpha': alpha_range[opt_a_idx],\n            'cv_results': cv_results,\n            'recon_errors': recon_errors,\n            'split_reliability': split_reliability,\n            'mean_recon_errors': mean_recon_errors,\n            'mean_reliability': mean_reliability,\n            'composite_score': composite_score,\n            'component_range': component_range,\n            'alpha_range': alpha_range\n        }\n        \n        logger.info(f\"Cross-validation complete. Optimal: k={cv_dict['optimal_components']}, alpha={cv_dict['optimal_alpha']:.4f}\")\n        \n        return cv_dict\n    \n    def plot_cv_results(self, cv_results: Dict[str, Any]) -> None:\n        \"\"\"Plot cross-validation results.\"\"\"\n        import matplotlib.pyplot as plt\n        \n        fig, axes = plt.subplots(2, 2, figsize=(12, 10))\n        \n        # Plot 1: Mean reconstruction errors\n        im1 = axes[0, 0].imshow(\n            cv_results['mean_recon_errors'], \n            aspect='auto', \n            cmap='viridis'\n        )\n        axes[0, 0].set_title('Mean Reconstruction Errors')\n        axes[0, 0].set_xlabel('Alpha Index')\n        axes[0, 0].set_ylabel('Component Index')\n        plt.colorbar(im1, ax=axes[0, 0])\n        \n        # Plot 2: Mean reliability\n        im2 = axes[0, 1].imshow(\n            cv_results['mean_reliability'], \n            aspect='auto', \n            cmap='viridis'\n        )\n        axes[0, 1].set_title('Mean Split-Half Reliability')\n        axes[0, 1].set_xlabel('Alpha Index')\n        axes[0, 1].set_ylabel('Component Index')\n        plt.colorbar(im2, ax=axes[0, 1])\n        \n        # Plot 3: Composite score\n        im3 = axes[1, 0].imshow(\n            cv_results['composite_score'], \n            aspect='auto', \n            cmap='viridis'\n        )\n        axes[1, 0].set_title('Composite Score (Lower = Better)')\n        axes[1, 0].set_xlabel('Alpha Index')\n        axes[1, 0].set_ylabel('Component Index')\n        plt.colorbar(im3, ax=axes[1, 0])\n        \n        # Plot 4: Optimal parameter trace\n        component_range = cv_results['component_range']\n        alpha_range = cv_results['alpha_range']\n        \n        # Plot reconstruction error vs components for optimal alpha\n        opt_alpha_idx = alpha_range.index(cv_results['optimal_alpha'])\n        axes[1, 1].plot(\n            component_range, \n            cv_results['mean_recon_errors'][:, opt_alpha_idx],\n            'b-o', label='Reconstruction Error'\n        )\n        axes[1, 1].set_xlabel('Number of Components')\n        axes[1, 1].set_ylabel('Mean Reconstruction Error')\n        axes[1, 1].set_title(f'Optimal Alpha = {cv_results[\"optimal_alpha\"]:.4f}')\n        axes[1, 1].axvline(\n            cv_results['optimal_components'], \n            color='red', \n            linestyle='--', \n            label='Optimal K'\n        )\n        axes[1, 1].legend()\n        axes[1, 1].grid(True)\n        \n        plt.tight_layout()\n        plt.show()