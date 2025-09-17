"""
Model Fitting Module - Final NNMF Model Training
Trains the final NNMF model using optimal hyperparameters from cross-validation.
"""

import numpy as np
from typing import Dict, Any
from .nnmf_solver import ConstrainedNNMFSolver
import time
import logging

logger = logging.getLogger(__name__)


class ModelFitter:
    """Fits final NNMF model with optimal hyperparameters."""
    
    def __init__(self):
        pass
    
    def fit_final_model(
        self,
        electrode_data: np.ndarray,
        optimal_components: int,
        optimal_alpha: float,
        init_reps: int,
        constraint_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Fit final NNMF model on full dataset using optimal hyperparameters.
        
        Args:
            electrode_data: [n_electrodes × total_time] full dataset
            optimal_components: Optimal number of components from CV
            optimal_alpha: Optimal regularization parameter from CV
            init_reps: Number of random initializations
            constraint_config: Configuration dictionary
            
        Returns:
            final_model: Dictionary containing final model results
        """
        n_electrodes, total_time = electrode_data.shape
        
        logger.info(f"Fitting final model: k={optimal_components}, alpha={optimal_alpha:.4f}")
        
        # Configure constraints for final model
        final_config = constraint_config.copy()
        final_config['regularization_strength'] = optimal_alpha
        
        # Initialize solver
        solver = ConstrainedNNMFSolver(final_config)
        
        # Fit final model with timing
        start_time = time.time()
        
        W_final, H_final, costs, best_init = solver.fit_constrained_nnmf(
            electrode_data, optimal_components, init_reps
        )
        
        training_time = time.time() - start_time
        
        # Compute final metrics
        reconstruction = W_final @ H_final
        reconstruction_error = np.sum((electrode_data - reconstruction) ** 2) / electrode_data.size
        
        # Compute variance explained
        total_variance = np.var(electrode_data)
        residual_variance = np.var(electrode_data - reconstruction)
        variance_explained = 1 - (residual_variance / total_variance)
        
        # Compute component statistics
        component_variances = np.var(H_final, axis=1)
        component_var_explained = component_variances / np.sum(component_variances)
        
        # Component sparsity measures
        spatial_sparsity = self._compute_sparsity(W_final)
        temporal_sparsity = self._compute_sparsity(H_final.T)  # Transpose for time × components
        
        # Create final model dictionary
        final_model = {
            'W': W_final,
            'H': H_final,
            'reconstruction': reconstruction,
            'n_components': optimal_components,
            'alpha': optimal_alpha,
            'reconstruction_error': reconstruction_error,
            'variance_explained': variance_explained,
            'component_variances': component_variances,
            'component_var_explained': component_var_explained,
            'spatial_sparsity': spatial_sparsity,
            'temporal_sparsity': temporal_sparsity,
            'training_time': training_time,
            'cost_trajectory': costs,
            'best_initialization': best_init,
            'final_cost': costs[-1] if len(costs) > 0 else np.inf,
            'config_used': final_config,
            'data_shape': electrode_data.shape
        }
        
        logger.info(f"Final model fitted successfully:")
        logger.info(f"  Training time: {training_time:.2f} seconds")
        logger.info(f"  Final reconstruction error: {reconstruction_error:.6f}")
        logger.info(f"  Variance explained: {variance_explained:.3f}")
        logger.info(f"  Mean spatial sparsity: {np.mean(spatial_sparsity):.3f}")
        logger.info(f"  Mean temporal sparsity: {np.mean(temporal_sparsity):.3f}")
        
        return final_model
    
    def _compute_sparsity(self, matrix: np.ndarray) -> np.ndarray:
        """
        Compute sparsity measure for each column of the matrix.
        Sparsity = (sqrt(n) - L1/L2) / (sqrt(n) - 1)
        where n is the number of elements, L1 is L1 norm, L2 is L2 norm.
        
        Args:
            matrix: Input matrix [features × components]
            
        Returns:
            sparsity: Sparsity measure for each component [components]
        """
        n_features = matrix.shape[0]
        if n_features <= 1:
            return np.zeros(matrix.shape[1])
        
        sparsity = np.zeros(matrix.shape[1])
        
        for comp in range(matrix.shape[1]):
            col = matrix[:, comp]
            l1_norm = np.sum(np.abs(col))
            l2_norm = np.sqrt(np.sum(col ** 2))
            
            if l2_norm > 0:
                sparsity[comp] = (np.sqrt(n_features) - l1_norm/l2_norm) / (np.sqrt(n_features) - 1)
            else:
                sparsity[comp] = 0
        
        # Clamp to [0, 1] range
        sparsity = np.clip(sparsity, 0, 1)
        
        return sparsity