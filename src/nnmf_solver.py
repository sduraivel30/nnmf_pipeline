"""
NNMF Solver Module - Constrained Non-negative Matrix Factorization
Implements constrained NNMF using scikit-learn with custom constraints.
"""

import numpy as np
from sklearn.decomposition import NMF
from sklearn.metrics import mean_squared_error
from scipy.optimize import nnls
from typing import Tuple, Dict, Any, Optional
import logging
from joblib import Parallel, delayed

logger = logging.getLogger(__name__)


class ConstrainedNNMFSolver:
    """NNMF solver with multiple initializations and constraints."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.max_iter = config.get('max_iter', 200)
        self.tolerance = config.get('tolerance', 1e-6)
        self.alpha = config.get('regularization_strength', 0.0)
        self.l1_ratio = config.get('l1_ratio', 0.0)  # L1 vs L2 regularization mix
        self.init_method = config.get('init_method', 'nndsvd')
        
    def fit_constrained_nnmf(
        self, 
        data: np.ndarray, 
        n_components: int, 
        n_init_reps: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """
        Fit constrained NNMF with multiple random initializations.
        
        Args:
            data: [n_features × n_samples] data matrix
            n_components: Number of components
            n_init_reps: Number of random initializations
            
        Returns:
            W: [n_features × n_components] spatial factors
            H: [n_components × n_samples] temporal factors  
            costs: Cost trajectory for best initialization
            best_init: Index of best initialization
        """
        n_features, n_samples = data.shape
        
        best_cost = np.inf
        best_W = None
        best_H = None  
        best_init = 0
        best_costs = None
        
        logger.info(f\"Fitting NNMF with {n_init_reps} initializations, k={n_components}\")\n        \n        # Parallel initialization fitting\n        results = Parallel(n_jobs=-1)(\n            delayed(self._fit_single_init)(data, n_components, rep) \n            for rep in range(n_init_reps)\n        )\n        \n        # Find best initialization\n        for rep, (W, H, costs, final_cost) in enumerate(results):\n            if W is not None and final_cost < best_cost:\n                best_cost = final_cost\n                best_W = W\n                best_H = H\n                best_costs = costs\n                best_init = rep\n        \n        if best_W is None:\n            raise RuntimeError(\"All NNMF initializations failed\")\n            \n        logger.info(f\"Best initialization: {best_init} with cost {best_cost:.6f}\")\n        \n        return best_W, best_H, best_costs, best_init\n    \n    def _fit_single_init(\n        self, \n        data: np.ndarray, \n        n_components: int, \n        rep: int\n    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], float]:\n        \"\"\"Fit single NNMF initialization.\"\"\"\n        try:\n            # Use sklearn NMF with regularization\n            model = NMF(\n                n_components=n_components,\n                init=self.init_method,\n                max_iter=self.max_iter,\n                tol=self.tolerance,\n                alpha_W=self.alpha,\n                alpha_H=self.alpha,\n                l1_ratio=self.l1_ratio,\n                random_state=rep,\n                solver='mu'  # Multiplicative update solver\n            )\n            \n            # Fit the model\n            W = model.fit_transform(data.T).T  # sklearn expects [samples × features]\n            H = model.components_\n            \n            # Compute reconstruction cost\n            reconstruction = W @ H\n            mse_cost = mean_squared_error(data, reconstruction)\n            reg_cost = self.alpha * (np.sum(W**2) + np.sum(H**2))\n            total_cost = mse_cost + reg_cost\n            \n            # Create cost trajectory (simplified)\n            costs = np.array([total_cost])  # sklearn doesn't return trajectory\n            \n            return W, H, costs, total_cost\n            \n        except Exception as e:\n            logger.warning(f\"NNMF initialization {rep} failed: {str(e)}\")\n            return None, None, None, np.inf\n    \n    def project_to_test_data(\n        self, \n        test_data: np.ndarray, \n        W_train: np.ndarray\n    ) -> Tuple[np.ndarray, np.ndarray]:\n        \"\"\"\n        Project test data onto trained NNMF factors using Norman-Haignere method.\n        \n        Based on Norman-Haignere et al. 2022: \"Component electrode weights were \n        fixed to those estimated from the training data, and we estimated a new \n        set of component response time courses\"\n        \n        Args:\n            test_data: [n_electrodes × n_samples] test data matrix\n            W_train: [n_electrodes × n_components] trained spatial factors\n            \n        Returns:\n            W_test: [n_electrodes × n_components] spatial factors (same as W_train)\n            H_test: [n_components × n_samples] temporal factors for test data\n        \"\"\"\n        # Fix spatial weights from training\n        W_test = W_train.copy()\n        \n        # Solve for H using non-negative least squares\n        n_components, n_samples = W_train.shape[1], test_data.shape[1]\n        H_test = np.zeros((n_components, n_samples))\n        \n        # Solve column-wise using NNLS\n        for t in range(n_samples):\n            H_test[:, t], _ = nnls(W_test, test_data[:, t])\n        \n        return W_test, H_test\n    \n    def compute_split_half_reliability(\n        self,\n        W: np.ndarray,\n        H: np.ndarray, \n        run1: np.ndarray,\n        run2: np.ndarray\n    ) -> float:\n        \"\"\"\n        Compute split-half reliability using Norman-Haignere method.\n        \n        Args:\n            W: [n_electrodes × n_components] spatial factors\n            H: [n_components × n_samples] temporal factors\n            run1: [n_electrodes × n_samples1] first run data\n            run2: [n_electrodes × n_samples2] second run data\n            \n        Returns:\n            reliability: Scalar reliability measure\n        \"\"\"\n        try:\n            # Project each run onto learned components using fixed weights\n            _, H1 = self.project_to_test_data(run1, W)\n            _, H2 = self.project_to_test_data(run2, W)\n            \n            # Compute split-half correlation for each component\n            n_components = W.shape[1]\n            component_reliabilities = np.zeros(n_components)\n            \n            for comp in range(n_components):\n                if np.std(H1[comp]) > 0 and np.std(H2[comp]) > 0:\n                    component_reliabilities[comp] = np.corrcoef(\n                        H1[comp], H2[comp]\n                    )[0, 1]\n            \n            # Average reliability across components\n            valid_reliabilities = component_reliabilities[\n                ~np.isnan(component_reliabilities)\n            ]\n            \n            if len(valid_reliabilities) > 0:\n                reliability = np.mean(valid_reliabilities)\n                # Apply Spearman-Brown correction for split-half reliability\n                reliability = (2 * reliability) / (1 + reliability)\n            else:\n                reliability = 0.0\n            \n            # Ensure reliability is in valid range\n            reliability = np.clip(reliability, 0, 1)\n            \n            return reliability\n            \n        except Exception as e:\n            logger.warning(f\"Split-half reliability computation failed: {str(e)}\")\n            return 0.0


def define_constraint_config(\n    stat_analysis: Dict[str, Any], \n    min_story_length: int\n) -> Dict[str, Any]:\n    \"\"\"\n    Define constraint configuration based on data characteristics.\n    \n    Args:\n        stat_analysis: Statistical analysis results\n        min_story_length: Minimum story length for constraints\n        \n    Returns:\n        constraint_config: Configuration dict for NNMF constraints\n    \"\"\"\n    # Extract reliability metrics\n    reliability_fraction = (\n        stat_analysis.get('cross_run', {}).get('n_significant', 0) / \n        len(stat_analysis.get('cross_run', {}).get('correlations', [1]))\n    )\n    \n    config = {\n        # Basic parameters\n        'max_iter': 200,\n        'tolerance': 1e-6,\n        'init_method': 'nndsvd',\n        \n        # Regularization based on data reliability\n        'alpha_range': [0.0, 0.001, 0.01, 0.1, 0.5],\n        'l1_ratio': 0.0,  # Pure L2 regularization\n        \n        # Adaptive regularization\n        'adaptive_alpha': True,\n        'min_component_length': max(10, min_story_length // 10),\n        \n        # Component validity criteria\n        'min_component_variance': 1e-6,\n        'min_component_correlation': 0.05\n    }\n    \n    # Adjust regularization based on data quality\n    if reliability_fraction > 0.7:\n        config['regularization_strength'] = 0.01  # Light regularization for reliable data\n    else:\n        config['regularization_strength'] = 0.1   # Stronger regularization for noisy data\n    \n    logger.info(f\"Constraint configuration: alpha={config['regularization_strength']}, reliability={reliability_fraction:.3f}\")\n    \n    return config