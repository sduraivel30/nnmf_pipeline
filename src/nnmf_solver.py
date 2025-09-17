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
        self.init_method = config.get('init_method', 'nnsvda')

    def fit_constrained_nnmf(
        self,
        data: np.ndarray,
        n_components: int,
        n_init_reps: int
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], int]:
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

        logger.info(f"Fitting NNMF with {n_init_reps} initializations, k={n_components}")

        # Parallel initialization fitting
        results = Parallel(n_jobs=-1)(
            delayed(self._fit_single_init)(data, n_components, rep) 
            for rep in range(n_init_reps)
        )

        # Find best initialization
        for rep, (W, H, costs, final_cost) in enumerate(results):
            if W is not None and final_cost < best_cost:
                best_cost = final_cost
                best_W = W
                best_H = H
                best_costs = costs
                best_init = rep

        if best_W is None:
            raise RuntimeError("All NNMF initializations failed")

        logger.info(f"Best initialization: {best_init} with cost {best_cost:.6f}")

        return best_W, best_H, best_costs, best_init

    def _fit_single_init(
        self,
        data: np.ndarray,
        n_components: int,
        rep: int
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], float]:
        """Fit single NNMF initialization."""
        try:
            # Use sklearn NMF with regularization
            model = NMF(
                n_components=n_components,
                init=self.init_method,
                max_iter=self.max_iter,
                tol=self.tolerance,
                alpha_W=self.alpha,
                alpha_H=self.alpha,
                l1_ratio=self.l1_ratio,
                random_state=rep,
                solver='mu'  # Multiplicative update solver
            )

            # Fit the model
            W = model.fit_transform(data)  # sklearn expects [samples × features]
            H = model.components_
           # logger.info(f"W shape: {W.shape}, H shape: {H.shape}, data shape: {data.shape}")
    
            # Compute reconstruction cost
            reconstruction = W @ H
          #  logger.info(f"reconstruction shape: {reconstruction.shape}")
    
            mse_cost = mean_squared_error(data, reconstruction)
            reg_cost = self.alpha * (np.sum(W**2) + np.sum(H**2))
            total_cost = mse_cost + reg_cost
            
            # Create cost trajectory (simplified)
            costs = np.array([total_cost])  # sklearn doesn't return trajectory

            return W, H, costs, total_cost

        except Exception as e:
            logger.warning(f"NNMF initialization {rep} failed: {str(e)}")
            return None, None, None, np.inf

    def project_to_test_data(
        self,
        test_data: np.ndarray,
        W_train: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Project test data onto trained NNMF factors using Norman-Haignere method.

        Based on Norman-Haignere et al. 2022: "Component electrode weights were 
        fixed to those estimated from the training data, and we estimated a new 
        set of component response time courses"

        Args:
            test_data: [n_electrodes × n_samples] test data matrix
            W_train: [n_electrodes × n_components] trained spatial factors

        Returns:
            W_test: [n_electrodes × n_components] spatial factors (same as W_train)
            H_test: [n_components × n_samples] temporal factors for test data
        """
        # Fix spatial weights from training
        W_test = W_train.copy()

        # Solve for H using non-negative least squares
        n_components, n_samples = W_train.shape[1], test_data.shape[1]
        H_test = np.zeros((n_components, n_samples))

        # Solve column-wise using NNLS
        for t in range(n_samples):
            H_test[:, t], _ = nnls(W_test, test_data[:, t])

        return W_test, H_test

    def compute_split_half_reliability(
        self,
        W: np.ndarray,
        H: np.ndarray,
        run1: np.ndarray,
        run2: np.ndarray
    ) -> float:
        """
        Compute split-half reliability using Norman-Haignere method.

        Args:
            W: [n_electrodes × n_components] spatial factors
            H: [n_components × n_samples] temporal factors
            run1: [n_electrodes × n_samples1] first run data
            run2: [n_electrodes × n_samples2] second run data

        Returns:
            reliability: Scalar reliability measure
        """
        try:
            # Project each run onto learned components using fixed weights
            _, H1 = self.project_to_test_data(run1, W)
            _, H2 = self.project_to_test_data(run2, W)

            # Compute split-half correlation for each component
            n_components = W.shape[1]
            component_reliabilities = np.zeros(n_components)

            for comp in range(n_components):
                if np.std(H1[comp]) > 0 and np.std(H2[comp]) > 0:
                    component_reliabilities[comp] = np.corrcoef(
                        H1[comp], H2[comp]
                    )[0, 1]

            # Average reliability across components
            valid_reliabilities = component_reliabilities[
                ~np.isnan(component_reliabilities)
            ]

            if len(valid_reliabilities) > 0:
                reliability = np.mean(valid_reliabilities)
                # Apply Spearman-Brown correction for split-half reliability
                reliability = (2 * reliability) / (1 + reliability)
            else:
                reliability = 0.0

            # Ensure reliability is in valid range
            reliability = np.clip(reliability, 0, 1)

            return reliability

        except Exception as e:
            logger.warning(f"Split-half reliability computation failed: {str(e)}")
            return 0.0
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

def define_constraint_config(
    stat_analysis: Dict[str, Any],
    min_story_length: int
) -> Dict[str, Any]:
    """
    Define constraint configuration based on data characteristics.

    Args:
        stat_analysis: Statistical analysis results
        min_story_length: Minimum story length for constraints

    Returns:
        constraint_config: Configuration dict for NNMF constraints
    """
    # Extract reliability metrics
    cross_run = stat_analysis.get('cross_run', {})
    n_significant = cross_run.get('n_significant', 0)
    correlations = cross_run.get('correlations', [1])
    reliability_fraction = n_significant / len(correlations) if correlations else 0.0

    config = {
        # Basic parameters
        'max_iter': 200,
        'tolerance': 1e-6,
        'init_method': 'nndsvd',

        # Regularization based on data reliability
        'alpha_range': [0.0, 0.001, 0.01, 0.1, 0.5],
        'l1_ratio': 0.0,  # Pure L2 regularization

        # Adaptive regularization
        'adaptive_alpha': True,
        'min_component_length': max(10, min_story_length // 10),

        # Component validity criteria
        'min_component_variance': 1e-6,
        'min_component_correlation': 0.05
    }

    # Adjust regularization based on data quality
    if reliability_fraction > 0.7:
        config['regularization_strength'] = 0.01  # Light regularization for reliable data
    else:
        config['regularization_strength'] = 0.1   # Stronger regularization for noisy data

    logger.info(f"Constraint configuration: alpha={config['regularization_strength']}, reliability={reliability_fraction:.3f}")

    return config
