"""
Baseline Comparison Module - Alternative Method Comparisons
Compares NNMF results against baseline methods (PCA, ICA, etc.).
"""

import numpy as np
from sklearn.decomposition import PCA, FastICA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from typing import Dict, Any, List, Tuple
import logging
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class BaselineComparator:
    """Compares NNMF against baseline dimensionality reduction methods."""
    
    def __init__(self):
        pass
    
    def run_baseline_comparisons(
        self,
        electrode_data: np.ndarray,
        story_lengths: np.ndarray,
        optimal_components: int,
        cv_splits: int,
        constraint_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compare NNMF against baseline methods.
        
        Args:
            electrode_data: [n_electrodes × total_time] data matrix
            story_lengths: [n_stories] story length array
            optimal_components: Number of components to use for comparison
            cv_splits: Number of CV splits for evaluation
            constraint_config: Configuration dictionary
            
        Returns:
            baseline_results: Dictionary with comparison results
        """
        logger.info(f"Running baseline comparisons with k={optimal_components}")
        
        baseline_results = {
            'n_components': optimal_components,
            'data_shape': electrode_data.shape,
            'methods': {}
        }
        
        # Method 1: Principal Component Analysis (PCA)
        logger.info("Running PCA baseline...")
        pca_results = self._run_pca_baseline(electrode_data, optimal_components, cv_splits)
        baseline_results['methods']['pca'] = pca_results
        
        # Method 2: Independent Component Analysis (ICA)
        logger.info("Running ICA baseline...")
        ica_results = self._run_ica_baseline(electrode_data, optimal_components, cv_splits)
        baseline_results['methods']['ica'] = ica_results
        
        # Method 3: K-Means clustering
        logger.info("Running K-Means baseline...")
        kmeans_results = self._run_kmeans_baseline(electrode_data, optimal_components, cv_splits)
        baseline_results['methods']['kmeans'] = kmeans_results
        
        # Method 4: Random projections
        logger.info("Running random projection baseline...")
        random_results = self._run_random_baseline(electrode_data, optimal_components, cv_splits)
        baseline_results['methods']['random'] = random_results
        
        # Summary comparison
        baseline_results['summary'] = self._create_comparison_summary(baseline_results)
        
        logger.info("Baseline comparisons completed")
        return baseline_results
    
    def _run_pca_baseline(
        self, 
        electrode_data: np.ndarray, 
        n_components: int, 
        cv_splits: int
    ) -> Dict[str, Any]:
        """Run PCA baseline comparison."""
        try:
            # Fit PCA
            pca = PCA(n_components=n_components, random_state=42)
            components = pca.fit_transform(electrode_data.T).T  # [n_components × time]
            loadings = pca.components_.T  # [n_electrodes × n_components]
            
            # Compute reconstruction
            reconstruction = loadings @ components
            
            # Metrics
            reconstruction_error = np.mean((electrode_data - reconstruction) ** 2)
            variance_explained = np.sum(pca.explained_variance_ratio_)
            
            # Cross-validation
            cv_errors = self._cross_validate_method(
                electrode_data, lambda X: self._fit_pca(X, n_components), cv_splits
            )
            
            return {
                'method_name': 'PCA',
                'components': components,
                'loadings': loadings,
                'reconstruction': reconstruction,
                'reconstruction_error': reconstruction_error,
                'variance_explained': variance_explained,
                'explained_variance_ratio': pca.explained_variance_ratio_,
                'cv_errors': cv_errors,
                'mean_cv_error': np.mean(cv_errors),
                'std_cv_error': np.std(cv_errors),
                'success': True
            }
            
        except Exception as e:
            logger.warning(f"PCA baseline failed: {str(e)}")
            return {'method_name': 'PCA', 'success': False, 'error': str(e)}
    
    def _run_ica_baseline(
        self, 
        electrode_data: np.ndarray, 
        n_components: int, 
        cv_splits: int
    ) -> Dict[str, Any]:
        """Run ICA baseline comparison."""
        try:
            # Fit ICA
            ica = FastICA(n_components=n_components, random_state=42, max_iter=500)
            components = ica.fit_transform(electrode_data.T).T  # [n_components × time]
            mixing_matrix = ica.mixing_.T  # [n_electrodes × n_components]
            
            # Compute reconstruction
            reconstruction = mixing_matrix @ components
            
            # Metrics
            reconstruction_error = np.mean((electrode_data - reconstruction) ** 2)
            
            # Estimate variance explained (approximate)
            total_var = np.var(electrode_data)
            residual_var = np.var(electrode_data - reconstruction)
            variance_explained = 1 - residual_var / total_var
            
            # Cross-validation
            cv_errors = self._cross_validate_method(
                electrode_data, lambda X: self._fit_ica(X, n_components), cv_splits
            )
            
            return {
                'method_name': 'ICA',
                'components': components,
                'mixing_matrix': mixing_matrix,
                'reconstruction': reconstruction,
                'reconstruction_error': reconstruction_error,
                'variance_explained': variance_explained,
                'cv_errors': cv_errors,
                'mean_cv_error': np.mean(cv_errors),
                'std_cv_error': np.std(cv_errors),
                'success': True
            }
            
        except Exception as e:
            logger.warning(f"ICA baseline failed: {str(e)}")
            return {'method_name': 'ICA', 'success': False, 'error': str(e)}
    
    def _run_kmeans_baseline(
        self, 
        electrode_data: np.ndarray, 
        n_components: int, 
        cv_splits: int
    ) -> Dict[str, Any]:
        """Run K-Means clustering baseline."""
        try:
            # Transpose for clustering (cluster time points)
            data_for_clustering = electrode_data.T  # [time × electrodes]
            
            # Fit K-Means
            kmeans = KMeans(n_clusters=n_components, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(data_for_clustering)
            centroids = kmeans.cluster_centers_.T  # [n_electrodes × n_components]
            
            # Create component activations
            components = np.zeros((n_components, electrode_data.shape[1]))
            for t in range(electrode_data.shape[1]):
                cluster = cluster_labels[t]
                components[cluster, t] = 1.0
            
            # Compute reconstruction
            reconstruction = centroids @ components
            
            # Metrics
            reconstruction_error = np.mean((electrode_data - reconstruction) ** 2)
            
            # Silhouette score as quality measure
            if len(np.unique(cluster_labels)) > 1:
                silhouette = silhouette_score(data_for_clustering, cluster_labels)
            else:
                silhouette = -1
            
            # Cross-validation
            cv_errors = self._cross_validate_method(
                electrode_data, lambda X: self._fit_kmeans(X, n_components), cv_splits
            )
            
            return {
                'method_name': 'K-Means',
                'components': components,
                'centroids': centroids,
                'cluster_labels': cluster_labels,
                'reconstruction': reconstruction,
                'reconstruction_error': reconstruction_error,
                'silhouette_score': silhouette,
                'cv_errors': cv_errors,
                'mean_cv_error': np.mean(cv_errors),
                'std_cv_error': np.std(cv_errors),
                'success': True
            }
            
        except Exception as e:
            logger.warning(f"K-Means baseline failed: {str(e)}")
            return {'method_name': 'K-Means', 'success': False, 'error': str(e)}
    
    def _run_random_baseline(
        self, 
        electrode_data: np.ndarray, 
        n_components: int, 
        cv_splits: int
    ) -> Dict[str, Any]:
        """Run random projection baseline."""
        try:
            n_electrodes, n_timepoints = electrode_data.shape
            
            # Random projection matrix
            np.random.seed(42)
            projection_matrix = np.random.randn(n_electrodes, n_components)
            projection_matrix = projection_matrix / np.linalg.norm(projection_matrix, axis=0, keepdims=True)
            
            # Project data
            components = projection_matrix.T @ electrode_data  # [n_components × time]
            
            # Pseudo-inverse for reconstruction
            reconstruction = projection_matrix @ components
            
            # Metrics
            reconstruction_error = np.mean((electrode_data - reconstruction) ** 2)
            
            # Approximate variance explained
            total_var = np.var(electrode_data)
            residual_var = np.var(electrode_data - reconstruction)
            variance_explained = 1 - residual_var / total_var
            
            # Cross-validation (use different random seeds)
            cv_errors = []
            for cv in range(cv_splits):
                np.random.seed(42 + cv)
                cv_proj = np.random.randn(n_electrodes, n_components)
                cv_proj = cv_proj / np.linalg.norm(cv_proj, axis=0, keepdims=True)
                cv_comp = cv_proj.T @ electrode_data
                cv_recon = cv_proj @ cv_comp
                cv_error = np.mean((electrode_data - cv_recon) ** 2)
                cv_errors.append(cv_error)
            
            return {
                'method_name': 'Random Projection',
                'components': components,
                'projection_matrix': projection_matrix,
                'reconstruction': reconstruction,
                'reconstruction_error': reconstruction_error,
                'variance_explained': variance_explained,
                'cv_errors': cv_errors,
                'mean_cv_error': np.mean(cv_errors),
                'std_cv_error': np.std(cv_errors),
                'success': True
            }
            
        except Exception as e:
            logger.warning(f"Random projection baseline failed: {str(e)}")
            return {'method_name': 'Random Projection', 'success': False, 'error': str(e)}
    
    def _cross_validate_method(
        self, 
        data: np.ndarray, 
        fit_func, 
        cv_splits: int
    ) -> List[float]:
        """Generic cross-validation for baseline methods."""
        n_electrodes, n_timepoints = data.shape
        test_size = n_timepoints // cv_splits
        cv_errors = []
        
        for cv in range(cv_splits):
            # Create train/test split
            start_test = cv * test_size
            end_test = min((cv + 1) * test_size, n_timepoints)
            
            train_mask = np.ones(n_timepoints, dtype=bool)
            train_mask[start_test:end_test] = False
            
            train_data = data[:, train_mask]
            test_data = data[:, start_test:end_test]
            
            try:
                # Fit on training data and reconstruct test data
                reconstruction = fit_func(train_data)
                if reconstruction is not None and reconstruction.shape == test_data.shape:
                    cv_error = np.mean((test_data - reconstruction[:, :test_data.shape[1]]) ** 2)
                else:
                    cv_error = np.inf
            except:
                cv_error = np.inf
            
            cv_errors.append(cv_error)
        
        return cv_errors
    
    def _fit_pca(self, data: np.ndarray, n_components: int) -> np.ndarray:
        """Fit PCA and return reconstruction."""
        try:
            pca = PCA(n_components=n_components, random_state=42)
            components = pca.fit_transform(data.T).T
            reconstruction = pca.components_.T @ components
            return reconstruction
        except:
            return None
    
    def _fit_ica(self, data: np.ndarray, n_components: int) -> np.ndarray:
        """Fit ICA and return reconstruction."""
        try:
            ica = FastICA(n_components=n_components, random_state=42, max_iter=200)
            components = ica.fit_transform(data.T).T
            reconstruction = ica.mixing_.T @ components
            return reconstruction
        except:
            return None
    
    def _fit_kmeans(self, data: np.ndarray, n_components: int) -> np.ndarray:
        """Fit K-Means and return reconstruction."""
        try:
            kmeans = KMeans(n_clusters=n_components, random_state=42, n_init=5)
            labels = kmeans.fit_predict(data.T)
            centroids = kmeans.cluster_centers_.T
            
            # Create components
            components = np.zeros((n_components, data.shape[1]))
            for t in range(data.shape[1]):
                components[labels[t], t] = 1.0
            
            reconstruction = centroids @ components
            return reconstruction
        except:
            return None
    
    def _create_comparison_summary(self, baseline_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create summary comparison of all methods."""
        methods = baseline_results['methods']
        
        summary = {
            'method_names': [],
            'reconstruction_errors': [],
            'variance_explained': [],
            'cv_errors_mean': [],
            'cv_errors_std': [],
            'success_rates': []
        }
        
        for method_name, results in methods.items():
            if results.get('success', False):
                summary['method_names'].append(results['method_name'])
                summary['reconstruction_errors'].append(results['reconstruction_error'])
                summary['variance_explained'].append(results.get('variance_explained', np.nan))
                summary['cv_errors_mean'].append(results['mean_cv_error'])
                summary['cv_errors_std'].append(results['std_cv_error'])
                summary['success_rates'].append(1.0)
            else:
                summary['method_names'].append(results['method_name'])
                summary['reconstruction_errors'].append(np.inf)
                summary['variance_explained'].append(np.nan)
                summary['cv_errors_mean'].append(np.inf)
                summary['cv_errors_std'].append(np.inf)
                summary['success_rates'].append(0.0)
        
        # Find best methods
        if summary['reconstruction_errors']:
            best_recon_idx = np.nanargmin(summary['reconstruction_errors'])
            summary['best_reconstruction_method'] = summary['method_names'][best_recon_idx]
            
            best_var_idx = np.nanargmax(summary['variance_explained'])
            summary['best_variance_method'] = summary['method_names'][best_var_idx]
            
            best_cv_idx = np.nanargmin(summary['cv_errors_mean'])
            summary['best_cv_method'] = summary['method_names'][best_cv_idx]
        
        return summary