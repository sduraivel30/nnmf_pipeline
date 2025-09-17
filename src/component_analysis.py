"""
Component Analysis Module - NNMF Component Characterization
Analyzes and summarizes NNMF components with detailed statistics and visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List
from scipy.stats import pearsonr
import logging

logger = logging.getLogger(__name__)


class ComponentAnalyzer:
    """Analyzes and characterizes NNMF components."""
    
    def __init__(self):
        pass
    
    def summarize_components(
        self,
        final_model: Dict[str, Any],
        electrode_data: np.ndarray,
        story_halves: Dict[str, Any],
        subject_ids: List[str]
    ) -> Dict[str, Any]:
        """
        Analyze and summarize NNMF components.
        
        Args:
            final_model: Dictionary containing fitted NNMF model
            electrode_data: [n_electrodes × total_time] original data
            story_halves: Story boundary information
            subject_ids: [n_electrodes] subject ID per electrode
            
        Returns:
            component_summary: Detailed component analysis
        """
        W = final_model['W']
        H = final_model['H']
        n_components = final_model['n_components']
        
        logger.info(f"Analyzing {n_components} NNMF components...")
        
        component_summary = {
            'n_components': n_components,
            'analysis_timestamp': np.datetime64('now'),
            'data_shape': electrode_data.shape
        }
        
        # Analyze each component
        for comp_idx in range(n_components):
            comp_key = f'component_{comp_idx + 1}'
            
            comp_analysis = self._analyze_single_component(
                comp_idx, W, H, electrode_data, subject_ids
            )
            
            component_summary[comp_key] = comp_analysis
        
        # Global component interactions
        component_summary.update(
            self._analyze_component_interactions(W, H)
        )
        
        # Story-specific analysis
        if story_halves and story_halves.get('n_stories', 0) > 1:
            component_summary.update(
                self._analyze_story_specificity(H, story_halves)
            )
        
        # Subject-specific analysis
        component_summary.update(
            self._analyze_subject_specificity(W, subject_ids)
        )
        
        logger.info("Component analysis completed")
        
        return component_summary
    
    def _analyze_single_component(
        self,
        comp_idx: int,
        W: np.ndarray,
        H: np.ndarray,
        electrode_data: np.ndarray,
        subject_ids: List[str]
    ) -> Dict[str, Any]:
        """Analyze a single component in detail."""
        
        # Extract component vectors
        spatial_pattern = W[:, comp_idx]
        temporal_pattern = H[comp_idx, :]
        
        # Basic statistics
        spatial_stats = {
            'mean': np.mean(spatial_pattern),
            'std': np.std(spatial_pattern),
            'max': np.max(spatial_pattern),
            'min': np.min(spatial_pattern),
            'sparsity': self._compute_sparsity_single(spatial_pattern)
        }
        
        temporal_stats = {
            'mean': np.mean(temporal_pattern),
            'std': np.std(temporal_pattern),
            'max': np.max(temporal_pattern),
            'min': np.min(temporal_pattern),
            'sparsity': self._compute_sparsity_single(temporal_pattern)
        }
        
        # Spatial analysis
        top_electrodes = self._find_top_electrodes(spatial_pattern, n_top=10)
        spatial_concentration = self._compute_spatial_concentration(spatial_pattern)
        
        # Temporal analysis
        temporal_smoothness = self._compute_temporal_smoothness(temporal_pattern)
        temporal_peaks = self._find_temporal_peaks(temporal_pattern)
        
        # Component contribution to reconstruction
        component_reconstruction = np.outer(spatial_pattern, temporal_pattern)
        reconstruction_contribution = np.sum(component_reconstruction ** 2) / np.sum(electrode_data ** 2)
        
        # Subject distribution
        subject_distribution = self._analyze_subject_distribution(spatial_pattern, subject_ids)
        
        return {
            'component_index': comp_idx,
            'spatial_stats': spatial_stats,
            'temporal_stats': temporal_stats,
            'top_electrodes': top_electrodes,
            'spatial_concentration': spatial_concentration,
            'temporal_smoothness': temporal_smoothness,
            'temporal_peaks': temporal_peaks,
            'reconstruction_contribution': reconstruction_contribution,
            'subject_distribution': subject_distribution
        }
    
    def _analyze_component_interactions(self, W: np.ndarray, H: np.ndarray) -> Dict[str, Any]:
        """Analyze interactions between components."""
        n_components = W.shape[1]
        
        # Spatial correlations between components
        spatial_correlations = np.corrcoef(W.T)
        
        # Temporal correlations between components
        temporal_correlations = np.corrcoef(H)
        
        # Component orthogonality measures
        spatial_orthogonality = self._compute_orthogonality(W)
        temporal_orthogonality = self._compute_orthogonality(H.T)
        
        return {
            'component_interactions': {
                'spatial_correlations': spatial_correlations,
                'temporal_correlations': temporal_correlations,
                'mean_spatial_correlation': np.mean(np.abs(spatial_correlations[np.triu_indices(n_components, k=1)])),
                'mean_temporal_correlation': np.mean(np.abs(temporal_correlations[np.triu_indices(n_components, k=1)])),
                'spatial_orthogonality': spatial_orthogonality,
                'temporal_orthogonality': temporal_orthogonality
            }
        }
    
    def _analyze_story_specificity(self, H: np.ndarray, story_halves: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze component activity across different stories."""
        n_stories = story_halves['n_stories']
        boundaries = story_halves['boundaries']
        n_components = H.shape[0]
        
        story_activities = np.zeros((n_components, n_stories))
        
        for story_idx in range(n_stories):
            start_idx, _, end_idx, _ = boundaries[story_idx]
            story_data = H[:, start_idx:end_idx]
            story_activities[:, story_idx] = np.mean(story_data, axis=1)
        
        # Compute story specificity index for each component
        story_specificity = np.zeros(n_components)
        for comp_idx in range(n_components):
            activity = story_activities[comp_idx, :]
            # Normalized entropy as specificity measure
            p = activity / np.sum(activity) if np.sum(activity) > 0 else np.ones(n_stories) / n_stories
            entropy = -np.sum(p * np.log(p + 1e-10))
            max_entropy = np.log(n_stories)
            story_specificity[comp_idx] = 1 - entropy / max_entropy
        
        return {
            'story_analysis': {
                'story_activities': story_activities,
                'story_specificity': story_specificity,
                'most_story_specific_component': np.argmax(story_specificity),
                'mean_story_specificity': np.mean(story_specificity)
            }
        }
    
    def _analyze_subject_specificity(self, W: np.ndarray, subject_ids: List[str]) -> Dict[str, Any]:
        """Analyze component spatial patterns across subjects."""
        unique_subjects = list(set(subject_ids))
        n_subjects = len(unique_subjects)
        n_components = W.shape[1]
        
        subject_loadings = np.zeros((n_components, n_subjects))
        
        for subj_idx, subject in enumerate(unique_subjects):
            subject_electrodes = [i for i, sid in enumerate(subject_ids) if sid == subject]
            if subject_electrodes:
                subject_loadings[:, subj_idx] = np.mean(W[subject_electrodes, :], axis=0)
        
        # Compute subject specificity index
        subject_specificity = np.zeros(n_components)
        for comp_idx in range(n_components):
            loading = subject_loadings[comp_idx, :]
            # Normalized entropy as specificity measure
            p = loading / np.sum(loading) if np.sum(loading) > 0 else np.ones(n_subjects) / n_subjects
            entropy = -np.sum(p * np.log(p + 1e-10))
            max_entropy = np.log(n_subjects)
            subject_specificity[comp_idx] = 1 - entropy / max_entropy
        
        return {
            'subject_analysis': {
                'unique_subjects': unique_subjects,
                'subject_loadings': subject_loadings,
                'subject_specificity': subject_specificity,
                'most_subject_specific_component': np.argmax(subject_specificity),
                'mean_subject_specificity': np.mean(subject_specificity)
            }
        }
    
    def _compute_sparsity_single(self, vector: np.ndarray) -> float:
        """Compute sparsity measure for a single vector."""
        n = len(vector)
        if n <= 1:
            return 0.0
        
        l1_norm = np.sum(np.abs(vector))
        l2_norm = np.sqrt(np.sum(vector ** 2))
        
        if l2_norm > 0:
            sparsity = (np.sqrt(n) - l1_norm/l2_norm) / (np.sqrt(n) - 1)
        else:
            sparsity = 0.0
        
        return np.clip(sparsity, 0, 1)
    
    def _find_top_electrodes(self, spatial_pattern: np.ndarray, n_top: int = 10) -> Dict[str, Any]:
        """Find electrodes with highest loadings for this component."""
        sorted_indices = np.argsort(spatial_pattern)[::-1]
        top_indices = sorted_indices[:n_top]
        top_values = spatial_pattern[top_indices]
        
        return {
            'indices': top_indices.tolist(),
            'values': top_values.tolist(),
            'fraction_of_max': (top_values / np.max(spatial_pattern)).tolist()
        }
    
    def _compute_spatial_concentration(self, spatial_pattern: np.ndarray) -> float:
        """Compute how concentrated the spatial pattern is."""
        # Gini coefficient as concentration measure
        sorted_pattern = np.sort(np.abs(spatial_pattern))
        n = len(sorted_pattern)
        cumsum = np.cumsum(sorted_pattern)
        
        if np.sum(sorted_pattern) == 0:
            return 0.0
        
        gini = (n + 1 - 2 * np.sum((n + 1 - np.arange(1, n + 1)) * sorted_pattern) / np.sum(sorted_pattern)) / n
        return gini
    
    def _compute_temporal_smoothness(self, temporal_pattern: np.ndarray) -> float:
        """Compute temporal smoothness of the component."""
        if len(temporal_pattern) <= 1:
            return 0.0
        
        # Compute first derivative
        diff = np.diff(temporal_pattern)
        # Smoothness inversely related to variance of derivative
        smoothness = 1 / (1 + np.var(diff))
        return smoothness
    
    def _find_temporal_peaks(self, temporal_pattern: np.ndarray, min_distance: int = 50) -> Dict[str, Any]:
        """Find significant peaks in temporal pattern."""
        # Simple peak finding - can be enhanced with scipy.signal.find_peaks
        if len(temporal_pattern) < 3:
            return {'peak_indices': [], 'peak_values': [], 'n_peaks': 0}
        
        # Find local maxima
        peaks = []
        for i in range(1, len(temporal_pattern) - 1):
            if (temporal_pattern[i] > temporal_pattern[i-1] and 
                temporal_pattern[i] > temporal_pattern[i+1]):
                peaks.append(i)
        
        # Filter by minimum distance
        if min_distance > 0 and len(peaks) > 1:
            filtered_peaks = [peaks[0]]
            for peak in peaks[1:]:
                if peak - filtered_peaks[-1] >= min_distance:
                    filtered_peaks.append(peak)
            peaks = filtered_peaks
        
        peak_values = [temporal_pattern[p] for p in peaks]
        
        return {
            'peak_indices': peaks,
            'peak_values': peak_values,
            'n_peaks': len(peaks)
        }
    
    def _analyze_subject_distribution(self, spatial_pattern: np.ndarray, subject_ids: List[str]) -> Dict[str, Any]:
        """Analyze how the component is distributed across subjects."""
        unique_subjects = list(set(subject_ids))
        subject_means = {}
        subject_maxes = {}
        subject_counts = {}
        
        for subject in unique_subjects:
            subject_electrodes = [i for i, sid in enumerate(subject_ids) if sid == subject]
            if subject_electrodes:
                subject_pattern = spatial_pattern[subject_electrodes]
                subject_means[subject] = np.mean(subject_pattern)
                subject_maxes[subject] = np.max(subject_pattern)
                subject_counts[subject] = len(subject_electrodes)
        
        return {
            'subject_means': subject_means,
            'subject_maxes': subject_maxes,
            'subject_counts': subject_counts,
            'dominant_subject': max(subject_means.keys(), key=lambda k: subject_means[k]) if subject_means else None
        }
    
    def _compute_orthogonality(self, matrix: np.ndarray) -> float:
        """Compute average orthogonality of matrix columns."""
        n_cols = matrix.shape[1]
        if n_cols <= 1:
            return 1.0
        
        # Normalize columns
        normalized = matrix / (np.linalg.norm(matrix, axis=0, keepdims=True) + 1e-10)
        
        # Compute gram matrix
        gram = normalized.T @ normalized
        
        # Average off-diagonal absolute values (lower = more orthogonal)
        off_diag = np.abs(gram[np.triu_indices(n_cols, k=1)])
        orthogonality = 1 - np.mean(off_diag)
        
        return orthogonality