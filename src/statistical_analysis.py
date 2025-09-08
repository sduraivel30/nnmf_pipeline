"""
Statistical Analysis Module - Permutation Testing Framework
Computes cross-run correlations, electrode pair correlations, and significance testing.
"""

import numpy as np
from scipy.stats import pearsonr
from scipy.linalg import tril_indices
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
from joblib import Parallel, delayed
import logging

logger = logging.getLogger(__name__)


class StatisticalAnalyzer:
    """Statistical analysis with shuffle tests and visualizations."""
    
    def __init__(self, n_permutations: int = 1000, n_jobs: int = -1):
        self.n_permutations = n_permutations
        self.n_jobs = n_jobs
    
    def compute_stats(
        self,
        run1: np.ndarray,
        run2: np.ndarray, 
        electrode_data: np.ndarray,
        story_halves: Dict[str, Any],
        subject_ids: List[str]
    ) -> Dict[str, Any]:
        """
        Compute statistical measures with permutation test thresholds.
        
        Args:
            run1, run2: [n_electrodes × time] split data runs
            electrode_data: [n_electrodes × total_time] full data
            story_halves: dict with story boundary information
            subject_ids: [n_electrodes] list of subject IDs
            
        Returns:
            stat_analysis: dict with computed metrics and figures
        """
        n_electrodes, total_time = electrode_data.shape
        
        # Remove common mode per subject
        run1_clean, run2_clean = self._remove_common_mode_per_subject(
            run1, run2, subject_ids
        )
        
        stat_analysis = {}
        
        # Compute subject boundaries for visualization
        unique_subjects = list(dict.fromkeys(subject_ids))  # preserve order
        subject_boundaries = self._get_subject_boundaries(subject_ids, unique_subjects)
        
        # 1. Cross-run correlation per electrode
        logger.info("Computing cross-run correlations...")
        cross_run_results = self._compute_cross_run_correlations(
            run1_clean, run2_clean
        )
        stat_analysis['cross_run'] = cross_run_results
        
        # 2. Cross-electrode correlations and shuffle test
        logger.info("Computing cross-electrode correlation matrices...")
        cross_electrode_results = self._compute_cross_electrode_correlations(
            run1_clean, run2_clean, subject_boundaries
        )
        stat_analysis['cross_electrode'] = cross_electrode_results
        
        # 3. Significant electrodes from significant pairs
        logger.info("Computing significant electrodes from pairs...")
        significant_electrode_results = self._compute_significant_electrodes_from_pairs(
            cross_electrode_results, n_electrodes
        )
        stat_analysis['cross_electrode'].update(significant_electrode_results)
        
        # Generate visualizations
        self._create_visualizations(stat_analysis, subject_boundaries)
        
        # Summary
        self._print_summary(stat_analysis, n_electrodes)
        
        return stat_analysis
    
    def _remove_common_mode_per_subject(
        self, 
        run1: np.ndarray, 
        run2: np.ndarray, 
        subject_ids: List[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Remove common mode signal within each subject."""
        run1_clean = run1.copy()
        run2_clean = run2.copy()
        
        unique_subjects = list(set(subject_ids))
        
        for subject in unique_subjects:
            subject_mask = np.array([sid == subject for sid in subject_ids])
            subject_indices = np.where(subject_mask)[0]
            
            if len(subject_indices) > 1:
                # Compute and subtract common mode
                mean_run1 = np.mean(run1[subject_indices], axis=0)
                mean_run2 = np.mean(run2[subject_indices], axis=0)
                
                run1_clean[subject_indices] -= mean_run1[np.newaxis, :]
                run2_clean[subject_indices] -= mean_run2[np.newaxis, :]
        
        return run1_clean, run2_clean
    
    def _get_subject_boundaries(
        self, 
        subject_ids: List[str], 
        unique_subjects: List[str]
    ) -> np.ndarray:
        """Get electrode indices where subjects change."""
        boundaries = []
        for i, subject in enumerate(unique_subjects[:-1]):
            last_idx = len(subject_ids) - 1 - subject_ids[::-1].index(subject)
            boundaries.append(last_idx)
        return np.array(boundaries)
    
    def _compute_cross_run_correlations(
        self, 
        run1: np.ndarray, 
        run2: np.ndarray
    ) -> Dict[str, Any]:
        """Compute cross-run correlations with shuffle threshold."""
        n_electrodes = run1.shape[0]
        cross_run_corrs = np.full(n_electrodes, np.nan)
        
        # Compute correlations
        for e in range(n_electrodes):
            if np.std(run1[e]) > 0 and np.std(run2[e]) > 0:
                cross_run_corrs[e] = pearsonr(run1[e], run2[e])[0]
        
        valid_corrs = cross_run_corrs[~np.isnan(cross_run_corrs)]
        
        # Shuffle null distribution
        null_corrs = self._compute_shuffle_null_correlations(run1, run2)
        threshold = np.percentile(null_corrs, 95)
        
        # Significant electrodes
        significant_electrodes = (~np.isnan(cross_run_corrs)) & (cross_run_corrs > threshold)
        
        return {
            'correlations': cross_run_corrs,
            'valid_correlations': valid_corrs,
            'mean_correlation': np.mean(valid_corrs),
            'median_correlation': np.median(valid_corrs),
            'shuffle_threshold': threshold,
            'significant_electrodes': significant_electrodes,
            'n_significant': np.sum(significant_electrodes)
        }
    
    def _compute_shuffle_null_correlations(
        self, 
        run1: np.ndarray, 
        run2: np.ndarray
    ) -> np.ndarray:
        """Compute null distribution for cross-run correlations."""
        def shuffle_correlation():
            # Use first electrode as representative
            e = 0
            if np.std(run1[e]) > 0 and np.std(run2[e]) > 0:
                idx = np.random.permutation(run2.shape[1])
                return pearsonr(run1[e], run2[e, idx])[0]
            return np.nan
        
        null_corrs = Parallel(n_jobs=self.n_jobs)(
            delayed(shuffle_correlation)() for _ in range(self.n_permutations)
        )
        
        return np.array([corr for corr in null_corrs if not np.isnan(corr)])
    
    def _compute_cross_electrode_correlations(
        self, 
        run1: np.ndarray, 
        run2: np.ndarray,
        subject_boundaries: np.ndarray
    ) -> Dict[str, Any]:
        """Compute cross-electrode correlations and shuffle test."""
        # Compute correlation matrices
        corr_run1 = np.corrcoef(run1)
        corr_run2 = np.corrcoef(run2)
        
        # Extract lower triangle (excluding diagonal)
        n_electrodes = run1.shape[0]
        tril_indices_lower = tril_indices(n_electrodes, k=-1)
        
        # Mask matrices for visualization
        corr_run1_masked = corr_run1.copy()
        corr_run2_masked = corr_run2.copy()
        mask = np.triu(np.ones_like(corr_run1_masked, dtype=bool))
        corr_run1_masked[mask] = np.nan
        corr_run2_masked[mask] = np.nan
        
        # Extract valid pairs
        v1 = corr_run1[tril_indices_lower]
        v2 = corr_run2[tril_indices_lower]
        valid_pairs = ~(np.isnan(v1) | np.isnan(v2))
        v1_clean = v1[valid_pairs]
        v2_clean = v2[valid_pairs]
        
        # Second-level correlation
        second_level_corr = pearsonr(v1_clean, v2_clean)[0] if len(v1_clean) > 10 else np.nan
        
        # Shuffle test on electrode pairs
        obs_prod = v1_clean * v2_clean
        null_prod_matrix = self._compute_pair_shuffle_null(v1_clean, v2_clean)
        
        # Determine significant pairs
        threshold_prod = np.percentile(null_prod_matrix, 95, axis=1)
        significant_pairs = obs_prod > threshold_prod
        
        return {
            'corr_matrix_run1': corr_run1_masked,
            'corr_matrix_run2': corr_run2_masked,
            'second_level_correlation': second_level_corr,
            'n_valid_pairs': np.sum(valid_pairs),
            'total_pairs': len(v1),
            'mean_corr_run1': np.nanmean(corr_run1_masked),
            'mean_corr_run2': np.nanmean(corr_run2_masked),
            'std_corr_run1': np.nanstd(corr_run1_masked),
            'std_corr_run2': np.nanstd(corr_run2_masked),
            'significant_pairs': significant_pairs,
            'n_significant_pairs': np.sum(significant_pairs),
            'pair_indices': (tril_indices_lower[0][valid_pairs], tril_indices_lower[1][valid_pairs]),
            'shuffle_threshold': threshold_prod
        }
    
    def _compute_pair_shuffle_null(
        self, 
        v1: np.ndarray, 
        v2: np.ndarray
    ) -> np.ndarray:
        """Compute null distribution for electrode pair products."""
        n_pairs = len(v1)
        null_matrix = np.zeros((n_pairs, self.n_permutations))
        
        def compute_permutation(perm_idx):
            perm_indices = np.random.permutation(len(v2))
            return v1 * v2[perm_indices]
        
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(compute_permutation)(p) for p in range(self.n_permutations)
        )
        
        for p, result in enumerate(results):
            null_matrix[:, p] = result
            
        return null_matrix
    
    def _compute_significant_electrodes_from_pairs(
        self, 
        cross_electrode_results: Dict[str, Any],
        n_electrodes: int
    ) -> Dict[str, Any]:
        """Compute significant electrodes from significant pairs using leave-one-out."""
        significant_pairs = cross_electrode_results['significant_pairs']
        pair_indices = cross_electrode_results['pair_indices']
        
        # Count occurrences of each electrode in significant pairs
        electrode_counts = np.zeros(n_electrodes)
        sig_row_idx = pair_indices[0][significant_pairs]
        sig_col_idx = pair_indices[1][significant_pairs]
        
        for i in range(len(sig_row_idx)):
            electrode_counts[sig_row_idx[i]] += 1
            electrode_counts[sig_col_idx[i]] += 1
        
        # Leave-one-out permutation test for electrode counts
        null_counts = self._compute_electrode_count_null(
            pair_indices, n_electrodes
        )
        
        # Per-electrode thresholds
        null_thresholds = np.percentile(null_counts, 95, axis=1)
        significant_electrodes_perm = electrode_counts > null_thresholds
        
        return {
            'electrode_pair_counts': electrode_counts,
            'electrode_pair_scores': electrode_counts / np.max(electrode_counts) if np.max(electrode_counts) > 0 else electrode_counts,
            'null_thresholds': null_thresholds,
            'significant_electrodes_perm_test': significant_electrodes_perm,
            'n_significant_electrodes_perm': np.sum(significant_electrodes_perm)
        }
    
    def _compute_electrode_count_null(
        self, 
        pair_indices: Tuple[np.ndarray, np.ndarray],
        n_electrodes: int
    ) -> np.ndarray:
        """Compute null distribution for electrode pair counts."""
        row_idx, col_idx = pair_indices
        n_pairs = len(row_idx)
        
        def compute_null_counts():
            # Shuffle pair labels
            shuffled_pairs = np.random.permutation(n_pairs)[:np.sum(np.random.rand(n_pairs) < 0.5)]  # Random subset
            
            temp_counts = np.zeros(n_electrodes)
            for i in shuffled_pairs:
                temp_counts[row_idx[i]] += 1
                temp_counts[col_idx[i]] += 1
            
            return temp_counts
        
        null_results = Parallel(n_jobs=self.n_jobs)(
            delayed(compute_null_counts)() for _ in range(self.n_permutations)
        )
        
        return np.array(null_results).T  # [n_electrodes × n_permutations]
    
    def _create_visualizations(
        self, 
        stat_analysis: Dict[str, Any], 
        subject_boundaries: np.ndarray
    ):
        """Create visualization plots."""
        # Cross-run correlation histogram
        plt.figure(figsize=(10, 6))
        
        plt.subplot(2, 3, 1)
        valid_corrs = stat_analysis['cross_run']['valid_correlations']
        threshold = stat_analysis['cross_run']['shuffle_threshold']
        
        plt.hist(valid_corrs, bins=20, alpha=0.7, color='blue', edgecolor='black')
        plt.axvline(threshold, color='red', linestyle='--', linewidth=2)
        plt.xlabel('Cross-run Correlation')
        plt.ylabel('Count')
        plt.title('Cross-run Correlations & Shuffle Threshold')
        plt.grid(True)
        
        # Correlation matrices
        plt.subplot(2, 3, 2)
        corr_matrix = stat_analysis['cross_electrode']['corr_matrix_run1']
        im1 = plt.imshow(corr_matrix, cmap='jet', vmin=-1, vmax=1)
        plt.title('Run 1 Correlations')
        plt.colorbar(im1)
        
        # Add subject boundaries
        for boundary in subject_boundaries:
            plt.axhline(boundary + 0.5, color='black', linestyle='--', linewidth=1)
            plt.axvline(boundary + 0.5, color='black', linestyle='--', linewidth=1)
        
        plt.subplot(2, 3, 3)
        corr_matrix = stat_analysis['cross_electrode']['corr_matrix_run2']
        im2 = plt.imshow(corr_matrix, cmap='jet', vmin=-1, vmax=1)
        plt.title('Run 2 Correlations')
        plt.colorbar(im2)
        
        # Add subject boundaries
        for boundary in subject_boundaries:
            plt.axhline(boundary + 0.5, color='black', linestyle='--', linewidth=1)
            plt.axvline(boundary + 0.5, color='black', linestyle='--', linewidth=1)
        
        # Electrode pair counts vs thresholds
        plt.subplot(2, 3, 4)
        counts = stat_analysis['cross_electrode']['electrode_pair_counts']
        thresholds = stat_analysis['cross_electrode']['null_thresholds']
        
        sorted_indices = np.argsort(counts)[::-1]
        sorted_counts = counts[sorted_indices]
        sorted_thresholds = thresholds[sorted_indices]
        
        plt.plot(sorted_counts, 'b-', linewidth=2, label='Observed Counts')
        plt.plot(sorted_thresholds, 'r--', linewidth=2, label='95% Perm Threshold')
        plt.xlabel('Electrodes (sorted)')
        plt.ylabel('Number of Significant Pairs')
        plt.title('Electrode Significant Pair Counts vs Null')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def _print_summary(self, stat_analysis: Dict[str, Any], n_electrodes: int):
        """Print analysis summary."""
        print(\"\\nStatistical analysis complete.\")\n        \n        # Cross-run stats\n        cr_stats = stat_analysis['cross_run']\n        print(f\"  Cross-run significant electrodes: {cr_stats['n_significant']}/{n_electrodes} ({100*cr_stats['n_significant']/n_electrodes:.1f}%)\")\n        \n        # Cross-electrode stats\n        ce_stats = stat_analysis['cross_electrode']\n        print(f\"  2nd-level electrode-pair correlation: {ce_stats['second_level_correlation']:.3f}\")\n        print(f\"  Significant electrode pairs: {ce_stats['n_significant_pairs']}/{ce_stats['n_valid_pairs']}\")\n        print(f\"  Permutation-test significant electrodes: {ce_stats['n_significant_electrodes_perm']}/{n_electrodes} ({100*ce_stats['n_significant_electrodes_perm']/n_electrodes:.1f}%)\")