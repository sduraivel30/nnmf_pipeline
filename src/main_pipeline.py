"""
Main Pipeline Module - Complete NNMF Analysis Framework
Orchestrates the complete analysis pipeline from data input to final results.
"""

import numpy as np
from typing import Dict, Any, List, Optional
import logging
from pathlib import Path

# Import pipeline modules
from .data_splitting import split_stories_into_runs, remove_common_mode_per_subject
from .statistical_analysis import StatisticalAnalyzer
from .nnmf_solver import ConstrainedNNMFSolver, define_constraint_config
from .cross_validation import CrossValidator
from .model_fitting import ModelFitter
from .component_analysis import ComponentAnalyzer
from .baseline_comparison import BaselineComparator

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NNMFPipeline:
    """Complete NNMF analysis pipeline based on Norman-Haignere et al. 2022."""

    def __init__(self, 
                 n_permutations: int = 1000,
                 n_jobs: int = -1,
                 random_state: Optional[int] = None):
        """
        Initialize NNMF Pipeline.

        Args:
            n_permutations: Number of permutations for statistical tests
            n_jobs: Number of parallel jobs (-1 for all cores)
            random_state: Random seed for reproducibility
        """
        self.n_permutations = n_permutations
        self.n_jobs = n_jobs

        if random_state is not None:
            np.random.seed(random_state)

        # Initialize pipeline components
        self.statistical_analyzer = StatisticalAnalyzer(n_permutations, n_jobs)
        self.cross_validator = CrossValidator(n_jobs)
        self.model_fitter = ModelFitter()
        self.component_analyzer = ComponentAnalyzer()
        self.baseline_comparator = BaselineComparator()

        logger.info(f"NNMF Pipeline initialized with {n_permutations} permutations")

    def run_complete_analysis(
        self,
        electrode_data: np.ndarray,
        story_lengths: np.ndarray,
        subject_ids: List[str],
        max_components: int = 15,
        cv_splits: int = 5,
        init_reps: int = 10,
        test_frac: float = 0.2,
        save_results: bool = True,
        output_dir: Optional[str] = None,
        identify_significant_electrodes: bool = False  # New optional argument
    ) -> Dict[str, Any]:
        """
        Run complete NNMF analysis pipeline.

        Args:
            electrode_data: [n_electrodes × total_time] neural data
            story_lengths: [n_stories] length of each story in samples  
            subject_ids: [n_electrodes] subject ID for each electrode
            max_components: Maximum number of components to test
            cv_splits: Number of cross-validation splits
            init_reps: Number of random initializations
            test_frac: Fraction of data for testing
            save_results: Whether to save results to disk
            output_dir: Directory to save results (default: current directory)
            identify_significant_electrodes: Whether to perform significant electrode identification (optional)

        Returns:
            results: Complete analysis results dictionary
        """
        logger.info("Starting complete NNMF analysis pipeline...")
        logger.info(f"Data shape: {electrode_data.shape}, Stories: {len(story_lengths)}, Subjects: {len(set(subject_ids))}")

        # Validate inputs
        self._validate_inputs(electrode_data, story_lengths, subject_ids)

        results = {}

        # Step 1: Split data into runs
        logger.info("Step 1: Splitting stories into temporal runs...")
        run1, run2, story_halves = split_stories_into_runs(electrode_data, story_lengths)
        results['data_splits'] = {
            'run1': run1,
            'run2': run2,
            'story_halves': story_halves
        }

        # Step 2: Conditional statistical analysis
        if identify_significant_electrodes:
            logger.info("Step 2: Computing statistical reliability measures...")
            stat_analysis = self.statistical_analyzer.compute_stats(
                run1, run2, electrode_data, story_halves, subject_ids
            )
            results['statistics'] = stat_analysis

            # Optional: Identify and select only significant electrodes for NNMF
            significant_mask = np.array(stat_analysis['cross_run'].get('significant_electrodes_mask', []))
            if significant_mask.size == 0:
                logger.warning("No significance mask found in statistics; skipping electrode filtering.")
            else:
                n_sig = np.sum(significant_mask)
                logger.info(f"Selecting {n_sig} significant electrodes out of {electrode_data.shape[0]}")
                electrode_data = electrode_data[significant_mask, :]
                subject_ids = [subject_ids[i] for i, flag in enumerate(significant_mask) if flag]
                run1 = run1[significant_mask, :]
                run2 = run2[significant_mask, :]
                results['data_splits']['run1'] = run1
                results['data_splits']['run2'] = run2

        else:
            logger.info("Skipping statistical analysis and significant electrode identification.")
            # Provide empty/default stats dict to avoid errors downstream
            stat_analysis = {
                'cross_run': {
                    'n_significant': 0,
                    'significant_electrodes_mask': np.array([]),
                    'correlations': []
                },
                'cross_electrode': {
                    'n_significant_electrodes_perm': 0,
                    'second_level_correlation': 0.0
                }
            }
            results['statistics'] = stat_analysis

        # Step 3: Define constraint configuration
        logger.info("Step 3: Configuring NNMF constraints...")
        constraint_config = define_constraint_config(stat_analysis, np.min(story_lengths))
        results['constraint_config'] = constraint_config

        # Step 4: Cross-validation search
        logger.info("Step 4: Running cross-validation hyperparameter search...")
        cv_results = self.cross_validator.run_cross_validation(
            electrode_data, story_lengths, max_components,
            cv_splits, init_reps, test_frac, constraint_config
        )
        results['cross_validation'] = cv_results

        # Step 5: Fit final model on full dataset
        logger.info("Step 5: Fitting final NNMF model...")
        final_model = self.model_fitter.fit_final_model(
            electrode_data,
            cv_results['optimal_components'],
            cv_results['optimal_alpha'],
            init_reps,
            constraint_config
        )
        results['final_model'] = final_model

        # Step 6: Component analysis
        logger.info("Step 6: Analyzing NNMF components...")
        component_analysis = self.component_analyzer.summarize_components(
            final_model, electrode_data, story_halves, subject_ids
        )
        results['components'] = component_analysis

        # Step 7: Baseline comparisons
        logger.info("Step 7: Running baseline method comparisons...")
        baseline_results = self.baseline_comparator.run_baseline_comparisons(
            electrode_data, story_lengths, cv_results['optimal_components'],
            cv_splits, constraint_config
        )
        results['baselines'] = baseline_results

        # Save results if requested
        if save_results:
            output_path = Path(output_dir) if output_dir else Path.cwd()
            self._save_results(results, output_path)

        logger.info("Complete NNMF analysis pipeline finished successfully!")
        self._print_final_summary(results)

        return results


    def run_step_by_step(self, **kwargs) -> Dict[str, Any]:
        """Run pipeline step-by-step for interactive analysis."""
        # This allows users to run individual steps and inspect results
        return self.run_complete_analysis(**kwargs)

    def _validate_inputs(
        self, 
        electrode_data: np.ndarray, 
        story_lengths: np.ndarray, 
        subject_ids: List[str]
    ):
        """Validate input data consistency."""
        n_electrodes, total_time = electrode_data.shape

        # Check dimensions
        assert len(subject_ids) == n_electrodes, f"Subject IDs length {len(subject_ids)} != n_electrodes {n_electrodes}"
        assert np.sum(story_lengths) == total_time, f"Story lengths sum {np.sum(story_lengths)} != total_time {total_time}"

        # Check data quality
        assert not np.any(np.isnan(electrode_data)), "Electrode data contains NaN values"
        assert not np.any(np.isinf(electrode_data)), "Electrode data contains infinite values"
        assert np.all(story_lengths > 0), "All story lengths must be positive"

        logger.info("Input validation passed")

    def _save_results(self, results: Dict[str, Any], output_dir: Path):
        """Save analysis results to disk."""
        output_dir.mkdir(exist_ok=True)

        # Save key results as numpy arrays
        np.save(output_dir / 'final_model_W.npy', results['final_model']['W'])
        np.save(output_dir / 'final_model_H.npy', results['final_model']['H'])

        # Save statistics and parameters as pickled dict
        import pickle
        with open(output_dir / 'analysis_results.pkl', 'wb') as f:
            pickle.dump(results, f)

        logger.info(f"Results saved to {output_dir}")

    def _print_final_summary(self, results: Dict[str, Any]):
        """Print final analysis summary."""
        print("\n" + "="*60)
        print("NNMF ANALYSIS PIPELINE - FINAL SUMMARY")
        print("="*60)

        # Data summary
        final_model = results['final_model']
        stats = results['statistics']
        cv = results['cross_validation']

        print(f"\nData Summary:")
        print(f"  Electrodes: {final_model['W'].shape[0]}")
        print(f"  Time points: {final_model['H'].shape[1]}")
        print(f"  Stories: {results['data_splits']['story_halves']['n_stories']}")

        print(f"\nOptimal Model:")
        print(f"  Components: {final_model['n_components']}")
        print(f"  Alpha: {final_model['alpha']:.4f}")
        print(f"  Variance explained: {final_model['variance_explained']:.3f}")

        print(f"\nElectrode Reliability:")
        n_electrodes = final_model['W'].shape[0]
        n_sig_cr = stats['cross_run']['n_significant']
        n_sig_perm = stats['cross_electrode']['n_significant_electrodes_perm']
        print(f"  Cross-run significant: {n_sig_cr}/{n_electrodes} ({100*n_sig_cr/n_electrodes:.1f}%)")
        print(f"  Permutation significant: {n_sig_perm}/{n_electrodes} ({100*n_sig_perm/n_electrodes:.1f}%)")

        print(f"\nModel Quality:")
        print(f"  2nd-level correlation: {stats['cross_electrode']['second_level_correlation']:.3f}")
        print(f"  Reconstruction error: {final_model['reconstruction_error']:.6f}")

        print("\n" + "="*60)

# Convenience functions for direct usage
def run_nnmf_analysis(
    electrode_data: np.ndarray,
    story_lengths: np.ndarray, 
    subject_ids: List[str],
    **kwargs
) -> Dict[str, Any]:
    """Convenience function to run complete analysis."""
    pipeline = NNMFPipeline()
    return pipeline.run_complete_analysis(
        electrode_data, story_lengths, subject_ids, **kwargs
    )

def load_analysis_results(results_path: str) -> Dict[str, Any]:
    """Load saved analysis results."""
    import pickle
    with open(results_path, 'rb') as f:
        return pickle.load(f)
