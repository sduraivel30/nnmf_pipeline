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
        output_dir: Optional[str] = None
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
            
        Returns:
            results: Complete analysis results dictionary
        """
        logger.info(\"Starting complete NNMF analysis pipeline...\")\n        logger.info(f\"Data shape: {electrode_data.shape}, Stories: {len(story_lengths)}, Subjects: {len(set(subject_ids))}\")\n        \n        # Validate inputs\n        self._validate_inputs(electrode_data, story_lengths, subject_ids)\n        \n        results = {}\n        \n        # Step 1: Split data into runs\n        logger.info(\"Step 1: Splitting stories into temporal runs...\")\n        run1, run2, story_halves = split_stories_into_runs(electrode_data, story_lengths)\n        results['data_splits'] = {\n            'run1': run1,\n            'run2': run2, \n            'story_halves': story_halves\n        }\n        \n        # Step 2: Statistical analysis\n        logger.info(\"Step 2: Computing statistical reliability measures...\")\n        stat_analysis = self.statistical_analyzer.compute_stats(\n            run1, run2, electrode_data, story_halves, subject_ids\n        )\n        results['statistics'] = stat_analysis\n        \n        # Step 3: Define constraint configuration\n        logger.info(\"Step 3: Configuring NNMF constraints...\")\n        constraint_config = define_constraint_config(stat_analysis, np.min(story_lengths))\n        results['constraint_config'] = constraint_config\n        \n        # Step 4: Cross-validation search\n        logger.info(\"Step 4: Running cross-validation hyperparameter search...\")\n        cv_results = self.cross_validator.run_cross_validation(\n            electrode_data, story_lengths, max_components, \n            cv_splits, init_reps, test_frac, constraint_config\n        )\n        results['cross_validation'] = cv_results\n        \n        # Step 5: Fit final model on full dataset\n        logger.info(\"Step 5: Fitting final NNMF model...\")\n        final_model = self.model_fitter.fit_final_model(\n            electrode_data, \n            cv_results['optimal_components'],\n            cv_results['optimal_alpha'],\n            init_reps,\n            constraint_config\n        )\n        results['final_model'] = final_model\n        \n        # Step 6: Component analysis\n        logger.info(\"Step 6: Analyzing NNMF components...\")\n        component_analysis = self.component_analyzer.summarize_components(\n            final_model, electrode_data, story_halves, subject_ids\n        )\n        results['components'] = component_analysis\n        \n        # Step 7: Baseline comparisons\n        logger.info(\"Step 7: Running baseline method comparisons...\")\n        baseline_results = self.baseline_comparator.run_baseline_comparisons(\n            electrode_data, story_lengths, cv_results['optimal_components'],\n            cv_splits, constraint_config\n        )\n        results['baselines'] = baseline_results\n        \n        # Save results if requested\n        if save_results:\n            output_path = Path(output_dir) if output_dir else Path.cwd()\n            self._save_results(results, output_path)\n        \n        logger.info(\"Complete NNMF analysis pipeline finished successfully!\")\n        self._print_final_summary(results)\n        \n        return results\n    \n    def run_step_by_step(self, **kwargs) -> Dict[str, Any]:\n        \"\"\"Run pipeline step-by-step for interactive analysis.\"\"\"\n        # This allows users to run individual steps and inspect results\n        return self.run_complete_analysis(**kwargs)\n    \n    def _validate_inputs(\n        self, \n        electrode_data: np.ndarray, \n        story_lengths: np.ndarray, \n        subject_ids: List[str]\n    ):\n        \"\"\"Validate input data consistency.\"\"\"\n        n_electrodes, total_time = electrode_data.shape\n        \n        # Check dimensions\n        assert len(subject_ids) == n_electrodes, f\"Subject IDs length {len(subject_ids)} != n_electrodes {n_electrodes}\"\n        assert np.sum(story_lengths) == total_time, f\"Story lengths sum {np.sum(story_lengths)} != total_time {total_time}\"\n        \n        # Check data quality\n        assert not np.any(np.isnan(electrode_data)), \"Electrode data contains NaN values\"\n        assert not np.any(np.isinf(electrode_data)), \"Electrode data contains infinite values\"\n        assert np.all(story_lengths > 0), \"All story lengths must be positive\"\n        \n        logger.info(\"Input validation passed\")\n    \n    def _save_results(self, results: Dict[str, Any], output_dir: Path):\n        \"\"\"Save analysis results to disk.\"\"\"\n        output_dir.mkdir(exist_ok=True)\n        \n        # Save key results as numpy arrays\n        np.save(output_dir / 'final_model_W.npy', results['final_model']['W'])\n        np.save(output_dir / 'final_model_H.npy', results['final_model']['H'])\n        \n        # Save statistics and parameters as pickled dict\n        import pickle\n        with open(output_dir / 'analysis_results.pkl', 'wb') as f:\n            pickle.dump(results, f)\n        \n        logger.info(f\"Results saved to {output_dir}\")\n    \n    def _print_final_summary(self, results: Dict[str, Any]):\n        \"\"\"Print final analysis summary.\"\"\"\n        print(\"\\n\" + \"=\"*60)\n        print(\"NNMF ANALYSIS PIPELINE - FINAL SUMMARY\")\n        print(\"=\"*60)\n        \n        # Data summary\n        final_model = results['final_model']\n        stats = results['statistics']\n        cv = results['cross_validation']\n        \n        print(f\"\\nData Summary:\")\n        print(f\"  Electrodes: {final_model['W'].shape[0]}\")\n        print(f\"  Time points: {final_model['H'].shape[1]}\")\n        print(f\"  Stories: {results['data_splits']['story_halves']['n_stories']}\")\n        \n        print(f\"\\nOptimal Model:\")\n        print(f\"  Components: {final_model['n_components']}\")\n        print(f\"  Alpha: {final_model['alpha']:.4f}\")\n        print(f\"  Variance explained: {final_model['variance_explained']:.3f}\")\n        \n        print(f\"\\nElectrode Reliability:\")\n        n_electrodes = final_model['W'].shape[0]\n        n_sig_cr = stats['cross_run']['n_significant']\n        n_sig_perm = stats['cross_electrode']['n_significant_electrodes_perm']\n        print(f\"  Cross-run significant: {n_sig_cr}/{n_electrodes} ({100*n_sig_cr/n_electrodes:.1f}%)\")\n        print(f\"  Permutation significant: {n_sig_perm}/{n_electrodes} ({100*n_sig_perm/n_electrodes:.1f}%)\")\n        \n        print(f\"\\nModel Quality:\")\n        print(f\"  2nd-level correlation: {stats['cross_electrode']['second_level_correlation']:.3f}\")\n        print(f\"  Reconstruction error: {final_model['reconstruction_error']:.6f}\")\n        \n        print(\"\\n\" + \"=\"*60)\n\n\n# Convenience functions for direct usage\ndef run_nnmf_analysis(\n    electrode_data: np.ndarray,\n    story_lengths: np.ndarray, \n    subject_ids: List[str],\n    **kwargs\n) -> Dict[str, Any]:\n    \"\"\"Convenience function to run complete analysis.\"\"\"\n    pipeline = NNMFPipeline()\n    return pipeline.run_complete_analysis(\n        electrode_data, story_lengths, subject_ids, **kwargs\n    )\n\n\ndef load_analysis_results(results_path: str) -> Dict[str, Any]:\n    \"\"\"Load saved analysis results.\"\"\"\n    import pickle\n    with open(results_path, 'rb') as f:\n        return pickle.load(f)