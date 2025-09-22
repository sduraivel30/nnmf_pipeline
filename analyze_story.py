"""
Story Analysis Script for NNMF Pipeline

Main analysis script that integrates with existing NNMF pipeline to analyze story data.
Place this in the root of your nnmf_pipeline repository.
"""

import numpy as np
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional

# Import your existing NNMF pipeline
from src.main_pipeline import NNMFPipeline
from story_data_loader import load_story_data, preprocess_zscore_hg_for_nnmf, filter_electrodes
from matlab_exporter import export_results_to_matlab
from story_visualizer import plot_story_overview, plot_nnmf_results


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def analyze_story_with_nnmf(
    file_path: str,
    use_bipolar: bool = False,
    use_langloc: bool = True,
    preprocessing_method: str = 'baseline_shift',
    max_components: int = 15,
    cv_splits: int = 3,
    init_reps: int = 5,
    test_frac: float = 0.2,
    n_permutations: int = 50,
    plot_overview: bool = True,
    save_results: bool = True,
    output_dir: str = './story_results'
) -> Optional[Dict[str, Any]]:
    """
    Complete NNMF analysis pipeline for story data.
    
    Args:
        file_path: Path to HDF5 story data file
        use_bipolar: Whether to use bipolar vs monopolar data
        use_langloc: Whether to filter for language-localized electrodes
        preprocessing_method: Method for making data positive ('baseline_shift', etc.)
        max_components: Maximum number of components to test
        cv_splits: Number of cross-validation splits
        init_reps: Number of NNMF initializations per fit
        test_frac: Fraction of data for testing
        n_permutations: Number of permutation tests
        plot_overview: Whether to plot data overview
        save_results: Whether to save results
        output_dir: Directory to save results
        
    Returns:
        Dictionary containing NNMF analysis results
    """
    try:
        # Load story data
        logger.info("Loading story data...")
        data = load_story_data(file_path)
        
        # Plot overview if requested
        if plot_overview:
            plot_story_overview(data)
        
        # Select data type
        if use_bipolar and data['bipolar_data'] is not None:
            electrode_data = data['bipolar_data']
            logger.info("Using bipolar data")
        else:
            electrode_data = data['electrode_data']
            logger.info("Using monopolar electrode data")
        
        # Filter electrodes
        filtered_data, filtered_subject_ids, electrode_mask = filter_electrodes(
            electrode_data,
            data['subject_ids'],
            data['is_significant'],
            data['is_langloc'],
            use_langloc=use_langloc
        )
        
        # Preprocess for NNMF
        logger.info("Preprocessing data for NNMF...")
        processed_data = preprocess_zscore_hg_for_nnmf(
            filtered_data, 
            method=preprocessing_method
        )
        
        # Initialize NNMF pipeline
        pipeline = NNMFPipeline(
            n_permutations=n_permutations,
            n_jobs=-1,
            random_state=42
        )
        
        # Run NNMF analysis
        logger.info("Running NNMF analysis...")
        results = pipeline.run_complete_analysis(
            electrode_data=processed_data,
            story_lengths=data['story_lengths'],
            subject_ids=filtered_subject_ids,
            max_components=max_components,
            cv_splits=cv_splits,
            init_reps=init_reps,
            test_frac=test_frac,
            save_results=False  # We handle saving
        )
        
        if results is None:
            logger.error("NNMF analysis failed")
            return None
        
        # Add metadata to results
        results['metadata'] = {
            'story_name': data.get('story_name', 'Unknown'),
            'file_path': str(file_path),
            'original_electrodes': electrode_data.shape[0],
            'filtered_electrodes': processed_data.shape[0],
            'electrode_mask': electrode_mask,
            'use_bipolar': use_bipolar,
            'use_langloc': use_langloc,
            'preprocessing_method': preprocessing_method,
            'n_subjects': len(set(filtered_subject_ids)),
            'epoch_start': data.get('epoch_start', 0),
            'epoch_end': data.get('epoch_end', 0),
            'story_duration': np.sum(data['story_lengths'])
        }
        
        # Plot results
        plot_nnmf_results(results)
        
        # Save results if requested
        if save_results:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            story_name = data.get('story_name', 'Unknown')
            
            # Export to MATLAB
            matlab_path = output_path / f"{story_name}_nnmf_results.mat"
            export_results_to_matlab(results, matlab_path)
            
            # Save Python results as numpy
            python_path = output_path / f"{story_name}_nnmf_results.npz"
            save_python_results(results, python_path)
            
            logger.info(f"Results saved to {output_dir}/")
        
        logger.info("Story analysis completed successfully!")
        return results
        
    except Exception as e:
        logger.error(f"Story analysis failed: {e}")
        return None


def save_python_results(results: Dict[str, Any], output_path: Path) -> None:
    """Save results as compressed numpy arrays."""
    # Prepare data for numpy save
    save_dict = {}
    
    # Final model
    if 'final_model' in results:
        fm = results['final_model']
        save_dict.update({
            'W': fm.get('W', np.array([])),
            'H': fm.get('H', np.array([])),
            'n_components': fm.get('n_components', 0),
            'alpha': fm.get('alpha', 0.0),
            'variance_explained': fm.get('variance_explained', 0.0)
        })
    
    # Cross-validation
    if 'cross_validation' in results:
        cv = results['cross_validation']
        save_dict.update({
            'cv_mean_recon_errors': cv.get('mean_recon_errors', np.array([])),
            'cv_mean_reliability': cv.get('mean_reliability', np.array([])),
            'cv_component_range': np.array(cv.get('component_range', [])),
            'cv_alpha_range': np.array(cv.get('alpha_range', []))
        })
    
    # Metadata
    if 'metadata' in results:
        meta = results['metadata']
        save_dict.update({
            'story_name': meta.get('story_name', ''),
            'electrode_mask': meta.get('electrode_mask', np.array([])),
            'filtered_electrodes': meta.get('filtered_electrodes', 0),
            'original_electrodes': meta.get('original_electrodes', 0)
        })
    
    np.savez_compressed(output_path, **save_dict)
    logger.info(f"Python results saved to: {output_path}")


def print_analysis_summary(results: Dict[str, Any]) -> None:
    """Print a summary of the analysis results."""
    if results is None:
        print("No results to summarize")
        return
    
    metadata = results.get('metadata', {})
    final_model = results.get('final_model', {})
    
    print("\n" + "="*60)
    print("NNMF STORY ANALYSIS SUMMARY")
    print("="*60)
    print(f"Story: {metadata.get('story_name', 'Unknown')}")
    print(f"Electrodes used: {metadata.get('filtered_electrodes', 0)}/{metadata.get('original_electrodes', 0)}")
    print(f"Subjects: {metadata.get('n_subjects', 0)}")
    print(f"Preprocessing: {metadata.get('preprocessing_method', 'Unknown')}")
    print(f"Language electrodes: {metadata.get('use_langloc', False)}")
    print(f"Bipolar data: {metadata.get('use_bipolar', False)}")
    print()
    print(f"Optimal components: {final_model.get('n_components', 0)}")
    print(f"Optimal alpha: {final_model.get('alpha', 0):.4f}")
    print(f"Variance explained: {final_model.get('variance_explained', 0):.3f}")
    
    if 'cross_validation' in results:
        cv = results['cross_validation']
        print(f"Mean CV error: {np.mean(cv.get('mean_recon_errors', [0])):.6f}")
        print(f"Mean CV reliability: {np.mean(cv.get('mean_reliability', [0])):.3f}")
    
    if 'statistics' in results:
        stats = results['statistics']
        if 'cross_run' in stats:
            n_sig = stats['cross_run'].get('n_significant', 0)
            print(f"Significant electrodes (cross-run): {n_sig}")
    
    print("="*60)


def run_aqua_analysis():
    """Example: Run NNMF analysis on Aqua story data."""
    
    # Update this path to your actual file location
    file_path = "Aqua_20250902_180809_complete_data.h5"
    
    if not Path(file_path).exists():
        print(f"Error: File not found at {file_path}")
        print("Please update the file_path variable in run_aqua_analysis()")
        return None
    
    # Run analysis
    results = analyze_story_with_nnmf(
        file_path=file_path,
        use_bipolar=False,  # Use monopolar data
        use_langloc=True,   # Filter for language electrodes
        preprocessing_method='baseline_shift',
        max_components=15,
        cv_splits=3,
        init_reps=5,
        test_frac=0.2,
        n_permutations=50,  # Reduced for faster computation
        plot_overview=True,
        save_results=True,
        output_dir="./aqua_results"
    )
    
    if results is not None:
        print_analysis_summary(results)
        
        print("\nFiles saved:")
        print("- aqua_results/Aqua_nnmf_results.mat (MATLAB format)")
        print("- aqua_results/Aqua_nnmf_results.npz (Python format)")
        print("\nTo analyze in MATLAB:")
        print("  addpath('matlab_scripts');")
        print("  results = load_nnmf_results('aqua_results/Aqua_nnmf_results.mat');")
        print("  visualize_components(results);")
        print("  analysis = interpret_results(results);")
    
    return results


if __name__ == "__main__":
    # Run the analysis
    results = run_aqua_analysis()