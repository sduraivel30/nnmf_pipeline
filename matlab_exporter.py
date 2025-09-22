"""
MATLAB Exporter for NNMF Results

Exports NNMF analysis results to MATLAB-compatible format for visualization and interpretation.
Integrates with existing nnmf_pipeline repository.
"""

import numpy as np
import logging
from typing import Dict, Any
from pathlib import Path
import datetime

try:
    import scipy.io as sio
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

logger = logging.getLogger(__name__)


def export_results_to_matlab(results: Dict[str, Any], output_path: str) -> None:
    """
    Export NNMF results to MATLAB file.
    
    Args:
        results: NNMF analysis results dictionary
        output_path: Path to output .mat file
    """
    if not SCIPY_AVAILABLE:
        logger.error("scipy is required for MATLAB export. Install with: pip install scipy")
        return
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare data for MATLAB
    matlab_data = prepare_matlab_data(results)
    
    try:
        sio.savemat(str(output_path), matlab_data, format='5', do_compression=True)
        logger.info(f"Results exported to MATLAB format: {output_path}")
    except Exception as e:
        logger.error(f"Failed to export to MATLAB: {e}")
        raise


def prepare_matlab_data(results: Dict[str, Any]) -> Dict[str, Any]:
    """Prepare results data for MATLAB export."""
    matlab_data = {}
    
    # Final model results
    if 'final_model' in results:
        final_model = results['final_model']
        matlab_data['final_model'] = {
            'W': final_model.get('W', np.array([])),
            'H': final_model.get('H', np.array([])),
            'n_components': final_model.get('n_components', 0),
            'alpha': final_model.get('alpha', 0.0),
            'variance_explained': final_model.get('variance_explained', 0.0),
            'reconstruction_error': final_model.get('reconstruction_error', 0.0),
            'convergence_costs': final_model.get('costs', np.array([]))
        }
    
    # Cross-validation results
    if 'cross_validation' in results:
        cv = results['cross_validation']
        matlab_data['cross_validation'] = {
            'optimal_components': cv.get('optimal_components', 0),
            'optimal_alpha': cv.get('optimal_alpha', 0.0),
            'component_range': np.array(cv.get('component_range', [])),
            'alpha_range': np.array(cv.get('alpha_range', [])),
            'mean_recon_errors': cv.get('mean_recon_errors', np.array([])),
            'mean_reliability': cv.get('mean_reliability', np.array([])),
            'composite_score': cv.get('composite_score', np.array([]))
        }
    
    # Statistical analysis
    if 'statistics' in results:
        stats = results['statistics']
        matlab_stats = {}
        
        if 'cross_run' in stats:
            cr = stats['cross_run']
            matlab_stats['cross_run'] = {
                'n_significant': cr.get('n_significant', 0),
                'correlations': np.array(cr.get('correlations', [])),
                'p_values': np.array(cr.get('p_values', [])),
                'significant_electrodes_mask': cr.get('significant_electrodes_mask', np.array([], dtype=bool))
            }
        
        if 'cross_electrode' in stats:
            ce = stats['cross_electrode']
            matlab_stats['cross_electrode'] = {
                'n_significant_electrodes_perm': ce.get('n_significant_electrodes_perm', 0),
                'second_level_correlation': ce.get('second_level_correlation', 0.0),
                'permutation_p_values': np.array(ce.get('permutation_p_values', []))
            }
        
        matlab_data['statistics'] = matlab_stats
    
    # Component analysis
    if 'components' in results:
        components = results['components']
        matlab_components = {}
        
        for key, comp in components.items():
            if isinstance(comp, dict):
                matlab_components[key] = {
                    'spatial_sparsity': comp.get('spatial_sparsity', 0.0),
                    'temporal_sparsity': comp.get('temporal_sparsity', 0.0),
                    'peak_activation': comp.get('peak_activation', 0.0),
                    'duration': comp.get('duration', 0.0)
                }
        
        matlab_data['components'] = matlab_components
    
    # Data splits information
    if 'data_splits' in results:
        splits = results['data_splits']
        matlab_data['data_splits'] = {
            'run1_shape': splits.get('run1', np.array([])).shape if 'run1' in splits else (0, 0),
            'run2_shape': splits.get('run2', np.array([])).shape if 'run2' in splits else (0, 0),
            'story_halves_info': 'Data split into temporal halves'
        }
    
    # Metadata
    if 'metadata' in results:
        metadata = results['metadata']
        matlab_data['metadata'] = {
            'story_name': metadata.get('story_name', ''),
            'file_path': metadata.get('file_path', ''),
            'original_electrodes': metadata.get('original_electrodes', 0),
            'filtered_electrodes': metadata.get('filtered_electrodes', 0),
            'electrode_mask': metadata.get('electrode_mask', np.array([], dtype=bool)),
            'use_bipolar': metadata.get('use_bipolar', False),
            'use_langloc': metadata.get('use_langloc', False),
            'preprocessing_method': metadata.get('preprocessing_method', ''),
            'n_subjects': metadata.get('n_subjects', 0),
            'epoch_start': metadata.get('epoch_start', 0.0),
            'epoch_end': metadata.get('epoch_end', 0.0),
            'story_duration_samples': metadata.get('story_duration', 0)
        }
    
    # Add analysis information
    matlab_data['analysis_info'] = {
        'export_timestamp': datetime.datetime.now().isoformat(),
        'python_version': get_python_version(),
        'analysis_type': 'NNMF_Story_Analysis',
        'pipeline_version': '1.0'
    }
    
    return matlab_data


def get_python_version() -> str:
    """Get Python version string."""
    import sys
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"