"""
Story Data Loader for NNMF Pipeline

Integration script to load HDF5 story data and prepare it for NNMF analysis.
Integrates with existing nnmf_pipeline repository.
"""

import h5py
import numpy as np
import logging
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


def load_story_data(file_path: str) -> Dict[str, Any]:
    """
    Load story data from HDF5 file.
    
    Args:
        file_path: Path to HDF5 file containing story data
        
    Returns:
        Dictionary containing loaded data and metadata
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
        
    logger.info(f"Loading story data from: {file_path}")
    
    try:
        with h5py.File(file_path, 'r') as f:
            # Load all datasets
            data = {
                'electrode_data': f['electrode_data'][:],  # Shape: (1052, 12026)
                'bipolar_data': f['bipolar_data'][:] if 'bipolar_data' in f else None,
                'channel_names': [name.decode('utf-8') for name in f['channel_names'][:]],
                'subject_ids': [subj.decode('utf-8') for subj in f['subject_ids'][:]],
                'story_lengths': f['story_lengths'][:],
                'is_significant': f['is_significant'][:].astype(bool) if 'is_significant' in f else None,
                'is_langloc': f['is_langloc'][:].astype(bool) if 'is_langloc' in f else None
            }
            
            # Load attributes
            attrs = dict(f.attrs)
            for key, value in attrs.items():
                if isinstance(value, bytes):
                    attrs[key] = value.decode('utf-8')
            
            data.update(attrs)
            
    except Exception as e:
        logger.error(f"Failed to load data from {file_path}: {e}")
        raise
    
    logger.info(f"Data loaded successfully:")
    logger.info(f"  Story: {data.get('story_name', 'Unknown')}")
    logger.info(f"  Subjects: {data.get('num_subjects', 'Unknown')}")
    logger.info(f"  Electrodes: {data.get('num_electrodes', data['electrode_data'].shape[0])}")
    logger.info(f"  Electrode data shape: {data['electrode_data'].shape}")
    if data['bipolar_data'] is not None:
        logger.info(f"  Bipolar data shape: {data['bipolar_data'].shape}")
    
    return data


def preprocess_zscore_hg_for_nnmf(
    zscore_data: np.ndarray, 
    method: str = 'baseline_shift',
    preserve_dynamics: bool = True,
    baseline_percentile: float = 10.0
) -> np.ndarray:
    """
    Transform z-scored high gamma signals to positive values for NNMF analysis.
    
    Based on literature findings for neural data preprocessing.
    
    Args:
        zscore_data: [n_electrodes × n_samples] z-scored HG data (can have negatives)
        method: 'baseline_shift', 'percentile_shift', 'rectified_linear'
        preserve_dynamics: Whether to add small epsilon to prevent exact zeros
        baseline_percentile: Percentile to use as new baseline (for percentile_shift)
        
    Returns:
        Non-negative data matrix suitable for NNMF
    """
    data = zscore_data.copy().astype(np.float64)
    n_electrodes, n_samples = data.shape
    
    logger.info(f"Input z-scored data: shape={data.shape}, min={np.min(data):.3f}, max={np.max(data):.3f}")
    
    if method == 'baseline_shift':
        # Most common in literature - preserves all relative relationships
        min_val = np.min(data)
        if min_val < 0:
            data = data - min_val  # Shift so minimum becomes 0
        logger.info(f"Applied baseline shift: added {-min_val:.3f}")
        
    elif method == 'percentile_shift':
        # Use robust baseline (e.g., 10th percentile) as new zero
        baseline_val = np.percentile(data, baseline_percentile)
        data = data - baseline_val
        data = np.clip(data, 0, None)  # Clip remaining negatives
        logger.info(f"Applied percentile shift: subtracted {baseline_val:.3f}")
        
    elif method == 'rectified_linear':
        # Zero out negatives - preserves positive activations only
        negative_fraction = np.mean(data < 0)
        data = np.maximum(data, 0)
        logger.info(f"Applied ReLU: zeroed {negative_fraction:.1%} of values")
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Final safety checks
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    
    if np.any(data < 0):
        logger.warning(f"Still have {np.sum(data < 0)} negative values after transformation")
        data = np.clip(data, 0, None)
    
    assert np.all(data >= 0), f"Data still contains negatives: min={np.min(data)}"
    
    if preserve_dynamics:
        epsilon = 1e-8
        data = data + epsilon
        logger.info(f"Added epsilon={epsilon} to prevent exact zeros")
    
    logger.info(f"Output data: min={np.min(data):.6f}, max={np.max(data):.3f}")
    return data


def filter_electrodes(
    electrode_data: np.ndarray,
    subject_ids: List[str],
    is_significant: Optional[np.ndarray] = None,
    is_langloc: Optional[np.ndarray] = None,
    use_langloc: bool = True,
    min_subjects: int = 3
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """
    Filter electrodes based on significance and localization.
    
    Args:
        electrode_data: Electrode data matrix [n_electrodes × n_samples]
        subject_ids: Subject ID for each electrode
        is_significant: Boolean mask for significant electrodes
        is_langloc: Boolean mask for language-localized electrodes
        use_langloc: Whether to filter by language localization
        min_subjects: Minimum number of subjects required
        
    Returns:
        Tuple of (filtered_data, filtered_subject_ids, electrode_mask)
    """
    n_electrodes = electrode_data.shape[0]
    electrode_mask = np.ones(n_electrodes, dtype=bool)
    
    # Apply significance filter
    if is_significant is not None:
        electrode_mask = electrode_mask & is_significant
        logger.info(f"Significant electrodes: {np.sum(electrode_mask)}")
        
    # Apply language localization filter
    if use_langloc and is_langloc is not None:
        electrode_mask = electrode_mask & is_langloc
        logger.info(f"Language-localized electrodes: {np.sum(electrode_mask)}")
        
    # Check subject distribution
    filtered_subjects = np.array(subject_ids)[electrode_mask]
    unique_subjects = np.unique(filtered_subjects)
    
    logger.info(f"Subjects represented: {len(unique_subjects)} ({list(unique_subjects)})")
    
    if len(unique_subjects) < min_subjects:
        logger.warning(f"Only {len(unique_subjects)} subjects, less than minimum {min_subjects}")
        
    # Apply filter
    filtered_data = electrode_data[electrode_mask, :]
    filtered_subject_ids = filtered_subjects.tolist()
    
    logger.info(f"Filtered data shape: {filtered_data.shape}")
    
    return filtered_data, filtered_subject_ids, electrode_mask