"""
Data Splitting Module - Norman-Haignere Methodology
Handles temporal splitting of story data into runs for reliability analysis.
"""

import numpy as np
from typing import Tuple, Dict, List, Any
import logging

logger = logging.getLogger(__name__)


def split_stories_into_runs(
    electrode_data: np.ndarray, 
    story_lengths: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Split story data using Norman-Haignere methodology with pipeline compatibility.
    
    Based on Norman-Haignere et al. 2022: temporal splitting within stories
    for split-half reliability analysis.
    
    Args:
        electrode_data: [n_electrodes × total_time] concatenated story data
        story_lengths: [n_stories] length of each story in samples
        
    Returns:
        run1: [n_electrodes × half_time] first temporal half
        run2: [n_electrodes × half_time] second temporal half  
        story_halves: dict with story boundary information
    """
    n_electrodes, total_time = electrode_data.shape
    n_stories = len(story_lengths)
    
    # Verify story lengths sum correctly
    assert np.sum(story_lengths) == total_time, f"Story lengths sum to {np.sum(story_lengths)}, expected {total_time}"
    
    # Calculate total samples for each run
    half_lengths = np.floor(story_lengths / 2).astype(int)
    total_half_samples = np.sum(half_lengths)
    
    # Pre-allocate output matrices
    run1 = np.zeros((n_electrodes, total_half_samples), dtype=electrode_data.dtype)
    run2 = np.zeros((n_electrodes, total_half_samples), dtype=electrode_data.dtype)
    
    # Initialize story_halves structure
    story_halves = {
        'n_stories': n_stories,
        'boundaries': np.zeros((n_stories, 4), dtype=int),  # [start, mid, end, half_length]
        'story_indices': [],
        'total_half_time': total_half_samples,
        'split_method': 'temporal_halves',
        'description': 'Stories split into first and second temporal halves'
    }
    
    # Track positions
    current_data_pos = 0
    run1_pos = 0
    run2_pos = 0
    
    logger.info(f"Splitting {n_stories} stories into temporal halves...")
    
    for s in range(n_stories):
        story_len = story_lengths[s]
        half_len = half_lengths[s]
        
        # Extract current story data
        story_start = current_data_pos
        story_end = current_data_pos + story_len
        story_data = electrode_data[:, story_start:story_end]
        
        # Split into temporal halves (Norman-Haignere method)
        first_half = story_data[:, :half_len]
        second_half = story_data[:, -half_len:]
        
        # Store in run arrays
        run1[:, run1_pos:run1_pos + half_len] = first_half
        run2[:, run2_pos:run2_pos + half_len] = second_half
        
        # Store boundary information
        story_mid = story_start + half_len
        story_halves['boundaries'][s] = [story_start, story_mid, story_end, half_len]
        story_halves['story_indices'].append({
            'run1': (run1_pos, run1_pos + half_len),
            'run2': (run2_pos, run2_pos + half_len)
        })
        
        # Update positions
        current_data_pos = story_end
        run1_pos += half_len
        run2_pos += half_len
    
    logger.info(f"Data split complete: {n_electrodes} electrodes, {n_stories} stories, {total_half_samples} time points per run")
    
    return run1, run2, story_halves


def remove_common_mode_per_subject(
    run1: np.ndarray,
    run2: np.ndarray, 
    subject_ids: List[str]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove common mode signal within each subject before correlation analysis.
    
    Args:
        run1: [n_electrodes × time] first run data
        run2: [n_electrodes × time] second run data
        subject_ids: [n_electrodes] list of subject IDs per electrode
        
    Returns:
        run1_demeaned: Common mode removed from run1
        run2_demeaned: Common mode removed from run2
    """
    run1_demeaned = run1.copy()
    run2_demeaned = run2.copy()
    
    unique_subjects = list(set(subject_ids))
    
    for subject in unique_subjects:
        # Get electrode indices for this subject
        subject_mask = np.array([subj == subject for subj in subject_ids])
        subject_indices = np.where(subject_mask)[0]
        
        if len(subject_indices) > 1:  # Only demean if multiple electrodes
            # Compute mean signal per subject (common mode)
            mean_run1 = np.mean(run1[subject_indices], axis=0)
            mean_run2 = np.mean(run2[subject_indices], axis=0)
            
            # Subtract common mode from each electrode for this subject
            run1_demeaned[subject_indices] = run1[subject_indices] - mean_run1[np.newaxis, :]
            run2_demeaned[subject_indices] = run2[subject_indices] - mean_run2[np.newaxis, :]
    
    logger.info(f"Common mode removed for {len(unique_subjects)} subjects")
    return run1_demeaned, run2_demeaned


def create_electrode_stratified_split(
    n_electrodes: int, 
    test_frac: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create electrode split for cross-validation.
    
    Based on Norman-Haignere et al. 2022: electrode-level stratification
    for cross-validation.
    
    Args:
        n_electrodes: Total number of electrodes
        test_frac: Fraction of electrodes for test set
        
    Returns:
        train_idx: Indices of training electrodes
        test_idx: Indices of test electrodes
    """
    n_test = int(np.round(n_electrodes * test_frac))
    
    # Ensure at least one electrode in each split
    n_test = max(1, min(n_test, n_electrodes - 1))
    
    # Random split
    all_indices = np.arange(n_electrodes)
    np.random.shuffle(all_indices)
    
    test_idx = all_indices[:n_test]
    train_idx = all_indices[n_test:]
    
    return train_idx, test_idx