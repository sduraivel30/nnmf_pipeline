"""
Data Loading Module - Pool Story Data Across Subjects
Python implementation of MATLAB pool_story_data_across_subjects function.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from scipy.io import loadmat
import h5py

logger = logging.getLogger(__name__)


class StoryDataPooler:
    """Pools electrode data across subjects with comprehensive tracking."""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        
    def pool_story_data_across_subjects(
        self,
        data_dir: Union[str, Path],
        epoch: List[float] = None,
        subject_list: List[str] = None,
        subject_info_in: Dict = None,
        summary_stats_file: str = "",
        include_significant: bool = False
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Pool electrode data for a story across subjects with subject ID tracking.
        
        Args:
            data_dir: Directory containing *HG_ZScore.mat files
            epoch: Time epoch [start_time, end_time] in seconds (default: [-0.5, 180])
            subject_list: List of subject IDs to include (default: None = all)
            subject_info_in: Optional dict with subject-specific channel filtering
            summary_stats_file: Path to summary stats file for langloc channels
            include_significant: Include significant channels in findStoryDataEpoch
            
        Returns:
            pooled_data: Dict with pooled electrode data and subject IDs
            subject_info_out: Dict with comprehensive subject and channel information
        """
        # Set defaults
        if epoch is None:
            epoch = [-0.5, 180.0]
        if subject_list is None:
            subject_list = []
            
        if self.verbose:
            print("--- Pooling story data across subjects with subject ID tracking ---")
            
        # Get story name from user input
        story_name = input("Enter the story name: ")
        if self.verbose:
            print(f"Story to pool: \"{story_name}\"")
            
        # Load language localization data
        langloc_map_unipolar, langloc_map_bipolar, has_langloc_data = self._load_langloc_data(
            summary_stats_file
        )
        
        # Find HG_ZScore.mat files
        data_dir = Path(data_dir)
        hg_files = list(data_dir.glob("*HG_ZScore.mat"))
        
        if not hg_files:
            raise FileNotFoundError(f"No HG_ZScore.mat files found in {data_dir}")
            
        if self.verbose:
            print(f"Found {len(hg_files)} subject files")
            
        # Initialize data structures
        pooled_data, subject_info_out = self._initialize_data_structures(
            story_name, epoch, has_langloc_data, summary_stats_file, include_significant
        )
        
        # Process each subject file
        all_uni_names = []
        all_bip_names = []
        all_uni_subject_ids = []
        all_bip_subject_ids = []
        all_uni_significant = []
        all_bip_significant = []
        
        subject_channel_counts = {}
        subject_bipolar_channel_counts = {}
        
        for hg_file in hg_files:
            subject_id = self._extract_subject_id(hg_file.name)
            
            # Skip if not in subject list
            if subject_list and subject_id not in subject_list:
                if self.verbose:
                    print(f"Skipping {subject_id} (not in subject_list)")
                continue
                
            if self.verbose:
                print(f"Processing subject {subject_id}...")
                
            # Load and process subject data
            try:
                story_data = self._load_subject_story_data(
                    hg_file, story_name, epoch, include_significant
                )
                
                if story_data is None:
                    if self.verbose:
                        print("  Story not found, skipping.")
                    continue
                    
                # Apply subject-specific filtering
                story_data = self._apply_subject_filtering(
                    story_data, subject_id, subject_info_in
                )
                
                # Process unipolar data
                if story_data['epoch_data'].shape[0] > 0:
                    n_uni_channels = story_data['epoch_data'].shape[0]
                    
                    # Append data
                    if pooled_data['unipolar'] is None:
                        pooled_data['unipolar'] = story_data['epoch_data']
                    else:
                        pooled_data['unipolar'] = np.vstack([
                            pooled_data['unipolar'], 
                            story_data['epoch_data']
                        ])
                    
                    # Track metadata
                    all_uni_names.extend(story_data['channel_names'])
                    all_uni_subject_ids.extend([subject_id] * n_uni_channels)
                    all_uni_significant.extend(
                        story_data.get('is_significant', [False] * n_uni_channels)
                    )
                    
                    subject_channel_counts[subject_id] = n_uni_channels
                    
                    if self.verbose:
                        print(f"  Added {n_uni_channels} unipolar channels from {subject_id}")
                
                # Process bipolar data
                if story_data.get('epoch_data_bip') is not None and story_data['epoch_data_bip'].shape[0] > 0:
                    n_bip_channels = story_data['epoch_data_bip'].shape[0]
                    
                    # Append data
                    if pooled_data['bipolar'] is None:
                        pooled_data['bipolar'] = story_data['epoch_data_bip']
                    else:
                        pooled_data['bipolar'] = np.vstack([
                            pooled_data['bipolar'],
                            story_data['epoch_data_bip']
                        ])
                    
                    # Track metadata
                    all_bip_names.extend(story_data.get('bipolar_channel_names', []))
                    all_bip_subject_ids.extend([subject_id] * n_bip_channels)
                    all_bip_significant.extend(
                        story_data.get('is_significant_bipolar', [False] * n_bip_channels)
                    )
                    
                    subject_bipolar_channel_counts[subject_id] = n_bip_channels
                    subject_info_out['has_bipolar_data'] = True
                    
                    if self.verbose:
                        print(f"  Added {n_bip_channels} bipolar channels from {subject_id}")
                
                # Add to processed subjects
                subject_info_out['subjects'].append(subject_id)
                
            except Exception as e:
                logger.warning(f"Failed to process {subject_id}: {str(e)}")
                continue
        
        # Remove NaN channels
        pooled_data, all_uni_names, all_uni_subject_ids, all_uni_significant = self._remove_nan_channels(
            pooled_data, 'unipolar', all_uni_names, all_uni_subject_ids, all_uni_significant
        )
        
        if subject_info_out['has_bipolar_data']:
            pooled_data, all_bip_names, all_bip_subject_ids, all_bip_significant = self._remove_nan_channels(
                pooled_data, 'bipolar', all_bip_names, all_bip_subject_ids, all_bip_significant
            )
        
        # Store final results
        pooled_data['subject_ids'] = all_uni_subject_ids
        if subject_info_out['has_bipolar_data']:
            pooled_data['subject_ids_bipolar'] = all_bip_subject_ids
        
        # Update subject info
        subject_info_out.update({
            'channel_names': all_uni_names,
            'subject_ids': all_uni_subject_ids,
            'num_unipolar_channels': len(all_uni_names),
            'is_significant': all_uni_significant,
            'subject_channel_count': subject_channel_counts
        })
        
        if subject_info_out['has_bipolar_data']:
            subject_info_out.update({
                'bipolar_channel_names': all_bip_names,
                'subject_ids_bipolar': all_bip_subject_ids,
                'num_bipolar_channels': len(all_bip_names),
                'is_significant_bipolar': all_bip_significant,
                'subject_bipolar_channel_count': subject_bipolar_channel_counts
            })
        
        # Generate language localization masks
        if has_langloc_data:
            self._generate_langloc_masks(
                subject_info_out, all_uni_names, all_uni_subject_ids,
                all_bip_names, all_bip_subject_ids,
                langloc_map_unipolar, langloc_map_bipolar
            )
        
        # Print summary
        if self.verbose:
            self._print_summary(subject_info_out, has_langloc_data)
        
        return pooled_data, subject_info_out
    
    def _load_langloc_data(self, summary_stats_file: str) -> Tuple[Dict, Dict, bool]:
        """Load language localization data from summary stats file."""
        langloc_map_unipolar = {}
        langloc_map_bipolar = {}
        has_langloc_data = False
        
        if not summary_stats_file or not Path(summary_stats_file).exists():
            return langloc_map_unipolar, langloc_map_bipolar, has_langloc_data
            
        if self.verbose:
            print(f"Loading language localization stats from {summary_stats_file}...")
        
        try:
            # Try loading as .mat file first
            if summary_stats_file.endswith('.mat'):
                data = loadmat(summary_stats_file)
                if 'all_summary_stats' in data:
                    # Convert MATLAB struct array to usable format
                    stats = data['all_summary_stats']
                    has_langloc_data = True
                    
                    # Parse MATLAB struct array (implementation depends on exact format)
                    # This is a simplified version - may need adjustment based on actual data structure
                    for i in range(len(stats)):
                        subj = str(stats[i]['subject'][0]) if 'subject' in stats.dtype.names else f"subject_{i}"
                        
                        if 'names_sig' in stats.dtype.names:
                            uni_chans = stats[i]['names_sig']
                            if uni_chans is not None:
                                langloc_map_unipolar[subj] = [str(ch) for ch in uni_chans.flatten()]
                        
                        if 'names_sig_bipolar' in stats.dtype.names:
                            bip_chans = stats[i]['names_sig_bipolar']
                            if bip_chans is not None:
                                langloc_map_bipolar[subj] = [str(ch) for ch in bip_chans.flatten()]
                                
            else:
                # Try loading as CSV/Excel
                df = pd.read_csv(summary_stats_file) if summary_stats_file.endswith('.csv') else pd.read_excel(summary_stats_file)
                has_langloc_data = True
                
                for _, row in df.iterrows():
                    subj = str(row.get('subject', ''))
                    if 'names_sig' in row and pd.notna(row['names_sig']):
                        langloc_map_unipolar[subj] = str(row['names_sig']).split(',')
                    if 'names_sig_bipolar' in row and pd.notna(row['names_sig_bipolar']):
                        langloc_map_bipolar[subj] = str(row['names_sig_bipolar']).split(',')
                        
            if self.verbose and has_langloc_data:
                print(f"Loaded langloc data for {len(langloc_map_unipolar)} subjects")
                print(f"  Unipolar langloc maps: {len(langloc_map_unipolar)} subjects")
                print(f"  Bipolar langloc maps: {len(langloc_map_bipolar)} subjects")
                
        except Exception as e:
            logger.warning(f"Failed to load summary stats file: {str(e)}")
            
        return langloc_map_unipolar, langloc_map_bipolar, has_langloc_data
    
    def _initialize_data_structures(
        self, story_name: str, epoch: List[float], has_langloc_data: bool,
        summary_stats_file: str, include_significant: bool
    ) -> Tuple[Dict, Dict]:
        """Initialize data structures for pooled data and subject info."""
        pooled_data = {
            'unipolar': None,
            'bipolar': None,
            'subject_ids': [],
            'subject_ids_bipolar': []
        }
        
        subject_info_out = {
            'subjects': [],
            'channel_names': [],
            'bipolar_channel_names': [],
            'subject_ids': [],
            'subject_ids_bipolar': [],
            'story_name': story_name,
            'epoch': epoch,
            'num_unipolar_channels': 0,
            'num_bipolar_channels': 0,
            'has_bipolar_data': False,
            'subject_channel_count': {},
            'subject_bipolar_channel_count': {},
            'is_significant': [],
            'is_langloc': [],
            'is_significant_bipolar': [],
            'is_langloc_bipolar': [],
            'has_langloc_data': has_langloc_data,
            'summary_stats_file': summary_stats_file,
            'include_significant': include_significant
        }
        
        return pooled_data, subject_info_out
    
    def _extract_subject_id(self, filename: str) -> str:
        """Extract subject ID from filename."""
        # Remove .mat extension and HG_ZScore suffix
        name = Path(filename).stem
        if name.endswith('_HG_ZScore'):
            name = name[:-10]  # Remove '_HG_ZScore'
        elif name.endswith('HG_ZScore'):
            name = name[:-9]   # Remove 'HG_ZScore'
        return name
    
    def _load_subject_story_data(
        self, hg_file: Path, story_name: str, epoch: List[float], include_significant: bool
    ) -> Optional[Dict[str, Any]]:
        """Load story data for a specific subject."""
        try:
            # Load MATLAB file
            data = loadmat(str(hg_file))
            obj = data.get('obj', data)  # Handle different file structures
            
            # Find story data (simplified implementation)
            # This would need to be adapted based on the actual structure of your MATLAB data
            story_data = self._find_story_data_epoch(
                obj, story_name, epoch, include_significant
            )
            
            return story_data
            
        except Exception as e:
            logger.error(f"Failed to load {hg_file}: {str(e)}")
            return None
    
    def _find_story_data_epoch(
        self, obj: Dict, story_name: str, epoch: List[float], include_significant: bool
    ) -> Optional[Dict[str, Any]]:
        """
        Find story data for specified epoch.
        This is a simplified implementation - would need to be adapted 
        based on your actual MATLAB data structure.
        """
        # Placeholder implementation - adapt based on your data structure
        # This would typically involve:
        # 1. Finding the story in the object structure
        # 2. Extracting the specified epoch
        # 3. Getting channel names and significance info
        
        try:
            # Example structure - adapt as needed
            if 'stories' in obj and story_name in obj['stories']:
                story_obj = obj['stories'][story_name]
                
                # Extract epoch data
                epoch_data = story_obj.get('data', np.array([]))
                channel_names = story_obj.get('channel_names', [])
                
                # Get bipolar data if available
                epoch_data_bip = story_obj.get('bipolar_data', None)
                bipolar_channel_names = story_obj.get('bipolar_channel_names', [])
                
                # Get significance info if available
                is_significant = story_obj.get('is_significant', None)
                is_significant_bipolar = story_obj.get('is_significant_bipolar', None)
                
                return {
                    'epoch_data': epoch_data,
                    'channel_names': channel_names,
                    'epoch_data_bip': epoch_data_bip,
                    'bipolar_channel_names': bipolar_channel_names,
                    'is_significant': is_significant,
                    'is_significant_bipolar': is_significant_bipolar
                }
            else:
                return None
                
        except Exception as e:
            logger.error(f"Failed to find story data: {str(e)}")
            return None
    
    def _apply_subject_filtering(
        self, story_data: Dict, subject_id: str, subject_info_in: Optional[Dict]
    ) -> Dict:
        """Apply per-subject channel filtering."""
        if subject_info_in is None:
            return story_data
            
        # Find subject in filtering info
        if 'subjects' in subject_info_in and subject_id in subject_info_in['subjects']:
            idx = subject_info_in['subjects'].index(subject_id)
            
            if self.verbose:
                print(f"  Applying channel filter for {subject_id}")
            
            # Filter unipolar channels
            if 'channel_names' in subject_info_in and idx < len(subject_info_in['channel_names']):
                keep_channels = subject_info_in['channel_names'][idx]
                keep_uni = [ch in keep_channels for ch in story_data['channel_names']]
                
                story_data['epoch_data'] = story_data['epoch_data'][keep_uni]
                story_data['channel_names'] = [ch for i, ch in enumerate(story_data['channel_names']) if keep_uni[i]]
                
                if story_data.get('is_significant') is not None:
                    story_data['is_significant'] = [sig for i, sig in enumerate(story_data['is_significant']) if keep_uni[i]]
            
            # Filter bipolar channels
            if ('bipolar_channel_names' in subject_info_in and 
                idx < len(subject_info_in['bipolar_channel_names']) and
                story_data.get('epoch_data_bip') is not None):
                
                keep_bip_channels = subject_info_in['bipolar_channel_names'][idx]
                keep_bip = [ch in keep_bip_channels for ch in story_data['bipolar_channel_names']]
                
                story_data['epoch_data_bip'] = story_data['epoch_data_bip'][keep_bip]
                story_data['bipolar_channel_names'] = [ch for i, ch in enumerate(story_data['bipolar_channel_names']) if keep_bip[i]]
                
                if story_data.get('is_significant_bipolar') is not None:
                    story_data['is_significant_bipolar'] = [sig for i, sig in enumerate(story_data['is_significant_bipolar']) if keep_bip[i]]
        
        return story_data
    
    def _remove_nan_channels(
        self, pooled_data: Dict, data_key: str, 
        channel_names: List, subject_ids: List, significant: List
    ) -> Tuple[Dict, List, List, List]:
        """Remove channels with NaN values."""
        if pooled_data[data_key] is None or len(pooled_data[data_key]) == 0:
            return pooled_data, channel_names, subject_ids, significant
            
        if self.verbose:
            print(f"Checking for NaNs in {data_key} data...")
            
        nan_mask = np.any(np.isnan(pooled_data[data_key]), axis=1)
        num_nan_removed = np.sum(nan_mask)
        
        if num_nan_removed > 0:
            # Remove NaN channels
            pooled_data[data_key] = pooled_data[data_key][~nan_mask]
            
            # Remove corresponding metadata
            channel_names = [ch for i, ch in enumerate(channel_names) if not nan_mask[i]]
            subject_ids = [sid for i, sid in enumerate(subject_ids) if not nan_mask[i]]
            significant = [sig for i, sig in enumerate(significant) if not nan_mask[i]]
            
            if self.verbose:
                print(f"  Removed {num_nan_removed} {data_key} channels with NaN values")
        
        return pooled_data, channel_names, subject_ids, significant
    
    def _generate_langloc_masks(
        self, subject_info_out: Dict, 
        all_uni_names: List, all_uni_subject_ids: List,
        all_bip_names: List, all_bip_subject_ids: List,
        langloc_map_unipolar: Dict, langloc_map_bipolar: Dict
    ):
        """Generate language localization channel masks."""
        if self.verbose:
            print("Generating langloc channel masks...")
        
        # Unipolar langloc mask
        n_channels = len(all_uni_names)
        is_langloc = [False] * n_channels
        
        for ch_idx in range(n_channels):
            subject_id = all_uni_subject_ids[ch_idx]
            chan_name = all_uni_names[ch_idx]
            
            if subject_id in langloc_map_unipolar:
                langloc_chans = langloc_map_unipolar[subject_id]
                if chan_name in langloc_chans:
                    is_langloc[ch_idx] = True
        
        subject_info_out['is_langloc'] = is_langloc
        
        # Bipolar langloc mask
        if subject_info_out['has_bipolar_data'] and all_bip_names:
            n_bip_channels = len(all_bip_names)
            is_langloc_bip = [False] * n_bip_channels
            
            for ch_idx in range(n_bip_channels):
                subject_id = all_bip_subject_ids[ch_idx]
                chan_name = all_bip_names[ch_idx]
                
                if subject_id in langloc_map_bipolar:
                    langloc_chans = langloc_map_bipolar[subject_id]
                    if chan_name in langloc_chans:
                        is_langloc_bip[ch_idx] = True
            
            subject_info_out['is_langloc_bipolar'] = is_langloc_bip
        else:
            subject_info_out['is_langloc_bipolar'] = []
        
        # Report counts
        n_langloc = sum(is_langloc)
        n_langloc_bip = sum(subject_info_out['is_langloc_bipolar'])
        
        if self.verbose:
            print(f"Langloc channels identified: {n_langloc} unipolar, {n_langloc_bip} bipolar")
    
    def _print_summary(self, subject_info_out: Dict, has_langloc_data: bool):
        """Print comprehensive summary of pooled data."""
        print("\n=== Pooling Summary ===")
        print(f"Story: {subject_info_out['story_name']}")
        print(f"Epoch: [{subject_info_out['epoch'][0]:.1f}, {subject_info_out['epoch'][1]:.1f}] seconds")
        print(f"Subjects pooled: {len(subject_info_out['subjects'])}")
        print(f"Total unipolar channels: {subject_info_out['num_unipolar_channels']}")
        print(f"Total bipolar channels: {subject_info_out.get('num_bipolar_channels', 0)}")
        print(f"Significant unipolar channels: {sum(subject_info_out['is_significant'])}")
        print(f"Significant bipolar channels: {sum(subject_info_out.get('is_significant_bipolar', []))}")
        
        if has_langloc_data:
            print(f"Langloc unipolar channels: {sum(subject_info_out['is_langloc'])}")
            print(f"Langloc bipolar channels: {sum(subject_info_out.get('is_langloc_bipolar', []))}")
            
            # Overlap statistics
            sig_langloc_uni = sum(
                s and l for s, l in zip(subject_info_out['is_significant'], subject_info_out['is_langloc'])
            )
            print(f"Significant + Langloc unipolar: {sig_langloc_uni}")
            
            if subject_info_out.get('is_significant_bipolar') and subject_info_out.get('is_langloc_bipolar'):
                sig_langloc_bip = sum(
                    s and l for s, l in zip(subject_info_out['is_significant_bipolar'], subject_info_out['is_langloc_bipolar'])
                )
                print(f"Significant + Langloc bipolar: {sig_langloc_bip}")
        
        # Subject-wise breakdown
        print("\nSubject-wise channel breakdown:")
        print(f"{'Subject':<12} {'Uni':<8} {'Bip':<8} {'Uni_Sig':<8} {'Bip_Sig':<8} {'Uni_Lang':<8} {'Bip_Lang':<8}")
        print("-" * 80)
        
        for subject_id in subject_info_out['subjects']:
            uni_count = subject_info_out['subject_channel_count'].get(subject_id, 0)
            bip_count = subject_info_out.get('subject_bipolar_channel_count', {}).get(subject_id, 0)
            
            # Count significant channels for this subject
            uni_sig_count = sum(
                1 for sid, sig in zip(subject_info_out['subject_ids'], subject_info_out['is_significant'])
                if sid == subject_id and sig
            )
            
            bip_sig_count = 0
            if subject_info_out.get('subject_ids_bipolar') and subject_info_out.get('is_significant_bipolar'):
                bip_sig_count = sum(
                    1 for sid, sig in zip(subject_info_out['subject_ids_bipolar'], subject_info_out['is_significant_bipolar'])
                    if sid == subject_id and sig
                )
            
            # Count langloc channels for this subject
            uni_lang_count = 0
            bip_lang_count = 0
            if has_langloc_data:
                uni_lang_count = sum(
                    1 for sid, lang in zip(subject_info_out['subject_ids'], subject_info_out['is_langloc'])
                    if sid == subject_id and lang
                )
                
                if subject_info_out.get('subject_ids_bipolar') and subject_info_out.get('is_langloc_bipolar'):
                    bip_lang_count = sum(
                        1 for sid, lang in zip(subject_info_out['subject_ids_bipolar'], subject_info_out['is_langloc_bipolar'])
                        if sid == subject_id and lang
                    )
            
            print(f"{subject_id:<12} {uni_count:<8} {bip_count:<8} {uni_sig_count:<8} {bip_sig_count:<8} {uni_lang_count:<8} {bip_lang_count:<8}")


# Convenience function for direct usage
def load_story_data_for_nnmf(
    data_dir: Union[str, Path],
    story_name: str = None,
    epoch: List[float] = None,
    subject_list: List[str] = None,
    summary_stats_file: str = "",
    include_significant: bool = False,
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Convenience function to load story data formatted for NNMF pipeline.
    
    Returns:
        electrode_data: [n_electrodes × total_time] neural data
        story_lengths: [n_stories] length array (single story, so [total_time])
        subject_ids: [n_electrodes] subject ID per electrode
    """
    pooler = StoryDataPooler(verbose=verbose)
    
    # Override story name input if provided
    if story_name:
        original_input = input
        input = lambda prompt: story_name
    
    pooled_data, subject_info = pooler.pool_story_data_across_subjects(
        data_dir=data_dir,
        epoch=epoch,
        subject_list=subject_list,
        summary_stats_file=summary_stats_file,
        include_significant=include_significant
    )
    
    # Restore input function
    if story_name:
        input = original_input
    
    # Format for NNMF pipeline
    electrode_data = pooled_data['unipolar']  # [n_electrodes × time]
    story_lengths = np.array([electrode_data.shape[1]])  # Single story
    subject_ids = pooled_data['subject_ids']
    
    return electrode_data, story_lengths, subject_ids