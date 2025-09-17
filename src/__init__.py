"""
NNMF Pipeline Package
A comprehensive Python implementation of the Norman-Haignere et al. (2022) 
NNMF analysis pipeline for intracranial electrode data.
"""

from .main_pipeline import NNMFPipeline, run_nnmf_analysis, load_analysis_results
from .data_loader import StoryDataPooler, load_story_data_for_nnmf
from .statistical_analysis import StatisticalAnalyzer
from .nnmf_solver import ConstrainedNNMFSolver, define_constraint_config
from .cross_validation import CrossValidator
from .model_fitting import ModelFitter
from .component_analysis import ComponentAnalyzer
from .baseline_comparison import BaselineComparator

__version__ = "1.0.0"
__author__ = "NNMF Pipeline Development Team"
__email__ = "your.email@domain.com"

__all__ = [
    # Main pipeline
    'NNMFPipeline',
    'run_nnmf_analysis',
    'load_analysis_results',
    
    # Data loading
    'StoryDataPooler', 
    'load_story_data_for_nnmf',
    
    # Core components
    'StatisticalAnalyzer',
    'ConstrainedNNMFSolver',
    'define_constraint_config',
    'CrossValidator',
    'ModelFitter', 
    'ComponentAnalyzer',
    'BaselineComparator'
]