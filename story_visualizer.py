"""
Story Data Visualizer for NNMF Pipeline

Visualization functions for story data and NNMF results.
Integrates with existing nnmf_pipeline repository.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def plot_story_overview(data: Dict[str, Any]) -> None:
    """Plot overview of loaded story data."""
    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Electrode data heatmap
        ax1 = axes[0, 0]
        electrode_data = data['electrode_data']
        # Sample data for visualization if too large
        if electrode_data.shape[1] > 5000:
            sample_indices = np.linspace(0, electrode_data.shape[1]-1, 2000, dtype=int)
            plot_data = electrode_data[:, sample_indices]
        else:
            plot_data = electrode_data
            
        im1 = ax1.imshow(plot_data, aspect='auto', cmap='viridis')
        ax1.set_title(f'Electrode Data: {data.get("story_name", "Unknown")}')
        ax1.set_xlabel('Time (samples)')
        ax1.set_ylabel('Electrodes')
        plt.colorbar(im1, ax=ax1)
        
        # Plot 2: Subject distribution
        ax2 = axes[0, 1]
        subject_counts = pd.Series(data['subject_ids']).value_counts()
        bars = ax2.bar(range(len(subject_counts)), subject_counts.values, color='skyblue')
        ax2.set_title('Electrodes per Subject')
        ax2.set_xlabel('Subject')
        ax2.set_ylabel('Number of Electrodes')
        ax2.set_xticks(range(len(subject_counts)))
        ax2.set_xticklabels(subject_counts.index, rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, subject_counts.values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    str(value), ha='center', va='bottom')
        
        # Plot 3: Electrode categories
        ax3 = axes[1, 0]
        categories = []
        counts = []
        
        categories.append('All')
        counts.append(len(data.get('subject_ids', [])))
        
        if data.get('is_significant') is not None:
            categories.append('Significant')
            counts.append(np.sum(data['is_significant']))
            
        if data.get('is_langloc') is not None:
            categories.append('Language-localized')
            counts.append(np.sum(data['is_langloc']))
            
        if data.get('is_significant') is not None and data.get('is_langloc') is not None:
            categories.append('Sig & LangLoc')
            counts.append(np.sum(data['is_significant'] & data['is_langloc']))\n        \n        colors = ['skyblue', 'orange', 'green', 'red'][:len(categories)]\n        bars = ax3.bar(categories, counts, color=colors)\n        ax3.set_title('Electrode Categories')\n        ax3.set_ylabel('Number of Electrodes')\n        plt.setp(ax3.get_xticklabels(), rotation=45)\n        \n        # Add value labels\n        for bar, value in zip(bars, counts):\n            if value > 0:\n                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01, \n                        str(value), ha='center', va='bottom')\n        \n        # Plot 4: Sample time courses\n        ax4 = axes[1, 1]\n        n_samples = min(5, electrode_data.shape[0])\n        sample_indices = np.random.choice(electrode_data.shape[0], n_samples, replace=False)\n        time_indices = slice(0, min(2000, electrode_data.shape[1]))\n        \n        for i, electrode_idx in enumerate(sample_indices):\n            ax4.plot(electrode_data[electrode_idx, time_indices], \n                    alpha=0.7, label=f'Electrode {electrode_idx}')\n        \n        ax4.set_title('Sample Time Courses')\n        ax4.set_xlabel('Time (samples)')\n        ax4.set_ylabel('Signal Amplitude')\n        ax4.legend()\n        ax4.grid(True, alpha=0.3)\n        \n        plt.tight_layout()\n        plt.show()\n        \n    except Exception as e:\n        logger.error(f\"Error plotting story overview: {e}\")\n\n\ndef plot_nnmf_results(results: Dict[str, Any]) -> None:\n    \"\"\"Plot comprehensive NNMF analysis results.\"\"\"\n    try:\n        fig, axes = plt.subplots(2, 3, figsize=(18, 12))\n        \n        # Plot 1: Cross-validation composite score\n        ax1 = axes[0, 0]\n        cv = results.get('cross_validation', {})\n        if 'composite_score' in cv:\n            im1 = ax1.imshow(cv['composite_score'], aspect='auto', cmap='viridis')\n            ax1.set_title('CV Composite Score\\n(Lower = Better)')\n            ax1.set_xlabel('Alpha Index')\n            ax1.set_ylabel('Component Index')\n            plt.colorbar(im1, ax=ax1)\n            \n            # Mark optimal point\n            if 'optimal_components' in cv and 'optimal_alpha' in cv:\n                try:\n                    opt_k_idx = cv['component_range'].index(cv['optimal_components'])\n                    opt_a_idx = cv['alpha_range'].index(cv['optimal_alpha'])\n                    ax1.scatter(opt_a_idx, opt_k_idx, color='red', s=100, \n                              marker='*', label='Optimal', zorder=5)\n                    ax1.legend()\n                except (ValueError, KeyError):\n                    pass\n        \n        # Plot 2: Spatial components\n        ax2 = axes[0, 1]\n        final_model = results.get('final_model', {})\n        if 'W' in final_model:\n            W = final_model['W']\n            im2 = ax2.imshow(W.T, aspect='auto', cmap='RdBu_r')\n            ax2.set_title(f'Spatial Components (W)\\nk={W.shape[1]}')\n            ax2.set_xlabel('Electrodes')\n            ax2.set_ylabel('Components')\n            plt.colorbar(im2, ax=ax2)\n        \n        # Plot 3: Temporal components\n        ax3 = axes[0, 2]\n        if 'H' in final_model:\n            H = final_model['H']\n            n_plot = min(5, H.shape[0])\n            colors = plt.cm.tab10(np.linspace(0, 1, n_plot))\n            \n            for i in range(n_plot):\n                # Sample temporal data if too long\n                if H.shape[1] > 2000:\n                    sample_indices = np.linspace(0, H.shape[1]-1, 2000, dtype=int)\n                    plot_data = H[i, sample_indices]\n                else:\n                    plot_data = H[i, :]\n                    \n                ax3.plot(plot_data, color=colors[i], alpha=0.8, \n                        label=f'Comp {i+1}', linewidth=1.5)\n            \n            ax3.set_title('Temporal Components (H)')\n            ax3.set_xlabel('Time')\n            ax3.set_ylabel('Activation')\n            ax3.legend()\n            ax3.grid(True, alpha=0.3)\n        \n        # Plot 4: Reconstruction error vs components\n        ax4 = axes[1, 0]\n        if 'component_range' in cv and 'mean_recon_errors' in cv:\n            comp_range = cv['component_range']\n            try:\n                opt_alpha_idx = cv['alpha_range'].index(cv['optimal_alpha'])\n                recon_errors = cv['mean_recon_errors'][:, opt_alpha_idx]\n                \n                ax4.plot(comp_range, recon_errors, 'o-', linewidth=2, markersize=6)\n                ax4.axvline(cv['optimal_components'], color='r', linestyle='--', \n                           linewidth=2, label=f'Optimal k={cv[\"optimal_components\"]}')\n                ax4.set_title('Reconstruction Error vs Components')\n                ax4.set_xlabel('Number of Components')\n                ax4.set_ylabel('Mean CV Error')\n                ax4.legend()\n                ax4.grid(True, alpha=0.3)\n            except (KeyError, ValueError, IndexError):\n                ax4.text(0.5, 0.5, 'CV data not available', ha='center', va='center')\n                ax4.set_title('Reconstruction Error vs Components')\n        \n        # Plot 5: Reliability vs components\n        ax5 = axes[1, 1]\n        if 'mean_reliability' in cv and 'component_range' in cv:\n            try:\n                reliability = cv['mean_reliability'][:, opt_alpha_idx]\n                ax5.plot(comp_range, reliability, 'o-', color='green', \n                        linewidth=2, markersize=6)\n                ax5.axvline(cv['optimal_components'], color='r', linestyle='--', \n                           linewidth=2, label=f'Optimal k={cv[\"optimal_components\"]}')\n                ax5.set_title('Reliability vs Components')\n                ax5.set_xlabel('Number of Components')\n                ax5.set_ylabel('Mean CV Reliability')\n                ax5.legend()\n                ax5.grid(True, alpha=0.3)\n            except (KeyError, ValueError, IndexError):\n                ax5.text(0.5, 0.5, 'Reliability data not available', ha='center', va='center')\n                ax5.set_title('Reliability vs Components')\n        \n        # Plot 6: Component activations heatmap\n        ax6 = axes[1, 2]\n        if 'H' in final_model:\n            H = final_model['H']\n            # Sample data if too large\n            if H.shape[1] > 2000:\n                sample_indices = np.linspace(0, H.shape[1]-1, 1000, dtype=int)\n                plot_data = H[:, sample_indices]\n            else:\n                plot_data = H\n                \n            im6 = ax6.imshow(plot_data, aspect='auto', cmap='viridis')\n            ax6.set_title('Component Activations')\n            ax6.set_xlabel('Time')\n            ax6.set_ylabel('Components')\n            plt.colorbar(im6, ax=ax6)\n        \n        # Add main title\n        metadata = results.get('metadata', {})\n        story_name = metadata.get('story_name', 'Unknown')\n        n_components = final_model.get('n_components', 0)\n        variance_exp = final_model.get('variance_explained', 0)\n        \n        fig.suptitle(f'NNMF Results: {story_name} (k={n_components}, '\n                    f'var_exp={variance_exp:.1%})', fontsize=16)\n        \n        plt.tight_layout()\n        plt.show()\n        \n    except Exception as e:\n        logger.error(f\"Error plotting NNMF results: {e}\")\n\n\ndef plot_component_comparison(results: Dict[str, Any], \n                             components: Optional[list] = None) -> None:\n    \"\"\"Plot detailed comparison of specific components.\"\"\"\n    try:\n        final_model = results.get('final_model', {})\n        if 'W' not in final_model or 'H' not in final_model:\n            logger.warning(\"Final model data not available for component comparison\")\n            return\n            \n        W, H = final_model['W'], final_model['H']\n        n_components = W.shape[1]\n        \n        if components is None:\n            components = list(range(min(4, n_components)))\n        \n        fig, axes = plt.subplots(2, len(components), figsize=(4*len(components), 8))\n        if len(components) == 1:\n            axes = axes.reshape(2, 1)\n        \n        for i, comp_idx in enumerate(components):\n            if comp_idx >= n_components:\n                continue\n                \n            # Spatial pattern\n            ax_spatial = axes[0, i]\n            ax_spatial.bar(range(len(W[:, comp_idx])), W[:, comp_idx])\n            ax_spatial.set_title(f'Component {comp_idx+1}\\nSpatial Pattern')\n            ax_spatial.set_xlabel('Electrodes')\n            ax_spatial.set_ylabel('Weight')\n            \n            # Temporal pattern\n            ax_temporal = axes[1, i]\n            # Sample temporal data if too long\n            if H.shape[1] > 2000:\n                sample_indices = np.linspace(0, H.shape[1]-1, 2000, dtype=int)\n                plot_data = H[comp_idx, sample_indices]\n            else:\n                plot_data = H[comp_idx, :]\n                \n            ax_temporal.plot(plot_data, linewidth=1.5)\n            ax_temporal.set_title(f'Component {comp_idx+1}\\nTemporal Pattern')\n            ax_temporal.set_xlabel('Time')\n            ax_temporal.set_ylabel('Activation')\n            ax_temporal.grid(True, alpha=0.3)\n        \n        plt.tight_layout()\n        plt.show()\n        \n    except Exception as e:\n        logger.error(f\"Error plotting component comparison: {e}\")