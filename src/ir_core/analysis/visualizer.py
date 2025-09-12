# src/ir_core/analysis/visualizer.py

"""
Framework-agnostic visualization utilities for analysis results.

This module provides visualization capabilities that work with any plotting backend
(matplotlib, seaborn, plotly) and can generate various types of charts for
analysis insights.
"""

from typing import Dict, List, Any, Optional, Union, Tuple
import matplotlib.pyplot as plt
import matplotlib.figure as mfigure
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from pathlib import Path
import warnings

from .core import AnalysisResult


class AnalysisVisualizer:
    """
    Framework-agnostic visualizer for analysis results.

    Supports multiple plotting backends and generates comprehensive
    visualizations for retrieval performance analysis.
    """

    def __init__(self, backend: str = 'matplotlib', style: str = 'default'):
        """
        Initialize the visualizer.

        Args:
            backend: Plotting backend ('matplotlib', 'seaborn', 'plotly')
            style: Visual style/theme to use
        """
        self.backend = backend
        self.style = style

        # Set up matplotlib/seaborn defaults
        if backend in ['matplotlib', 'seaborn']:
            plt.style.use(style)
            valid_styles = ['white', 'dark', 'whitegrid', 'darkgrid', 'ticks']
            sns_style = style if style in valid_styles else 'whitegrid'
            sns.set_style(sns_style)  # type: ignore

        # Create output directory
        self.output_dir = Path("outputs/visualizations")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def create_performance_distribution_histogram(
        self,
        results: List[AnalysisResult],
        save_path: Optional[str] = None
    ) -> Optional[Union[mfigure.Figure, go.Figure]]:
        """
        Create histogram of performance distribution (AP scores).

        Args:
            results: List of analysis results
            save_path: Optional path to save the plot

        Returns:
            Plot figure object
        """
        # Extract AP scores from all results
        ap_scores = []
        for result in results:
            for retrieval_result in result.retrieval_results:
                ap_scores.append(retrieval_result.ap_score)

        if not ap_scores:
            warnings.warn("No AP scores found in results")
            return None

        if self.backend == 'plotly':
            fig = px.histogram(
                x=ap_scores,
                nbins=20,
                title="Retrieval Performance Distribution",
                labels={'x': 'Average Precision (AP)', 'y': 'Frequency'},
                template='plotly_white'
            )
            fig.update_layout(showlegend=False)
        else:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(ap_scores, bins=20, alpha=0.7, edgecolor='black')
            ax.set_xlabel('Average Precision (AP)')
            ax.set_ylabel('Frequency')
            ax.set_title('Retrieval Performance Distribution')
            ax.grid(True, alpha=0.3)

        if save_path:
            self._save_figure(fig, save_path)

        return fig

    def create_query_length_performance_scatter(
        self,
        results: List[AnalysisResult],
        save_path: Optional[str] = None
    ) -> Optional[Union[mfigure.Figure, go.Figure]]:
        """
        Create scatter plot of query length vs performance.

        Args:
            results: List of analysis results
            save_path: Optional path to save the plot

        Returns:
            Plot figure object
        """
        query_lengths = []
        performances = []

        for result in results:
            for qa, rr in zip(result.query_analyses, result.retrieval_results):
                query_lengths.append(qa.query_length)
                performances.append(rr.ap_score)

        if not query_lengths:
            warnings.warn("No query analysis data found")
            return None

        data = pd.DataFrame({
            'query_length': query_lengths,
            'performance': performances
        })

        if self.backend == 'plotly':
            fig = px.scatter(
                data,
                x='query_length',
                y='performance',
                title="Query Length vs Performance",
                labels={
                    'query_length': 'Query Length (characters)',
                    'performance': 'Average Precision (AP)'
                },
                template='plotly_white',
                trendline="ols"
            )
        else:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(data=data, x='query_length', y='performance', ax=ax, alpha=0.6)
            sns.regplot(data=data, x='query_length', y='performance', ax=ax,
                       scatter=False, color='red', line_kws={'alpha': 0.7})
            ax.set_xlabel('Query Length (characters)')
            ax.set_ylabel('Average Precision (AP)')
            ax.set_title('Query Length vs Performance')
            ax.grid(True, alpha=0.3)

        if save_path:
            self._save_figure(fig, save_path)

        return fig

    def create_domain_performance_comparison(
        self,
        results: List[AnalysisResult],
        save_path: Optional[str] = None
    ) -> Optional[Union[mfigure.Figure, go.Figure]]:
        """
        Create bar chart comparing performance across domains.

        Args:
            results: List of analysis results
            save_path: Optional path to save the plot

        Returns:
            Plot figure object
        """
        domain_performance = {}

        for result in results:
            for qa, rr in zip(result.query_analyses, result.retrieval_results):
                for domain in qa.domain:
                    if domain not in domain_performance:
                        domain_performance[domain] = []
                    domain_performance[domain].append(rr.ap_score)

        if not domain_performance:
            warnings.warn("No domain performance data found")
            return None

        # Calculate mean performance per domain
        domain_means = {domain: np.mean(scores) for domain, scores in domain_performance.items()}
        domain_stds = {domain: np.std(scores) for domain, scores in domain_performance.items()}

        domains = list(domain_means.keys())
        means = list(domain_means.values())
        stds = list(domain_stds.values())

        if self.backend == 'plotly':
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=domains,
                y=means,
                error_y=dict(type='data', array=stds),
                name='Average Performance'
            ))
            fig.update_layout(
                title="Performance by Scientific Domain",
                xaxis_title="Domain",
                yaxis_title="Average Precision (AP)",
                template='plotly_white'
            )
        else:
            fig, ax = plt.subplots(figsize=(12, 6))
            bars = ax.bar(domains, means, yerr=stds, capsize=5, alpha=0.7)
            ax.set_xlabel('Scientific Domain')
            ax.set_ylabel('Average Precision (AP)')
            ax.set_title('Performance by Scientific Domain')
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45, ha='right')

        if save_path:
            self._save_figure(fig, save_path)

        return fig

    def create_error_pattern_heatmap(
        self,
        results: List[AnalysisResult],
        save_path: Optional[str] = None
    ) -> Optional[Union[mfigure.Figure, go.Figure]]:
        """
        Create heatmap of error patterns across domains and error types.

        Args:
            results: List of analysis results
            save_path: Optional path to save the plot

        Returns:
            Plot figure object
        """
        # Aggregate error patterns
        error_matrix = {}

        for result in results:
            for domain, error_rate in result.domain_error_rates.items():
                if domain not in error_matrix:
                    error_matrix[domain] = {}
                # Use a simplified error categorization
                error_type = 'total_errors'
                error_matrix[domain][error_type] = error_matrix[domain].get(error_type, 0) + error_rate

        if not error_matrix:
            warnings.warn("No error pattern data found")
            return None

        domains = list(error_matrix.keys())
        error_types = ['total_errors']  # Simplified for now

        heatmap_data = []
        for domain in domains:
            row = [error_matrix[domain].get('total_errors', 0)]
            heatmap_data.append(row)

        if self.backend == 'plotly':
            fig = go.Figure(data=go.Heatmap(
                z=heatmap_data,
                x=error_types,
                y=domains,
                colorscale='Reds',
                name='Error Rate'
            ))
            fig.update_layout(
                title="Error Patterns by Domain",
                xaxis_title="Error Type",
                yaxis_title="Domain",
                template='plotly_white'
            )
        else:
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(
                heatmap_data,
                xticklabels=error_types,
                yticklabels=domains,
                ax=ax,
                cmap='Reds',
                annot=True,
                fmt='.2f'
            )
            ax.set_title('Error Patterns by Domain')
            plt.xticks(rotation=45, ha='right')

        if save_path:
            self._save_figure(fig, save_path)

        return fig

    def create_time_series_performance_trends(
        self,
        results: List[AnalysisResult],
        timestamps: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ) -> Optional[Union[mfigure.Figure, go.Figure]]:
        """
        Create time series plot of performance trends.

        Args:
            results: List of analysis results
            timestamps: Optional timestamps for each result
            save_path: Optional path to save the plot

        Returns:
            Plot figure object
        """
        if timestamps is None:
            timestamps = [f"Run {i+1}" for i in range(len(results))]

        map_scores = [result.map_score for result in results]

        if self.backend == 'plotly':
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=map_scores,
                mode='lines+markers',
                name='MAP Score',
                line=dict(color='blue', width=2)
            ))
            fig.update_layout(
                title="Performance Trends Over Time",
                xaxis_title="Time/Run",
                yaxis_title="Mean Average Precision (MAP)",
                template='plotly_white'
            )
        else:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(timestamps, map_scores, marker='o', linestyle='-', linewidth=2, markersize=6)
            ax.set_xlabel('Time/Run')
            ax.set_ylabel('Mean Average Precision (MAP)')
            ax.set_title('Performance Trends Over Time')
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45, ha='right')

        if save_path:
            self._save_figure(fig, save_path)

        return fig

    def create_comprehensive_dashboard(
        self,
        results: List[AnalysisResult],
        save_path: Optional[str] = None
    ) -> Optional[Union[mfigure.Figure, go.Figure]]:
        """
        Create a comprehensive dashboard with multiple visualizations.

        Args:
            results: List of analysis results
            save_path: Optional path to save the plot

        Returns:
            Plot figure object
        """
        if self.backend == 'plotly':
            # Create subplot figure
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Performance Distribution',
                    'Query Length vs Performance',
                    'Domain Performance',
                    'Performance Trends'
                ),
                specs=[[{'type': 'histogram'}, {'type': 'scatter'}],
                       [{'type': 'bar'}, {'type': 'scatter'}]]
            )

            # Add performance distribution
            ap_scores = []
            for result in results:
                for rr in result.retrieval_results:
                    ap_scores.append(rr.ap_score)
            if ap_scores:
                fig.add_trace(
                    go.Histogram(x=ap_scores, name='Performance'),
                    row=1, col=1
                )

            # Add query length vs performance
            query_lengths = []
            performances = []
            for result in results:
                for qa, rr in zip(result.query_analyses, result.retrieval_results):
                    query_lengths.append(qa.query_length)
                    performances.append(rr.ap_score)
            if query_lengths:
                fig.add_trace(
                    go.Scatter(x=query_lengths, y=performances, mode='markers', name='Query vs Perf'),
                    row=1, col=2
                )

            # Add domain performance
            domain_performance = {}
            for result in results:
                for qa, rr in zip(result.query_analyses, result.retrieval_results):
                    for domain in qa.domain:
                        if domain not in domain_performance:
                            domain_performance[domain] = []
                        domain_performance[domain].append(rr.ap_score)
            if domain_performance:
                domains = list(domain_performance.keys())
                means = [np.mean(scores) for scores in domain_performance.values()]
                fig.add_trace(
                    go.Bar(x=domains, y=means, name='Domain Perf'),
                    row=2, col=1
                )

            # Add performance trends
            timestamps = [f"Run {i+1}" for i in range(len(results))]
            map_scores = [result.map_score for result in results]
            fig.add_trace(
                go.Scatter(x=timestamps, y=map_scores, mode='lines+markers', name='Trends'),
                row=2, col=2
            )

            fig.update_layout(
                title="Comprehensive Analysis Dashboard",
                template='plotly_white',
                showlegend=False
            )
        else:
            # For matplotlib, create a simple summary
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Comprehensive Analysis Dashboard', fontsize=16)

            # Performance distribution
            ap_scores = []
            for result in results:
                for rr in result.retrieval_results:
                    ap_scores.append(rr.ap_score)
            if ap_scores:
                ax1.hist(ap_scores, bins=20, alpha=0.7)
                ax1.set_title('Performance Distribution')
                ax1.set_xlabel('AP Score')
                ax1.set_ylabel('Frequency')

            # Query length vs performance
            query_lengths = []
            performances = []
            for result in results:
                for qa, rr in zip(result.query_analyses, result.retrieval_results):
                    query_lengths.append(qa.query_length)
                    performances.append(rr.ap_score)
            if query_lengths:
                ax2.scatter(query_lengths, performances, alpha=0.6)
                ax2.set_title('Query Length vs Performance')
                ax2.set_xlabel('Query Length')
                ax2.set_ylabel('AP Score')

            # Domain performance
            domain_performance = {}
            for result in results:
                for qa, rr in zip(result.query_analyses, result.retrieval_results):
                    for domain in qa.domain:
                        if domain not in domain_performance:
                            domain_performance[domain] = []
                        domain_performance[domain].append(rr.ap_score)
            if domain_performance:
                domains = list(domain_performance.keys())
                means = [np.mean(scores) for scores in domain_performance.values()]
                ax3.bar(domains, means, alpha=0.7)
                ax3.set_title('Domain Performance')
                ax3.set_xlabel('Domain')
                ax3.set_ylabel('Avg AP Score')
                ax3.tick_params(axis='x', rotation=45)

            # Performance trends
            timestamps = [f"Run {i+1}" for i in range(len(results))]
            map_scores = [result.map_score for result in results]
            ax4.plot(timestamps, map_scores, marker='o')
            ax4.set_title('Performance Trends')
            ax4.set_xlabel('Run')
            ax4.set_ylabel('MAP Score')
            ax4.tick_params(axis='x', rotation=45)

            plt.tight_layout()

        if save_path:
            self._save_figure(fig, save_path)

        return fig

    def _save_figure(self, fig: Union[mfigure.Figure, go.Figure], save_path: str):
        """Save figure to file."""
        full_path = self.output_dir / save_path
        full_path.parent.mkdir(parents=True, exist_ok=True)

        if self.backend == 'plotly':
            # Save HTML version (always works)
            fig.write_html(str(full_path.with_suffix('.html')))
            # Try to save PNG version (requires kaleido)
            try:
                fig.write_image(str(full_path.with_suffix('.png')))
            except (ImportError, ValueError) as e:
                print(f"Warning: Could not save PNG image for {save_path}. Install kaleido for PNG export: pip install kaleido")
                print(f"HTML version saved at: {full_path.with_suffix('.html')}")
        else:
            fig.savefig(str(full_path.with_suffix('.png')), dpi=300, bbox_inches='tight')
            plt.close(fig)


# Convenience functions for quick plotting
def plot_performance_distribution(results: List[AnalysisResult], backend: str = 'matplotlib'):
    """Quick function to plot performance distribution."""
    viz = AnalysisVisualizer(backend=backend)
    return viz.create_performance_distribution_histogram(results)


def plot_query_performance_correlation(results: List[AnalysisResult], backend: str = 'matplotlib'):
    """Quick function to plot query length vs performance."""
    viz = AnalysisVisualizer(backend=backend)
    return viz.create_query_length_performance_scatter(results)


def plot_domain_comparison(results: List[AnalysisResult], backend: str = 'matplotlib'):
    """Quick function to plot domain performance comparison."""
    viz = AnalysisVisualizer(backend=backend)
    return viz.create_domain_performance_comparison(results)
