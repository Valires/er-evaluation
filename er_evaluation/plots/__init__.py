"""
Helper Plots and Visualizations
"""
from er_evaluation.plots._plots import (
    compare_plots,
    plot_cluster_errors,
    plot_cluster_sizes_distribution,
    plot_comparison,
    plot_entropy_curve,
    plot_estimates,
    plot_metrics,
    plot_summaries,
)

from er_evaluation.plots._fairness import plot_performance_disparities

__all__ = [
    "compare_plots",
    "plot_cluster_errors",
    "plot_cluster_sizes_distribution",
    "plot_comparison",
    "plot_entropy_curve",
    "plot_estimates",
    "plot_metrics",
    "plot_performance_disparities",
    "plot_summaries",
]
