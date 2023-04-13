"""
Helper Plots and Visualizations
"""
from er_evaluation.plots._fairness import plot_performance_disparities
from er_evaluation.plots._plots import (
    add_ests_to_summaries,
    compare_plots,
    plot_cluster_errors,
    plot_cluster_sizes_distribution,
    plot_comparison,
    plot_entropy_curve,
    plot_estimates,
    plot_metrics,
    plot_summaries,
)
from er_evaluation.plots._dtree_plots import (
    plot_dt_regressor_sunburst,
    plot_dt_regressor_treemap,
    make_dt_regressor_plot,
    plot_dt_regressor_tree,
)

__all__ = [
    "add_ests_to_summaries",
    "compare_plots",
    "plot_cluster_errors",
    "plot_cluster_sizes_distribution",
    "plot_comparison",
    "make_dt_regressor_plot",
    "plot_dt_regressor_sunburst",
    "plot_dt_regressor_tree",
    "plot_dt_regressor_treemap",
    "plot_entropy_curve",
    "plot_estimates",
    "plot_metrics",
    "plot_performance_disparities",
    "plot_summaries",
]
