r"""
=====================================================
Performance Estimators Based on Ground Truth Clusters
=====================================================

The **estimators** module provides a set of functions to estimate performance metrics such as pairwise precision and recall, cluster precision and recall, F-scores, and B-cubed metrics. The functions take as input a predicted disambiguation, a set of ground truth clusters, and a set of cluster sampling weights (e.g., inverse probability weights for each cluster). They return an estimate of the performance metric, along with an estimate of the standard deviation of the estimate.

Representative performance estimators are necessary for accurate evaluation of entity resolution algorithms due to the following reasons:

1. It is typically infeasible to manually label enough data to cover an entire population of interest.
2. Naively computing performance metrics on benchmark datasets leads to highly biased and over-optimistic results that are not representative of real-world performance. This is due to the non-linear scaling of entity resolution: while it might be easy to disambiguate a small benchmark dataset, the complexity of the problem grows quadratically in the dataset size.

Using small, weighted samples of ground truth data in conjunction with representative statistical estimators addresses these two issues, by providing an efficient and accurate way of evaluating the performance of entity resolution algorithms on large datasets while also taking into account the sampling processes and biases.

Note: In order to obtain representative performance estimators, the set of predicted clusters given as an argument to estimator functions should cover the entire population of interest. Typically, this set of predicted clusters will be much larger than the set of sampled clusters.
"""
from er_evaluation.estimators._estimators import (
    b_cubed_precision_design_estimate,
    b_cubed_recall_design_estimate,
    cluster_f_design_estimate,
    cluster_precision_design_estimate,
    cluster_recall_design_estimate,
    estimates_table,
    pairwise_f_design_estimate,
    pairwise_precision_design_estimate,
    pairwise_recall_design_estimate,
)

__all__ = [
    "b_cubed_precision_design_estimate",
    "b_cubed_recall_design_estimate",
    "cluster_f_design_estimate",
    "cluster_precision_design_estimate",
    "cluster_recall_design_estimate",
    "estimates_table",
    "pairwise_f_design_estimate",
    "pairwise_precision_design_estimate",
    "pairwise_recall_design_estimate",
]
