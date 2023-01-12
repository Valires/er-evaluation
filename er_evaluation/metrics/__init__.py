"""
====================================================
Evaluation Metrics (Precision, Recall, B-cubed, etc)
====================================================

The **metrics** module provides a set of functions to compute performance evaluation metrics for entity resolution algorithms. These metrics are computed on the intersection of a predicted disambiguation and a benchmark dataset. The available metrics are:

- **pairwise precision**: the proportion of correctly predicted links among all predicted links.
- **pairwise recall**: the proportion of correctly predicted links among all true links.
- **cluster precision**: the proportion of correctly predicted clusters among all predicted clusters.
- **cluster recall**: the proportion of correctly predicted clusters among all true clusters.
- **F-scores**: harmonic mean of precision and recall
- **B-cubed metrics**: These are defined as a weighted average of mention-level measures of precision and recall.

It is important to note that these metrics are not representative of real-world performance, they only describe the performance for disambiguating the given benchmark dataset. When using these metrics, it is important to use a benchmark dataset that is representative of the population of entities being studied, otherwise, the results may not generalize.
"""
from er_evaluation.metrics._metrics import (
    adjusted_rand_score,
    b_cubed_f,
    b_cubed_precision,
    b_cubed_recall,
    cluster_completeness,
    cluster_f,
    cluster_homogeneity,
    cluster_precision,
    cluster_recall,
    cluster_v_measure,
    metrics_table,
    pairwise_f,
    pairwise_precision,
    pairwise_recall,
    rand_score,
)

__all__ = [
    "adjusted_rand_score",
    "b_cubed_f",
    "b_cubed_precision",
    "b_cubed_recall",
    "cluster_completeness",
    "cluster_f",
    "cluster_homogeneity",
    "cluster_precision",
    "cluster_recall",
    "cluster_v_measure",
    "metrics_table",
    "pairwise_f",
    "pairwise_precision",
    "pairwise_recall",
    "rand_score",
]
