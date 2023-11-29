"""
====================================================
Evaluation Metrics (Precision, Recall, B-cubed, etc)
====================================================

The **metrics** module provides a set of functions to compute common comparison metrics for entity resolution algorithms. These compare a **predicted** clustering to a **reference** clustering. The available metrics are:

- **pairwise precision**: the proportion of predicted links that also appear in the reference clustering.
- **pairwise recall**: the proportion of links in the reference clustering that also appear in the predicted clustering.
- **cluster precision**: the proportion of predicted clusters that are present in the reference clustering.
- **cluster recall**: the proportion of reference clusters that are also present in the predicted clustering.
- **F-scores**: harmonic mean of precision and recall
- **B-cubed metrics**: These are defined as a weighted average of mention-level measures of precision and recall.

**Notes:** 

- When computing these metrics, **all functions in this module first compute the inner join** of the reference and predicted clusterings. As such, only records that appear in both clusterings are accounted for by these metrics. All other records are dropped.
- Records with NA cluster identifier in the reference or predicted clusterings are dropped.
- The metrics in this module do not provide representative performance estimates. They are only useful for comparing two clusterings, such as a. For representative performance estimates, see the :mod:`er_evaluation.estimators` module.
"""
from er_evaluation.metrics._metrics import (adjusted_rand_score, b_cubed_f,
                                            b_cubed_precision, b_cubed_recall,
                                            cluster_completeness, cluster_f,
                                            cluster_homogeneity,
                                            cluster_precision, cluster_recall,
                                            cluster_v_measure, metrics_table,
                                            pairwise_f, pairwise_precision,
                                            pairwise_recall, rand_score)

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
