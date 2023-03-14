from er_evaluation.error_analysis import (
    cluster_sizes_from_table,
    error_indicator_from_table,
    expected_missing_from_table,
    expected_relative_extra_from_table,
    expected_relative_missing_from_table,
    expected_size_difference_from_table,
)
from er_evaluation.estimators._utils import ratio_of_means_estimator


@ratio_of_means_estimator
def _pairwise_precision_estimator_from_table(error_table, weights):
    cs = cluster_sizes_from_table(error_table)
    E_miss = expected_missing_from_table(error_table)
    E_size = expected_size_difference_from_table(error_table)

    N = cs * (cs - 1 - E_miss) * weights
    D = cs * (cs - 1 + E_size) * weights

    return (N, D)


@ratio_of_means_estimator
def _pairwise_recall_estimator_from_table(error_table, weights):
    cs = cluster_sizes_from_table(error_table)
    E_miss = expected_missing_from_table(error_table)

    N = cs * (cs - 1 - E_miss) * weights
    D = cs * (cs - 1) * weights

    return (N, D)


@ratio_of_means_estimator
def _pairwise_f_estimator_from_table(error_table, weights, beta=1.0):
    cs = cluster_sizes_from_table(error_table)
    E_miss = expected_missing_from_table(error_table)
    E_size = expected_size_difference_from_table(error_table)
    weights = 1 / cs

    N = cs * (cs - 1 - E_miss) * weights
    D = cs * (cs - 1 + beta**2 * E_size / (1 + beta**2)) * weights

    return N, D


@ratio_of_means_estimator
def _cluster_precision_estimator_from_table(error_table, weights, len_prediction, nunique_prediction):
    cs = cluster_sizes_from_table(error_table)
    E_delta = 1 - error_indicator_from_table(error_table)

    N = len_prediction * E_delta * weights
    D = nunique_prediction * cs * weights

    return N, D


@ratio_of_means_estimator
def _cluster_recall_estimator_from_table(error_table, weights):
    E_delta = 1 - error_indicator_from_table(error_table)

    N = E_delta * weights
    D = weights

    return N, D


@ratio_of_means_estimator
def _cluster_f_estimator_from_table(error_table, weights, len_prediction, nunique_prediction, beta=1.0):
    cs = cluster_sizes_from_table(error_table)
    E_delta = 1 - error_indicator_from_table(error_table)

    multiplier = len_prediction * (1 + beta**2) / nunique_prediction

    N = multiplier * E_delta * weights
    D = beta**2 * len_prediction / nunique_prediction + cs * weights

    return N, D


@ratio_of_means_estimator
def _b_cubed_precision_estimator_from_table(error_table, weights):
    E_extra_rel = expected_relative_extra_from_table(error_table)

    N = (1 - E_extra_rel) * weights
    D = weights

    return N, D


@ratio_of_means_estimator
def _b_cubed_recall_estimator_from_table(error_table, weights):
    E_miss_rel = expected_relative_missing_from_table(error_table)

    N = (1 - E_miss_rel) * weights
    D = weights

    return N, D
