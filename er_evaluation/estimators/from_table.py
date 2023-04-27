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
def pairwise_precision_estimator_from_table(error_table, weights):
    """
    Pairwise precision estimator from error table.

    Given an error table and weights (obtained from :py:func:`er_evaluation.record_error_table`), this function returns a pairwise precision estimate together with its estimated standard deviation.

    Args:
        error_table (DataFrame): The record error table obtained from :py:func:`er_evaluation.record_error_table`.
        weights (Series): A pandas Series representing the weights of different clusters.

    Returns:
        tuple: Precision estimate and standard deviation estimate.
    """
    cs = cluster_sizes_from_table(error_table)
    E_miss = expected_missing_from_table(error_table)
    E_size = expected_size_difference_from_table(error_table)

    N = cs * (cs - 1 - E_miss) * weights
    D = cs * (cs - 1 + E_size) * weights

    return (N, D)


@ratio_of_means_estimator
def pairwise_recall_estimator_from_table(error_table, weights):
    """
    Pairwise recall estimator from error table.

    Given an error table and weights (obtained from :py:func:`er_evaluation.record_error_table`), this function returns a pairwise recall estimate together with its estimated standard deviation.

    Args:
        error_table (DataFrame): The record error table obtained from :py:func:`er_evaluation.record_error_table`.
        weights (Series): A pandas Series representing the weights of different clusters.

    Returns:
        tuple: Recall estimate and standard deviation estimate.
    """
    cs = cluster_sizes_from_table(error_table)
    E_miss = expected_missing_from_table(error_table)

    N = cs * (cs - 1 - E_miss) * weights
    D = cs * (cs - 1) * weights

    return (N, D)


@ratio_of_means_estimator
def pairwise_f_estimator_from_table(error_table, weights, beta=1.0):
    """
    Pairwise f estimator from error table.

    Given an error table and weights (obtained from :py:func:`er_evaluation.record_error_table`), this function returns a pairwise f estimate together with its estimated standard deviation.

    Args:
        error_table (DataFrame): The record error table obtained from :py:func:`er_evaluation.record_error_table`.
        weights (Series): A pandas Series representing the weights of different clusters.
        beta (float, optional): The beta parameter. Defaults to 1.0.

    Returns:
        tuple: Recall estimate and standard deviation estimate.
    """
    cs = cluster_sizes_from_table(error_table)
    E_miss = expected_missing_from_table(error_table)
    E_size = expected_size_difference_from_table(error_table)

    N = cs * (cs - 1 - E_miss) * weights
    D = cs * (cs - 1 + beta**2 * E_size / (1 + beta**2)) * weights

    return N, D


@ratio_of_means_estimator
def cluster_precision_estimator_from_table(error_table, weights, len_prediction, nunique_prediction):
    """
    Cluster precision estimator from error table.

    Given an error table and weights (obtained from :py:func:`er_evaluation.record_error_table`), this function returns a cluster precision estimate together with its estimated standard deviation.

    Args:
        error_table (DataFrame): The record error table obtained from :py:func:`er_evaluation.record_error_table`.
        weights (Series): A pandas Series representing the weights of different clusters.

    Returns:
        tuple: Cluster precision estimate and standard deviation estimate.
    """
    cs = cluster_sizes_from_table(error_table)
    E_delta = 1 - error_indicator_from_table(error_table)

    N = len_prediction * E_delta * weights
    D = nunique_prediction * cs * weights

    return N, D


@ratio_of_means_estimator
def cluster_recall_estimator_from_table(error_table, weights):
    """
    Cluster recall estimator from error table.

    Given an error table and weights (obtained from :py:func:`er_evaluation.record_error_table`), this function returns a cluster recall estimate together with its estimated standard deviation.

    Args:
        error_table (DataFrame): The record error table obtained from :py:func:`er_evaluation.record_error_table`.
        weights (Series): A pandas Series representing the weights of different clusters.

    Returns:
        tuple: Cluster recall estimate and standard deviation estimate.
    """
    E_delta = 1 - error_indicator_from_table(error_table)

    N = E_delta * weights
    D = weights

    return N, D


@ratio_of_means_estimator
def cluster_f_estimator_from_table(error_table, weights, len_prediction, nunique_prediction, beta=1.0):
    """
    Cluster f estimator from error table.

    Given an error table and weights (obtained from :py:func:`er_evaluation.record_error_table`), this function returns a cluster f estimate together with its estimated standard deviation.

    Args:
        error_table (DataFrame): The record error table obtained from :py:func:`er_evaluation.record_error_table`.
        weights (Series): A pandas Series representing the weights of different clusters.

    Returns:
        tuple: Cluster f estimate and standard deviation estimate.
    """
    cs = cluster_sizes_from_table(error_table)
    E_delta = 1 - error_indicator_from_table(error_table)

    multiplier = len_prediction * (1 + beta**2) / nunique_prediction

    N = multiplier * E_delta * weights
    D = (beta**2 * len_prediction / nunique_prediction + cs) * weights

    return N, D


@ratio_of_means_estimator
def b_cubed_precision_estimator_from_table(error_table, weights):
    """
    B-cubed precision estimator from error table.

    Given an error table and weights (obtained from :py:func:`er_evaluation.record_error_table`), this function returns a B-cubed precision estimate together with its estimated standard deviation.

    Args:
        error_table (DataFrame): The record error table obtained from :py:func:`er_evaluation.record_error_table`.
        weights (Series): A pandas Series representing the weights of different clusters.

    Returns:
        tuple: B-cubed precision estimate and standard deviation estimate.
    """
    E_extra_rel = expected_relative_extra_from_table(error_table)

    N = (1 - E_extra_rel) * weights
    D = weights

    return N, D


@ratio_of_means_estimator
def b_cubed_recall_estimator_from_table(error_table, weights):
    """
    B-cubed recall estimator from error table.

    Given an error table and weights (obtained from :py:func:`er_evaluation.record_error_table`), this function returns a B-cubed recall estimate together with its estimated standard deviation.

    Args:
        error_table (DataFrame): The record error table obtained from :py:func:`er_evaluation.record_error_table`.
        weights (Series): A pandas Series representing the weights of different clusters.

    Returns:
        tuple: B-cubed recall estimate and standard deviation estimate.
    """
    E_miss_rel = expected_relative_missing_from_table(error_table)

    N = (1 - E_miss_rel) * weights
    D = weights

    return N, D
