import numpy as np
import pandas as pd
from scipy.special import comb

from er_evaluation.summary import cluster_sizes
from er_evaluation.data_structures import MembershipVector
from er_evaluation.error_analysis import (
    cluster_sizes_from_table,
    error_indicator_from_table,
    error_indicator,
    expected_missing_from_table,
    expected_relative_extra_from_table,
    expected_relative_missing_from_table,
    expected_size_difference_from_table,
    record_error_table,
)
from er_evaluation.estimators._utils import validate_prediction_sample, _parse_weights, validate_weights, ratio__of_means_estimator
from er_evaluation.utils import expand_grid


def _prepare_args(prediction, sample, weights):
    validate_prediction_sample(prediction, sample)
    sample = sample[sample.index.isin(prediction.index)]

    weights = _parse_weights(sample, weights)
    validate_weights(sample, weights)
    weights = weights[weights.index.isin(sample.values)]

    return prediction, sample, weights


def estimates_table(predictions, samples_weights, estimators):
    """
    Create table of estimates applied to all combinations of predictions and (sample, weights) pairs.

    Args:
        predictions (Dict): Dictionary of membership vectors.
        samples_weights (Dict): Dictionary of dictionaries of the form {"sample": sample, "weights": weights}, where `sample` is the sample membership vector and `weights` is the pandas Series of sampling weights. See estimators definitions for more information.
        estimators (Dict): Dictionary of estimator functions. Each estimator is expected to return a pair (estimate, std).

    Returns:
        DataFrame: Pandas DataFrame with columns "predition", "sample_weights", "estimator", "value", and "std", where value and std are the point estimate and standard deviation estimate for the estimator applied to the given prediction, sample and sampling weights.

    Examples:
        >>> import pandas as pd
        >>> from er_evaluation.estimators import *
        >>> predictions = {"prediction_1": pd.Series(index=[1,2,3,4,5,6,7,8], data=[1,1,2,3,2,4,4,4])}
        >>> samples_weights = {"sample_1": {"sample": pd.Series(index=[1,2,3,4,5,8], data=["c1", "c1", "c1", "c2", "c2", "c4"]), "weights": pd.Series(1, index=["c1", "c2", "c4"])}}
        >>> estimators = {"precision": pairwise_precision_estimator, "recall": pairwise_recall_estimator}
        >>> estimates_table(predictions, samples_weights, estimators) # doctest: +NORMALIZE_WHITESPACE
            prediction	    sample_weights	estimator	value	    std
        0	prediction_1	sample_1	    precision	0.388889	0.254588
        1	prediction_1	sample_1	    recall	    0.296875	0.108253
    """
    params = expand_grid(
        prediction=predictions,
        sample_weights=samples_weights,
        estimator=estimators,
    )

    def lambd(pred_key, ref_key, est_key):
        ests = estimators[est_key](
            predictions[pred_key],
            samples_weights[ref_key]["sample"],
            samples_weights[ref_key]["weights"],
        )

        return ests

    params[["value", "std"]] = pd.DataFrame(
        params.apply(
            lambda x: lambd(x["prediction"], x["sample_weights"], x["estimator"]),
            axis=1,
        ).tolist(),
        index=params.index,
    )

    return params


@ratio__of_means_estimator
def pairwise_precision_estimator(prediction, sample, weights):
    r"""
    Design estimator for pairwise precision.

    Given a predicted disambiguation `prediction`, a set of ground truth clusters `sample`, and a set of cluster sampling weights `weights` (e.g., inverse probability weights for each cluster), this returns a pairwise precision estimate together with its estimated standard deviation.

    Note:
        This is the precision estimator corresponding to cluster block sampling in [1].

    Args:
        prediction (Series): Membership vector indexed by cluster elements and with values corresponding to associated cluster identifier.
        sample (Series): Membership vector indexed by cluster elements and with values corresponding to associated cluster identifier.
        weights (Series): Pandas Series indexed by cluster identifier and with values corresponding to cluster sampling weights (e.g., inverse sampling probabilities). Can also be the string "uniform" for uniform sampling weights, or "cluster_size" for inverse cluster size sampling weights.

    Returns:
        tuple: Precision estimate and standard deviation estimate.

    Examples:
        >>> prediction = pd.Series(index=[1,2,3,4,5,6,7,8], data=[1,1,2,3,2,4,4,4])
        >>> sample = pd.Series(index=[1,2,3,4,5,8], data=["c1", "c1", "c1", "c2", "c2", "c4"])
        >>> weights = pd.Series(1, index=sample.unique()) # Uniform cluster weights
        >>> pairwise_precision_estimator(prediction, sample, weights)
        (0.3888888888888889, 0.2545875386086578)

    References:
        [1] Binette, Olivier, Sokhna A York, Emma Hickerson, Youngsoo Baek, Sarvo Madhavan, Christina Jones. (2022). Estimating the Performance of Entity Resolution Algorithms: Lessons Learned Through PatentsView.org. arXiv e-prints: arxiv:2210.01230
    """
    prediction = MembershipVector(prediction, dropna=True)
    sample = MembershipVector(sample, dropna=True)

    prediction, sample, weights = _prepare_args(prediction, sample, weights)

    inner = pd.concat(
        {"prediction": prediction, "reference": sample},
        axis=1,
        join="inner",
        copy=False,
    )
    split_cluster_sizes = inner.groupby(["prediction", "reference"]).size()
    # Number of correctly predicted links (TP) by reference cluster.
    TP_by_reference = (
        split_cluster_sizes.to_frame()
        .assign(cmb=comb(split_cluster_sizes.values, 2))
        .groupby("reference")
        .sum()
        .cmb.sort_index()
        .values
    )

    N = TP_by_reference
    K = prediction.isin(inner.prediction)

    def lambd(x):
        I = inner.prediction.index.isin(x.index)
        J = prediction[K].isin(inner.prediction[I])
        A = inner.prediction[I].value_counts(sort=False).sort_index().values
        B = prediction[K][J].value_counts(sort=False).sort_index().values
        return np.sum(A * (B - A))

    # Number of falsely predicted links (FP) by reference cluster.
    FP_by_reference = inner.groupby("reference").apply(lambd)
    D = TP_by_reference + 0.5 * FP_by_reference

    sorted_weights = weights.sort_index()
    N, D = (N * sorted_weights, D * sorted_weights)

    return N, D


@ratio__of_means_estimator
def pairwise_recall_estimator(prediction, sample, weights):
    r"""
    Design estimator for pairwise recall.

    Given a predicted disambiguation `prediction`, a set of ground truth clusters `sample`, and a set of cluster sampling weights `weights` (e.g., inverse probability weights for each cluster), this returns a pairwise recall estimate together with its estimated standard deviation.

    Note:
        This is the recall estimator corresponding to cluster block sampling in [1].

    Args:
        prediction (Series): Membership vector indexed by cluster elements and with values corresponding to associated cluster identifier.
        sample (Series): Membership vector indexed by cluster elements and with values corresponding to associated cluster identifier.
        weights (Series): Pandas Series indexed by cluster identifier and with values corresponding to cluster sampling weights (e.g., inverse sampling probabilities). Can also be the string "uniform" for uniform sampling weights, or "cluster_size" for inverse cluster size sampling weights.

    Returns:
        tuple: Recall estimate and standard deviation estimate.

    Examples:
        >>> prediction = pd.Series(index=[1,2,3,4,5,6,7,8], data=[1,1,2,3,2,4,4,4])
        >>> sample = pd.Series(index=[1,2,3,4,5,8], data=["c1", "c1", "c1", "c2", "c2", "c4"])
        >>> weights = pd.Series(1, index=sample.unique()) # Uniform cluster weights
        >>> pairwise_recall_estimator(prediction, sample, weights)
        (0.296875, 0.10825317547305482)

    References:
        [1] Binette, Olivier, Sokhna A York, Emma Hickerson, Youngsoo Baek, Sarvo Madhavan, Christina Jones. (2022). Estimating the Performance of Entity Resolution Algorithms: Lessons Learned Through PatentsView.org. arXiv e-prints: arxiv:2210.01230
    """
    prediction = MembershipVector(prediction, dropna=True)
    sample = MembershipVector(sample, dropna=True)

    prediction, sample, weights = _prepare_args(prediction, sample, weights)

    inner = pd.concat(
        {"prediction": prediction, "reference": sample},
        axis=1,
        join="inner",
        copy=False,
    )
    split_cluster_sizes = inner.groupby(["prediction", "reference"]).size()
    TP_by_reference = (
        split_cluster_sizes.to_frame()
        .assign(cmb=comb(split_cluster_sizes.values, 2))
        .groupby("reference")
        .sum()
        .cmb.sort_index()
        .values
    )
    cluster_sizes = inner.reference.value_counts(sort=False).sort_index().values

    N = TP_by_reference
    D = comb(cluster_sizes, 2)

    sorted_weights = weights.sort_index()
    N, D = (N * sorted_weights, D * sorted_weights)

    return N, D


@ratio__of_means_estimator
def pairwise_f_estimator(prediction, sample, weights, beta=1.0):
    """
    Design estimator for pairwise F-score.

    Given a predicted disambiguation `prediction`, a set of ground truth clusters `sample`, and a set of cluster sampling weights `weights` (e.g., inverse probability weights for each cluster), this returns a pairwise F-score estimate together with its estimated standard deviation.

    Args:
        prediction (Series): Membership vector indexed by cluster elements and with values corresponding to associated cluster identifier.
        sample (Series): Membership vector indexed by cluster elements and with values corresponding to associated cluster identifier.
        weights (Series): Pandas Series indexed by cluster identifier and with values corresponding to cluster sampling weights (e.g., inverse sampling probabilities). Can also be the string "uniform" for uniform sampling weights, or "cluster_size" for inverse cluster size sampling weights.
        beta (float): Weighting parameter for F-score. Default is 1.0.

    Returns:
        tuple: F-score estimate and standard deviation estimate.

    Examples:
        >>> prediction = pd.Series(index=[1,2,3,4,5,6,7,8], data=[1,1,2,3,2,4,4,4])
        >>> sample = pd.Series(index=[1,2,3,4,5], data=["c1", "c1", "c1", "c2", "c2"])
        >>> weights = pd.Series(1, index=sample.unique()) # Uniform cluster weights
        >>> pairwise_f_estimator(prediction, sample, weights)
        (0.36213991769547327, 0.19753086419753088)
    """
    prediction = MembershipVector(prediction, dropna=True)
    sample = MembershipVector(sample, dropna=True)

    prediction, sample, weights = _prepare_args(prediction, sample, weights)

    error_table = record_error_table(prediction, sample)
    cs = cluster_sizes_from_table(error_table)
    E_miss = expected_missing_from_table(error_table)
    E_size = expected_size_difference_from_table(error_table)
    weights = 1 / cs

    N = cs * (cs - 1 - E_miss) * weights
    D = cs * (cs - 1 + beta**2 * E_size / (1 + beta**2)) * weights

    return N, D


@ratio__of_means_estimator
def cluster_precision_estimator(prediction, sample, weights):
    """
    Cluster precision design estimator.

    Given a predicted disambiguation `prediction`, a set of ground truth clusters `sample`, and a set of cluster sampling weights `weights` (e.g., inverse probability weights for each cluster), this returns a cluster precision estimate together with its estimated standard deviation.

    Args:
        prediction (Series): Membership vector indexed by cluster elements and with values corresponding to associated cluster identifier. This should cover the entire target population for which cluster precision is being computed.
        sample (Series): Membership vector indexed by cluster elements and with values corresponding to associated cluster identifier.
        weights (Series): Pandas Series indexed by cluster identifier and with values corresponding to cluster sampling weights (e.g., inverse sampling probabilities). Can also be the string "uniform" for uniform sampling weights, or "cluster_size" for inverse cluster size sampling weights.

    Returns:
        tuple: Cluster precision estimate and standard deviation estimate.

    Examples:
        >>> prediction = pd.Series(index=[1,2,3,4,5,6,7,8], data=[1,1,2,3,2,4,4,4])
        >>> sample = pd.Series(index=[1,2,3,4,5,6,7, 8], data=["c1", "c1", "c1", "c2", "c2", "c3", "c3", "c3"])
        >>> cluster_precision_estimator(prediction, sample, weights="uniform")
        (0.26171875, 0.23593232610221093)

    Notes:

        * This estimator requires ``prediction`` to cover the entire population of interest from which sampled clusters were obtained. Do not subset ``prediction`` in any way.
    """
    prediction = MembershipVector(prediction, dropna=True)
    sample = MembershipVector(sample, dropna=True)

    prediction, sample, weights = _prepare_args(prediction, sample, weights)

    cs = cluster_sizes(sample)
    E_delta = 1 - error_indicator(prediction, sample)

    N = len(prediction) * E_delta * weights
    D = prediction.nunique() * cs * weights

    return N, D


@ratio__of_means_estimator
def cluster_recall_estimator(prediction, sample, weights):
    """
    Cluster recall design estimator.

    Given a predicted disambiguation `prediction`, a set of ground truth clusters `sample`, and a set of cluster sampling weights `weights` (e.g., inverse probability weights for each cluster), this returns a cluster recall estimate together with its estimated standard deviation.

    Args:
        prediction (Series): Membership vector indexed by cluster elements and with values corresponding to associated cluster identifier.
        sample (Series): Membership vector indexed by cluster elements and with values corresponding to associated cluster identifier.
        weights (Series): Pandas Series indexed by cluster identifier and with values corresponding to cluster sampling weights (e.g., inverse sampling probabilities). Can also be the string "uniform" for uniform sampling weights, or "cluster_size" for inverse cluster size sampling weights.

    Returns:
        tuple: Cluster recall estimate and standard deviation estimate.

    Examples:
        >>> prediction = pd.Series(index=[1,2,3,4,5,6,7,8], data=[1,1,2,3,2,4,4,4])
        >>> sample = pd.Series(index=[1,2,3,4,5,6,7, 8], data=["c1", "c1", "c1", "c2", "c2", "c3", "c3", "c3"])
        >>> cluster_recall_estimator(prediction, sample, weights="uniform")
        (0.3333333333333333, 0.3333333333333333)
    """
    prediction = MembershipVector(prediction, dropna=True)
    sample = MembershipVector(sample, dropna=True)

    prediction, sample, weights = _prepare_args(prediction, sample, weights)

    error_table = record_error_table(prediction, sample)
    E_delta = 1 - error_indicator_from_table(error_table)

    N = E_delta * weights
    D = weights

    return N, D


@ratio__of_means_estimator
def cluster_f_estimator(prediction, sample, weights, beta=1.0):
    """
    Cluster F-score design estimator.

    Given a predicted disambiguation `prediction`, a set of ground truth clusters `sample`, and a set of cluster sampling weights `weights` (e.g., inverse probability weights for each cluster), this returns a cluster F-score estimate together with its estimated standard deviation.

    Args:
        prediction (Series): Membership vector indexed by cluster elements and with values corresponding to associated cluster identifier. This should cover the entire target population for which cluster f-score is being computed.
        sample (Series): Membership vector indexed by cluster elements and with values corresponding to associated cluster identifier.
        weights (Series): Pandas Series indexed by cluster identifier and with values corresponding to cluster sampling weights (e.g., inverse sampling probabilities). Can also be the string "uniform" for uniform sampling weights, or "cluster_size" for inverse cluster size sampling weights.
        beta (float): F-score weight.

    Returns:
        tuple: Cluster F-score estimate and standard deviation estimate.

    Examples:
        >>> prediction = pd.Series(index=[1,2,3,4,5,6,7,8], data=[1,1,2,3,2,4,4,4])
        >>> sample = pd.Series(index=[1,2,3,4,5,6,7, 8], data=["c1", "c1", "c1", "c2", "c2", "c3", "c3", "c3"])
        >>> cluster_f_estimator(prediction, sample, weights="uniform")

    Notes:

        * This estimator requires ``prediction`` to cover the entire population of interest from which sampled clusters were obtained. Do not subset ``prediction`` in any way.
    """
    prediction = MembershipVector(prediction, dropna=True)
    sample = MembershipVector(sample, dropna=True)

    prediction, sample, weights = _prepare_args(prediction, sample, weights)

    error_table = record_error_table(prediction, sample)
    cs = cluster_sizes_from_table(error_table)
    E_delta = 1 - error_indicator_from_table(error_table)

    multiplier = len(prediction) * (1 + beta**2) / prediction.nunique()

    N = multiplier * E_delta * weights
    D = beta**2 * len(prediction) / prediction.nunique() + cs * weights

    return N, D


@ratio__of_means_estimator
def b_cubed_precision_estimator(prediction, sample, weights):
    """
    B-cubed precision design estimator.

    Given a predicted disambiguation `prediction`, a set of ground truth clusters `sample`, and a set of cluster sampling weights `weights` (e.g., inverse probability weights for each cluster), this returns a B-cubed precision estimate together with its estimated standard deviation.

    Args:
        prediction (Series): Membership vector indexed by cluster elements and with values corresponding to associated cluster identifier.
        sample (Series): Membership vector indexed by cluster elements and with values corresponding to associated cluster identifier.
        weights (Series): Pandas Series indexed by cluster identifier and with values corresponding to cluster sampling weights (e.g., inverse sampling probabilities). Can also be the string "uniform" for uniform sampling weights, or "cluster_size" for inverse cluster size sampling weights.

    Returns:
        tuple: B-cubed precision estimate and standard deviation estimate.

    Examples:
        >>> prediction = pd.Series(index=[1,2,3,4,5,6,7,8], data=[1,1,2,3,2,4,4,4])
        >>> sample = pd.Series(index=[1,2,3,4,5], data=["c1", "c1", "c1", "c2", "c2"])
        >>> weights = pd.Series(1, index=sample.unique()) # Uniform cluster weights
        >>> b_cubed_precision_estimator(prediction, sample, weights)
        (0.7916666666666667, 0.0416666666666673)
    """
    prediction = MembershipVector(prediction, dropna=True)
    sample = MembershipVector(sample, dropna=True)

    prediction, sample, weights = _prepare_args(prediction, sample, weights)

    error_table = record_error_table(prediction, sample)
    E_extra_rel = expected_relative_extra_from_table(error_table)

    N = (1 - E_extra_rel) * weights
    D = weights

    return N, D


@ratio__of_means_estimator
def b_cubed_recall_estimator(prediction, sample, weights):
    """
    B-cubed recall design estimator.

    Given a predicted disambiguation `prediction`, a set of ground truth clusters `sample`, and a set of cluster sampling weights `weights` (e.g., inverse probability weights for each cluster), this returns a B-cubed recall estimate together with its estimated standard deviation.

    Args:
        prediction (Series): Membership vector indexed by cluster elements and with values corresponding to associated cluster identifier.
        sample (Series): Membership vector indexed by cluster elements and with values corresponding to associated cluster identifier.
        weights (Series): Pandas Series indexed by cluster identifier and with values corresponding to cluster sampling weights (e.g., inverse sampling probabilities). Can also be the string "uniform" for uniform sampling weights, or "cluster_size" for inverse cluster size sampling weights.

    Returns:
        tuple: B-cubed recall estimate and standard deviation estimate.

    Examples:
        >>> prediction = pd.Series(index=[1,2,3,4,5,6,7,8], data=[1,1,2,3,2,4,4,4])
        >>> sample = pd.Series(index=[1,2,3,4,5], data=["c1", "c1", "c1", "c2", "c2"])
        >>> weights = pd.Series(1, index=sample.unique()) # Uniform cluster weights
        >>> b_cubed_recall_estimator(prediction, sample, weights)
        (0.5277777777777778, 0.027777777777778203)
    """
    prediction = MembershipVector(prediction, dropna=True)
    sample = MembershipVector(sample, dropna=True)

    prediction, sample, weights = _prepare_args(prediction, sample, weights)

    error_table = record_error_table(prediction, sample)
    E_miss_rel = expected_relative_missing_from_table(error_table)

    N = (1 - E_miss_rel) * weights
    D = weights

    return N, D
