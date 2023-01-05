r"""
Performance Estimators Based on Ground Truth Clusters
"""
import warnings
import pandas as pd
import numpy as np
from scipy.special import comb

from er_evaluation.data_structures import ismembership
from er_evaluation.utils import expand_grid, relevant_prediction_subset
from er_evaluation.summary import cluster_sizes


def _parse_weights(sample, weights):
    """
    Parse weights argument.

    Args:
        sample (Series): Membership vector representation of a clustering.
        weights (Union[Series, str]): Sampling weights. If "uniform", all weights are set to 1. If "cluster_size", weights are set to 1 / cluster size. Otherwise, weights are expected to be a pandas Series with the same index as sample.

    Returns:
        Series: Sampling weights.

    Examples:
        >>> sample = pd.Series(index=[1,2,3,4,5,6,7], data=["c1", "c1", "c1", "c2", "c2", "c3", "c3"])
        >>> _parse_weights(sample, "uniform")
        c1    1
        c2    1
        c3    1
        dtype: int64
        >>> _parse_weights(sample, "cluster_size")
        c1    0.333333
        c2    0.500000
        c3    0.500000
        dtype: int64
        >>> _parse_weights(sample, pd.Series(index=["c1", "c2", "c3"], data=[0.2, 0.4, 0.2]))
        c1    0.2
        c2    0.4
        c3    0.2
        dtype: float64
    """
    if isinstance(weights, str):
        if weights == "uniform":
            return pd.Series(1, index=sample.unique())
        elif weights == "cluster_size":
            return 1 / cluster_sizes(sample)
        else:
            raise ValueError(
                "If weights is a string, it must be either 'uniform' or 'cluster_size'."
            )
    else:
        return weights


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
        >>> estimators = {"precision": pairwise_precision_design_estimate, "recall": pairwise_recall_design_estimate}
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
            lambda x: lambd(
                x["prediction"], x["sample_weights"], x["estimator"]
            ),
            axis=1,
        ).tolist(),
        index=params.index,
    )

    return params


def validate_prediction_sample(prediction, sample):
    r"""
    Validate inputs to estimators.

    Args:
        prediction (Series): Membership vector indexed by cluster elements and with values corresponding to associated cluster identifier.
        sample (Series): Membership vector indexed by cluster elements and with values corresponding to associated cluster identifier.
        weights (Series): Pandas Series indexed by cluster identifier and with values corresponding to cluster sampling weights (e.g., inverse sampling probabilities).

    Raises:
        AssertionError
    """
    assert ismembership(prediction) and ismembership(sample)

    if not all(sample.index.isin(prediction.index)):
        warnings.warn("Some sample elements are not in the prediction.")


def validate_weights(sample, weights):
    assert isinstance(weights, pd.Series)
    assert all(weights.index.isin(sample.unique()))


def pairwise_precision_design_estimate(prediction, sample, weights):
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
        >>> pairwise_precision_design_estimate(prediction, sample, weights)
        (0.3888888888888889, 0.2545875386086578)

    References:
        [1] Binette, Olivier, Sokhna A York, Emma Hickerson, Youngsoo Baek, Sarvo Madhavan, Christina Jones. (2022). Estimating the Performance of Entity Resolution Algorithms: Lessons Learned Through PatentsView.org. arXiv e-prints: arxiv:2210.01230
    """
    validate_prediction_sample(prediction, sample)
    sample = sample[sample.index.isin(prediction.index)]
    prediction = relevant_prediction_subset(prediction, sample)

    weights = _parse_weights(sample, weights)
    validate_weights(sample, weights)
    weights = weights[weights.index.isin(sample.values)]

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

    return (ratio_estimator(N, D), std_dev(N, D))


def pairwise_recall_design_estimate(prediction, sample, weights):
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
        >>> pairwise_recall_design_estimate(prediction, sample, weights)
        (0.296875, 0.10825317547305482)

    References:
        [1] Binette, Olivier, Sokhna A York, Emma Hickerson, Youngsoo Baek, Sarvo Madhavan, Christina Jones. (2022). Estimating the Performance of Entity Resolution Algorithms: Lessons Learned Through PatentsView.org. arXiv e-prints: arxiv:2210.01230
    """
    validate_prediction_sample(prediction, sample)
    sample = sample[sample.index.isin(prediction.index)]
    prediction = relevant_prediction_subset(prediction, sample)

    weights = _parse_weights(sample, weights)
    validate_weights(sample, weights)
    weights = weights[weights.index.isin(sample.values)]

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

    return (ratio_estimator(N, D), std_dev(N, D))


def ratio_estimator(B, A):
    r"""Ratio estimator for mean(B)/mean(A) with bias adjustment."""
    assert len(A) == len(B)

    A_mean = np.mean(A)
    B_mean = np.mean(B)
    n = len(A)

    if B_mean == 0:
        return 0

    if len(A) == 1:
        adj = 1
    else:
        adj = 1 + ((n - 1) * A_mean) ** (-1) * np.mean(
            A * (B / B_mean - A / A_mean)
        )

    return adj * B_mean / A_mean


def std_dev(B, A):
    r"""Standard deviation estimate for ratio estimator."""
    assert len(A) == len(B)

    A_mean = np.mean(A)
    B_mean = np.mean(B)
    n = len(A)

    if np.allclose(A_mean * B_mean, 0):
        return np.nan

    if n == 1:
        return np.nan
    else:
        op = np.sum(
            (A / A_mean) ** 2
            + (B / B_mean) ** 2
            - 2 * (A * B) / (A_mean * B_mean)
        ) / (n * (n - 1))
        if op < 0:
            return np.nan

        return (B_mean / A_mean) * np.sqrt(op)
