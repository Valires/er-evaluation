r"""
Performance Estimators Based on Ground Truth Clusters
"""

import pandas as pd
import numpy as np
from scipy.special import comb

from .data_structures import ismembership


def validate_estimator_arguments(prediction, sample, weights):
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
    assert isinstance(weights, pd.Series)
    assert not weights.index.has_duplicates and all(
        weights.index.isin(sample.values)
    )


def pairwise_precision_design_estimate(prediction, sample, weights):
    r"""
    Design estimator for pairwise precision.

    Given a predicted disambiguation `prediction`, a set of ground truth clusters `sample`, and a set of cluster sampling weights `weights` (e.g., inverse probability weights for each cluster), this returns a pairwise precision estimate together with its estimated standard deviation.

    Note:
        This is the precision estimator corresponding to cluster block sampling in [1].

    Args:
        prediction (Series): Membership vector indexed by cluster elements and with values corresponding to associated cluster identifier.
        sample (Series): Membership vector indexed by cluster elements and with values corresponding to associated cluster identifier.
        weights (Series): Pandas Series indexed by cluster identifier and with values corresponding to cluster sampling weights (e.g., inverse sampling probabilities).

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
    validate_estimator_arguments(prediction, sample, weights)

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
        weights (Series): Pandas Series indexed by cluster identifier and with values corresponding to cluster sampling weights (e.g., inverse sampling probabilities).

    Returns:
        tuple: Recall estimate and standard deviation estimate.

    Examples:
        >>> prediction = pd.Series(index=[1,2,3,4,5,6,7,8], data=[1,1,2,3,2,4,4,4])
        >>> sample = pd.Series(index=[1,2,3,4,5,8], data=["c1", "c1", "c1", "c2", "c2", "c4"])
        >>> weights = pd.Series(1, index=sample.unique()) # Uniform cluster weights
        >>> pairwise_recall_design_estimate(prediction, sample, weights)
        (0.4027777777777778, 0.21245914639969934)

    References:
        [1] Binette, Olivier, Sokhna A York, Emma Hickerson, Youngsoo Baek, Sarvo Madhavan, Christina Jones. (2022). Estimating the Performance of Entity Resolution Algorithms: Lessons Learned Through PatentsView.org. arXiv e-prints: arxiv:2210.01230
    """
    validate_estimator_arguments(prediction, sample, weights)

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
