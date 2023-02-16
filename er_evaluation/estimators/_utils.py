import logging
import functools
import numpy as np
import pandas as pd

from er_evaluation.data_structures import MembershipVector
from er_evaluation.summary import cluster_sizes


def ratio__of_means_estimator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        N, D = func(*args, **kwargs)
        return (ratio_estimator(N, D), std_dev(N, D))
    return wrapper


def ratio_estimator(B, A):
    """Ratio estimator for mean(B)/mean(A) with bias adjustment."""
    assert len(A) == len(B)

    A_mean = np.mean(A)
    B_mean = np.mean(B)
    n = len(A)

    if B_mean == 0:
        return 0

    if len(A) == 1:
        adj = 1
    else:
        adj = 1 + ((n - 1) * A_mean) ** (-1) * np.mean(A * (B / B_mean - A / A_mean))

    return adj * B_mean / A_mean


def std_dev(B, A):
    """Standard deviation estimate for ratio estimator."""
    assert len(A) == len(B)

    A_mean = np.mean(A)
    B_mean = np.mean(B)
    n = len(A)

    if np.allclose(A_mean * B_mean, 0):
        return np.nan

    if n == 1:
        return np.nan
    else:
        op = np.sum((A / A_mean) ** 2 + (B / B_mean) ** 2 - 2 * (A * B) / (A_mean * B_mean)) / (n * (n - 1))
        if op < 0:
            return np.nan

        return (B_mean / A_mean) * np.sqrt(op)


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
    prediction = MembershipVector(prediction, dropna=True)
    sample = MembershipVector(sample, dropna=True)

    if not all(sample.index.isin(prediction.index)):
        logging.info("Some sample elements are not in the prediction.")


def validate_weights(sample, weights):
    assert isinstance(weights, pd.Series)
    assert all(weights.index.isin(sample.unique()))


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
        dtype: float64
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
            raise ValueError("If weights is a string, it must be either 'uniform' or 'cluster_size'.")
    else:
        return weights
