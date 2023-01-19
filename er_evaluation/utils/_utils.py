import itertools
import logging
from importlib import resources

import numpy as np
import pandas as pd

from er_evaluation.data_structures import MembershipVector


def load_module_parquet(module, filename):
    """
    Load parquet file from a submodule using pyarrow engine.

    Args:
        module (string): Path to a module, such as "er_evaluation.datasets.raw_data.rldata"
        filename (string): Name of the parquet file.

    Returns:
        pandas DataFrame
    """
    with resources.open_binary(module, filename) as f:
        data = pd.read_parquet(f, engine="pyarrow")

    return data


def load_module_tsv(module, filename, dtype=str):
    """
    Load tsv file from a submodule.

    Args:
        module (string): Path to a module, such as "er_evaluation.datasets.raw_data.rldata"
        filename (string): Name of the tsv file.
        dtype: Data type to use to read the file. Defaults to str.

    Returns:
        pandas DataFrame
    """
    with resources.open_text(module, filename) as f:
        data = pd.read_csv(f, sep="\t", dtype=dtype)

    return data


def sample_clusters(membership, weights="uniform", sample_prop=0.2, replace=True, random_state=1):
    """
    Sample clusters from a membership vector.

    Args:
        membership (Series): Membership vector.
        weights (str, optional): Probability weights to use. Should be one "uniform", "cluster_size", or a pandas Series indexed by cluster identifiers and with values corresponding to probability weights. Defaults to "uniform".
        sample_prop (float, optional): Proportion of clusters to sample. Defaults to 0.2.
        replace (bool, optional): Wether or not to sample with replacement. Defaults to True.
        random_state (int, optional): Random seed. Defaults to 1.

    Returns:
        Series: Membership vector with elements corresponding to sampled clusters.

    Examples:

        Load a toy dataset:

        >>> from er_evaluation.datasets import load_rldata10000_disambiguations
        >>> predictions, reference = load_rldata10000_disambiguations()

        Sample a set of ground truth clusters uniformly at random:

        >>> sample = sample_clusters(reference, weights="uniform", sample_prop=0.2)

        Compute pairwise_precision on the sample:

        >>> from er_evaluation.metrics import pairwise_precision
        >>> pairwise_precision(predictions['name_by'], sample)
        0.96

        Compare to the true precision on the full data:

        >>> pairwise_precision(predictions['name_by'], reference)
        0.7028571428571428

        The metric computed on a sample is over-optimistic (0.96 versus true precision of 0.7). Instead, use an estimator to accurately estimate pairwise precision from a sample, which returns a point estimate and its standard deviation estimate:

        >>> from er_evaluation.estimators import pairwise_precision_design_estimate
        >>> pairwise_precision_design_estimate(predictions['name_by'], sample, weights="uniform")
        (0.7633453805063894, 0.04223296142335369)
    """
    membership = MembershipVector(membership)
    np.random.seed(random_state)

    if isinstance(weights, pd.Series):
        selected_clusters = np.random.choice(
            weights.index,
            size=int(sample_prop * membership.nunique()),
            replace=replace,
            p=weights.values / np.sum(weights.values),
        )
    elif isinstance(weights, str):
        if weights == "uniform":
            selected_clusters = np.random.choice(
                membership.unique(), size=int(sample_prop * membership.nunique()), replace=replace
            )
        elif weights == "cluster_size":
            selected_clusters = np.random.choice(
                membership.values,
                size=int(sample_prop * membership.nunique()),
                replace=replace,
            )
        else:
            raise ValueError(
                f"Invalid weights argument. Valid strings are 'uniform' or 'cluster_size', instead got {weights}"
            )
    else:
        raise ValueError(
            f"Invalid weights argument. Should be a string or a pandas Series, instead got type {type(weights)}."
        )

    return membership[membership.isin(selected_clusters)]


def relevant_prediction_subset(prediction, sample):
    """Return predicted clusters which intersect sampled clusters."""
    prediction = MembershipVector(prediction)
    sample = MembershipVector(sample)

    I = prediction.index.isin(sample.index)
    J = prediction.isin(prediction[I].values)

    relevant_prediction = prediction[J]
    if len(relevant_prediction) == 0:
        logging.warning("Relevant prediction subset is empty: predicted clusters do not overlap sample clusters.")

    return relevant_prediction


def expand_grid(**kwargs):
    """
    Create DataFrame from all combination of elements.

    Args:
        kwargs: Dictionary of elements to combine. Keys become column names.

    Returns:
        DataFrame: DataFrame with columns corresponding to argument names and rows for each combination of argument values.

    Examples:
        >>> expand_grid(col1=[1,2], col2=["a", "b"])
           col1 col2
        0     1    a
        1     1    b
        2     2    a
        3     2    b

        >>> expand_grid(col1={1:"something", 2:"something"}, col2=["a", "b"])
           col1 col2
        0     1    a
        1     1    b
        2     2    a
        3     2    b
    """
    return pd.DataFrame.from_records(itertools.product(*kwargs.values()), columns=kwargs.keys())
