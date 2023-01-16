import itertools
import logging
from importlib import resources

import numpy as np
import pandas as pd

from er_evaluation.data_structures import MembershipVector


def load_module_tsv(module, filename, dtype=str):
    with resources.open_text(module, filename) as f:
        data = pd.read_csv(f, sep="\t", dtype=dtype)

    return data


def sample_clusters(membership, weights="uniform", sample_prop=0.2, replace=True, random_state=1):
    membership = MembershipVector(membership)
    np.random.seed(random_state)

    if isinstance(weights, pd.Series):
        selected_clusters = np.random.choice(
            weights.index, size=int(sample_prop * membership.nunique()), replace=replace, p=weights.values/np.sum(weights.values)
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
            raise ValueError(f"Invalid weights argument. Valid strings are 'uniform' or 'cluster_size', instead got {weights}")
    else:
        raise ValueError(f"Invalid weights argument. Should be a string or a pandas Series, instead got type {type(weights)}.")

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
