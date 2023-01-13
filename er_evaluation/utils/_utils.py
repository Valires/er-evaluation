import itertools
import logging

import pandas as pd

from er_evaluation.data_structures import MembershipVector


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
