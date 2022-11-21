"""
Evaluation Metrics (Precision, Recall, etc)
"""

import pandas as pd
import numpy as np
from scipy.special import comb

from .data_structures import ismembership
from .summary import number_of_links
from .utils import expand_grid


def metrics_table(predictions, references, metrics):
    """
    Apply a set of metrics to all combinations of prediction and reference membership vectors.

    Args:
        predictions (Dict): Dictionary of membership vectors.
        references (Dict): Dictionary of membership vectors.
        metrics (Dict): Dictionary of metrics to apply to the prediction and reference pairs.

    Returns:
        DataFrame: Dataframe with columns "prediction", "reference", "metric", and "value", containing the value of the given metric applied to the corresponding prediction and reference membership vector.

    Examples:
        >>> predictions = {"prediction_1": pd.Series(index=[1,2,3,4,5,6,7,8], data=[1,1,2,3,2,4,4,4])}
        >>> references = {"reference_1": pd.Series(index=[1,2,3,4,5,6,7,8], data=["c1", "c1", "c1", "c2", "c2", "c3", "c3", "c4"])}
        >>> metrics = {"precision": pairwise_precision, "recall": pairwise_recall}
        >>> metrics_table(predictions, references, metrics) # doctest: +NORMALIZE_WHITESPACE
            prediction	    reference	metric	    value
        0	prediction_1	reference_1	precision	0.4
        1	prediction_1	reference_1	recall	    0.4
    """
    params = expand_grid(
        prediction=predictions, reference=references, metric=metrics
    )

    def lambd(pred_key, ref_key, metrics_key):
        return metrics[metrics_key](predictions[pred_key], references[ref_key])

    params["value"] = params.apply(
        lambda x: lambd(x["prediction"], x["reference"], x["metric"]), axis=1
    )

    return params


def pairwise_precision(prediction, reference):
    r"""
    Pairwise precision for the inner join of two clusterings.

    Pairwise precision:
        Consider two clusterings of a set of records, refered to as the *predicted* and *reference* clusterings. Let $T$ be the set of record pairs which appear in the same reference cluster, and let $P$ be the set of record pairs which appear in the same predicted clusters. Pairwise precision is then defined as

        .. math::

            P = \frac{\lvert T \cap P \rvert}{\lvert P \rvert}

        This is the proportion of correctly predicted links among all predicted links.

    Args:
        prediction (Series): Membership vector for the predicted clustering.
        reference (Series): Membership vector for the reference clustering.

    Returns:
        float: Pairwise precision for the inner join of `prediction` and `reference`.

    Examples:
        >>> prediction = pd.Series(index=[1,2,3,4,5,6,7,8], data=[1,1,2,3,2,4,4,4])
        >>> reference = pd.Series(index=[1,2,3,4,5,6,7,8], data=["c1", "c1", "c1", "c2", "c2", "c3", "c3", "c4"])
        >>> pairwise_precision(prediction, reference)
        0.4
    """

    assert ismembership(prediction) and ismembership(reference)

    inner = pd.concat(
        {"prediction": prediction, "reference": reference},
        axis=1,
        join="inner",
        copy=False,
    )
    TP_cluster_sizes = inner.groupby(["prediction", "reference"]).size().values

    TP = np.sum(comb(TP_cluster_sizes, 2))
    P = number_of_links(inner.prediction)

    if P == 0:
        return 1.0
    else:
        return TP / P


def pairwise_recall(prediction, reference):
    r"""
    Pairwise recall for the inner join of two clusterings.

    Pairwise recall:
        Consider two clusterings of a set of records, refered to as the *predicted* and *reference* clusterings. Let $T$ be the set of record pairs which appear in the same reference cluster, and let $P$ be the set of record pairs which appear in the same predicted clusters. Pairwise recall is then defined as

        .. math::

            R = \frac{\lvert T \cap P \rvert}{\lvert T \rvert}

        This is the proportion of correctly predicted links among all true links.

    Args:
        prediction (Series): Membership vector for the predicted clustering.
        reference (Series): Membership vector for the reference clustering.

    Returns:
        float: Pairwise recall computed on the inner join of `predicted` and `reference`.

    Examples:
        >>> prediction = pd.Series(index=[1,2,3,4,5,6,7,8], data=[1,1,2,3,2,4,4,4])
        >>> reference = pd.Series(index=[1,2,3,4,5,6,7,8], data=["c1", "c1", "c1", "c2", "c2", "c3", "c3", "c4"])
        >>> pairwise_recall(prediction, reference)
        0.4
    """
    return pairwise_precision(reference, prediction)
