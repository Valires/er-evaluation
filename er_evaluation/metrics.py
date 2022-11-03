import pandas as pd
import numpy as np
from scipy.special import comb

from .data_structures import ismembership
from .summary import number_of_links, number_of_clusters


def pairwise_precision(prediction, reference):
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
    return pairwise_precision(reference, prediction)


def cluster_precision(prediction, reference):
    assert ismembership(prediction) and ismembership(reference)

    inner = pd.concat(
        {"prediction": prediction, "reference": reference},
        axis=1,
        join="inner",
        copy=False,
    )
    n_correct_clusters = np.sum(
        inner.groupby(["prediction"]).nunique()["reference"].values == 1
    )

    return n_correct_clusters / number_of_clusters(inner.prediction)


def cluster_recall(prediction, reference):
    return cluster_precision(reference, prediction)
