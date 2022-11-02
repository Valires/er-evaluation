import pandas as pd
import numpy as np
from scipy.special import comb

from .data_structures import ismembership


def number_of_clusters(membership):
    assert ismembership(membership)

    return membership.nunique()


def matching_rate(membership):
    """
    Proportion of elements belonging to clusters of size at least 2.
    """
    assert ismembership(membership)

    counts = membership.groupby(membership).count()

    return (counts * (counts > 1)).sum() / len(membership)


def cluster_sizes(membership):
    assert ismembership(membership)

    return membership.groupby(membership).count()


def cluster_sizes_distribution(membership):
    assert ismembership(membership)

    cluster_sizes = cluster_sizes(membership)
    return cluster_sizes.groupby(cluster_sizes).count()


def number_of_clusters(membership):
    assert ismembership(membership)

    return membership.nunique()


def number_of_links(membership):
    assert ismembership(membership)

    return np.sum(comb(cluster_sizes(membership), 2))


def cluster_hill_number(membership, alpha=1):
    assert ismembership(membership)

    cs_dist = cluster_sizes_distribution(membership)
    probs = cs_dist / np.sum(cs_dist)
    probs = probs[probs > 0]

    if alpha == 0:
        return len(probs)
    if alpha == 1:
        return np.exp(-np.sum(probs * np.log(probs)))
    if alpha == np.Inf:
        return 1 / np.max(probs)
    else:
        return np.sum(probs**alpha) ** (1 / (1 - alpha))


def homonimy_rate(membership, names):
    assert ismembership(membership)
    assert isinstance(names, pd.Series)
    assert membership.index.equals(names.index)

    df = pd.concat({"membership": membership, "name": names}, axis=1, copy=False)

    names_count = names.groupby(names).count().reset_index(name="total_count")
    name_count_per_cluster = (
        df.groupby(["name", "membership"]).size().reset_index(name="cluster_count")
    )
    merged = name_count_per_cluster.merge(
        names_count,
        left_on="name",
        right_on="index",
        copy=False,
        validate="m:1",
    )
    merged["diff"] = merged.total_count - merged.cluster_count

    return (merged.groupby("membership").agg({"diff": max}) > 0).mean()


def name_variation_rate(membership, names):
    assert ismembership(membership)
    assert isinstance(names, pd.Series)
    assert membership.index.equals(names.index)

    df = pd.concat({"membership": membership, "name": names}, axis=1, copy=False)

    return (df.groupby("membership").nunique() > 1).mean()


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
    P = number_of_links(inner.prediction).sum()

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
