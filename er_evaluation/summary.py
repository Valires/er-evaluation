"""
Clustering Summary Statistics
"""

import pandas as pd
import numpy as np
from scipy.special import comb

from .data_structures import ismembership


def number_of_clusters(membership):
    r"""
    Number of clusters in a given clustering.

    Args:
        membership (Series): Membership vector representation of a clustering.

    Returns:
        int: number of unique cluster identifiers. Note that NAs are not counted.

    Examples:
        >>> membership = pd.Series(index=[1,2,3,4,5,6,7,8], data=[1,1,2,3,2,4,4,4])
        >>> number_of_clusters(membership)
        4
    """
    assert ismembership(membership)

    return membership.nunique()


def number_of_links(membership):
    r"""
    Number of pairwise links associated with a given clustering.

    Args:
        membership (Series): Membership vector representation of a clustering.

    Returns:
        int: Number of pairs of elements belonging to the same cluster. Note that clusters identified by NA values are excluded.

    Examples:
        >>> membership = pd.Series(index=[1,2,3,4,5,6,7,8], data=[1,1,2,3,2,4,4,4])
        >>> number_of_links(membership)
        5.0

    """
    assert ismembership(membership)

    return np.sum(comb(cluster_sizes(membership), 2))


def matching_rate(membership):
    r"""
    Compute the **matching rate** for a given clustering.

    Matching rate:
        This is the proportion of elements belonging to clusters of size at least 2.

    Args:
        membership (Series): Membership vector representation of a clustering.

    Returns:
        int: Number of pairs of elements belonging to the same cluster. Note that clusters identified by NA values are excluded.

    Examples:
        >>> membership = pd.Series(index=[1,2,3,4,5,6,7,8], data=[1,1,2,3,2,4,4,4])
        >>> matching_rate(membership)
        0.875
    """
    assert ismembership(membership)

    counts = membership.groupby(membership).count()

    return (counts * (counts > 1)).sum() / membership.count()


def average_cluster_size(membership):
    """
    Compute the average cluster size.

    Args:
        membership (Series): Membership vector representation of a clustering.

    Returns:
        float: Average cluster size.

    Examples:
        >>> membership = pd.Series(index=[1,2,3,4,5,6,7,8], data=[1,1,2,3,2,4,4,4])
        >>> average_cluster_size(membership)
        2.0
    """
    return cluster_sizes(membership).mean()


def cluster_sizes(membership):
    r"""
    Compute the size of each cluster.

    Args:
        membership (Series): Membership vector representation of a clustering.

    Returns:
        Series: Series indexed by cluster identifier and with values corresponding to cluster size. Note that NA cluster identifiers are excluded.

    Examples:
        >>> membership = pd.Series(index=[1,2,3,4,5,6,7,8], data=[1,1,2,3,2,4,4,4])
        >>> cluster_sizes(membership)
        1    2
        2    2
        3    1
        4    3
        dtype: int64
    """
    assert ismembership(membership)

    return membership.groupby(membership).count()


def cluster_sizes_distribution(membership):
    r"""
    Compute the cluster size distribution

    Args:
        membership (Series): Membership vector representation of a clustering.

    Returns:
        Series: Pandas Series indexed by distinct cluster sizes and with values corresponding to the number of clusters of that size.

    Examples:
        >>> membership = pd.Series(index=[1,2,3,4,5,6,7,8], data=[1,1,2,3,2,4,4,4])
        >>> cluster_sizes_distribution(membership)
        1    1
        2    2
        3    1
        dtype: int64
    """
    assert ismembership(membership)

    cs = cluster_sizes(membership)
    return cs.groupby(cs).count()


def cluster_hill_number(membership, alpha=1):
    r"""
    Compute Hill number of a given order.

    Hill numbers:
        The Hill number of order :math:`\alpha \geq 0` of a given probability distribution :math:`p_i`, :math:`i =0,1,2, \dots`, is defined as

        .. math::

            H_\alpha = \left(\sum_{i} p_i^{\alpha} \right)^{1/(1-\alpha)}

        and continually extended at :math:`\alpha =0, 1`. Here, we let :math:`p_i` be the proportion of clusters of size :math:`i`.

    Args:
        membership (Series): Membership vector representation of a clustering.
        alpha (int, optional): Order of the Hill Number. Defaults to 1.

    Returns:
        float: Hill number of order `alpha` for the given clustering.

    Examples:
        >>> membership = pd.Series(index=[1,2,3,4,5,6,7,8], data=[1,1,2,3,2,4,4,4])
        >>> cluster_hill_number(membership, alpha=0)
        3

        >>> cluster_hill_number(membership, alpha=1)
        2.82842712474619

        >>> cluster_hill_number(membership, alpha=np.Inf)
        2.0
    """
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
    r"""
    Compute the homonimy rate of a given clustering with a set of associated names.

    Homonimy rate:
        The homonimy rate is the proportion of clusters which share a name with another cluster.

    Args:
        membership (Series): Membership vector representation of a clustering.
        names (Series): Series indexed by cluster elements and with values corresponding to the associated name. Note that the index of `names` should exactly match the index of `membership`.

    Returns:
        float: Homonimy rate

    Examples:
        >>> membership = pd.Series(index=[1,2,3,4,5,6,7,8], data=[1,1,2,3,2,4,4,4])
        >>> names = pd.Series(index=[1,2,3,4,5,6,7,8], data=["n1", "n2", "n3", "n4", "n3", "n1", "n2", "n8"])
        >>> homonimy_rate(membership, names)
        0.5
    """
    assert ismembership(membership)
    assert isinstance(names, pd.Series)
    assert membership.index.equals(names.index)

    df = pd.concat(
        {"membership": membership, "name": names}, axis=1, copy=False
    )

    names_count = names.groupby(names).count().reset_index(name="total_count")
    name_count_per_cluster = (
        df.groupby(["name", "membership"])
        .size()
        .reset_index(name="cluster_count")
    )
    merged = name_count_per_cluster.merge(
        names_count,
        left_on="name",
        right_on="index",
        copy=False,
        validate="m:1",
    )
    merged["diff"] = merged.total_count - merged.cluster_count

    return (
        (merged.groupby("membership").agg({"diff": max}) > 0).mean().values[0]
    )


def name_variation_rate(membership, names):
    r"""
    Compute the name variation rate of a given clustering with a set of associated names.

    Name variation rate:
        The name variation rate is the proportion of clusters with name variation within.

    Args:
        membership (Series): Membership vector representation of a clustering.
        names (Series): Series indexed by cluster elements and with values corresponding to the associated name. Note that the index of `names` should exactly match the index of `membership`.

    Returns:
        float: Name variation rate.

    Examples:
        >>> membership = pd.Series(index=[1,2,3,4,5,6,7,8], data=[1,1,2,3,2,4,4,4])
        >>> names = pd.Series(index=[1,2,3,4,5,6,7,8], data=["n1", "n2", "n3", "n4", "n3", "n1", "n2", "n8"])
        >>> name_variation_rate(membership, names)
        0.5
    """
    assert ismembership(membership)
    assert isinstance(names, pd.Series)
    assert membership.index.equals(names.index)

    joined = pd.concat(
        {"membership": membership, "name": names}, axis=1, copy=False
    )

    return (joined.groupby("membership").nunique() > 1).mean().values[0]
