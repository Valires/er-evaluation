r"""
=============================================================
Error Analysis: Analyze Errors Based on Ground Truth Clusters
=============================================================

The **error_analysis** submodule provides a set of tools to analyze errors, given a set of ground truth clusters. These ground truth clusters may correspond to a benchmark dataset which is *complete* (all of the entities within it are fully resolved and have no missing links), or to a probability sample of ground truth clusters. For more information on the underlying methodology, please refer to [XX].

The key assumptions used for this module are:
1. A *predicted* clustering is available as a membership vector (named  `prediction` throughout).
2. A set of ground truth clusters is available as a membership vector (named `sample` throughout).

Given these two elements, error metrics are associated to each true entity (corresponding to a cluster in `sample`). The corresponding error space can then be analyzed to identify systematic errors, identify performance disparities, and investigate root causes. 

**Toy Example**

For example, consider the following set of ground truth clusters and predicted clusters of records :math:`1,2,\dots, 8`::

                             ┌───────┐  ┌─────┐  ┌───┐
                             │ 1   2 │  │  4  │  │ 6 │  ┌───┐
              True clusters: │       │  │     │  │   │  │ 8 │
                             │   3   │  │  5  │  │ 7 │  └───┘
                             └───────┘  └─────┘  └───┘   c4
                                 c1        c2      c3
    
                             ┌───────┐  ┌─────┐  ┌──────────┐
                             │ 1   2 │  │  4  │  │ 6        │
         Predicted clusters: ├───────┴──┴─────┤  │        8 │
                             │   3         5  │  │ 7        │
                             └────────────────┘  └──────────┘
    

Assume that the ground truth clusters `c1`, `c2`, and `c4` are available in a benchmark dataset `sample`. Then, we have::

    >>> import pandas as pd
    >>> prediction = pd.Series(index=[1,2,3,4,5,6,7,8], data=[1,1,2,3,2,4,4,4])
    >>> sample = pd.Series(index=[1,2,3,4,5,8], data=["c1", "c1", "c1", "c2", "c2", "c4"])

The following error metrics, namely the splitting entropy, expected number of extraneous links, and expected number of missing links, are used to quantify errors associated with each ground truth cluster. Refer to the API documentation for full definitions::

    >>> from er_evaluation.error_analysis import (splitting_entropy, expected_extra_links, expected_missing_links)

    >>> expected_extra_links(prediction, sample)
    sample
    c1    0.333333
    c2    0.500000
    c4    2.000000
    Name: expected_extra_links, dtype: float64

    >>> expected_missing_links(prediction, sample)
    sample
    c1    1.333333
    c2    1.000000
    c4    0.000000
    Name: expected_missing_links, dtype: float64

    >>> splitting_entropy(prediction, sample)
    sample
    c1    1.889882
    c2    2.000000
    c4    1.000000
    Name: splitting_entropy_1, dtype: float64
"""

import pandas as pd
import numpy as np
from scipy.special import comb


def count_extra_links(prediction, sample):
    r"""
    Count number of extraneous links to sampled clusters.

    Given a predicted disambiguation `prediction` and a sample of true clusters `sample`, both represented as membership vectors, this functions returns the count of extraneous links for each true cluster. This is a pandas Series indexed by true cluster identifier and with values corresponding to the counts of extraneous links.

    Count of Extraneous Links
        For a given sampled cluster :math:`c` with records :math:`r \in c`, let :math:`A_r` be the set of records which are erroneously linked to :math:`r` in the predicted clustering. That is, if :math:`\hat c(r)` is the predicted cluster containing :math:`r`, then :math:`A_r = \hat c(r) \backslash c` Then the count of extraneous links for :math:`c` is :math:`\sum_{r\in c} \lvert A_r \rvert`.

    Args:
        prediction (Series): Membership vector representing a predicted disambiguation.
        sample (Series): Membership vector representing a set of true clusters.

    Returns:
        Series: Pandas Series indexed by true cluster identifiers (unique values in `sample`) and with values corresponding to the count of extraneous links.

    Examples:
        >>> prediction = pd.Series(index=[1,2,3,4,5,6,7,8], data=[1,1,2,3,2,4,4,4])
        >>> sample = pd.Series(index=[1,2,3,4,5,8], data=["c1", "c1", "c1", "c2", "c2", "c4"])
        >>> count_extra_links(prediction, sample)
        sample
        c1    1
        c2    1
        c4    2
        Name: count_extra_links, dtype: int64
    """
    # Index of elements with a predicted cluster intersecting sampled clusters:
    I = prediction.isin(prediction[prediction.index.isin(sample.index)])
    relevant_predictions = prediction[I]

    outer = pd.concat(
        {"prediction": relevant_predictions, "sample": sample},
        axis=1,
        copy=False,
        join="outer",
    )

    def lambd(sample_cluster):
        # Number of elements within sampled cluster split across predicted clusters:
        p = pd.value_counts(sample_cluster)
        # Number of elements within predicted clusters (restricted to current sampled cluster):
        u = outer.prediction.value_counts()[p.index].values

        n_links = np.sum(p * (u - p)) + np.sum(comb(u, 2))

        if n_links == 0:
            return 0

        return np.sum(p * (u - p))

    result = outer.groupby("sample").agg(lambd).prediction
    result.rename("count_extra_links", inplace=True)

    return result


def expected_extra_links(prediction, sample):
    r"""
    Expected number of extraneous links to records in sampled clusters.

    Given a predicted disambiguation `prediction` and a sample of true clusters `sample`, both represented as membership vectors, this functions returns the expected number of extraneous links for each true cluster. This is a pandas Series indexed by true cluster identifier and with values corresponding to the expected number of extraneous links.

    Expected Number of Extraneous Links
        For a given sampled cluster :math:`c` with records :math:`r \in c`, let :math:`A_r` be the set of records which are erroneously linked to :math:`r` in the predicted clustering. That is, if :math:`\hat c(r)` is the predicted cluster containing :math:`r`, then :math:`A_r = \hat c(r) \backslash c` Then the expected number of extraneous links for :math:`c` is :math:` \frac{1}{\lvert c \rvert}\sum_{r\in c} \lvert A_r \rvert :math:`. This is the expected number of erroneous links to a random record :math:`r \in c`.

    Args:
        prediction (Series): Membership vector representing a predicted disambiguation.
        sample (Series): Membership vector representing a set of true clusters.

    Returns:
        Series: Pandas Series indexed by true cluster identifiers (unique values in `sample`) and with values corresponding to the expected number of extraneous links.

    Examples:
        >>> prediction = pd.Series(index=[1,2,3,4,5,6,7,8], data=[1,1,2,3,2,4,4,4])
        >>> sample = pd.Series(index=[1,2,3,4,5,8], data=["c1", "c1", "c1", "c2", "c2", "c4"])
        >>> expected_extra_links(prediction, sample)
        sample
        c1    0.333333
        c2    0.500000
        c4    2.000000
        Name: expected_extra_links, dtype: float64
    """
    result = count_extra_links(prediction, sample)
    sizes = sample.groupby(sample).size()

    result = result / sizes
    result.rename("expected_extra_links", inplace=True)

    return result


def count_missing_links(prediction, sample):
    r"""
    Count number of missing links to sampled clusters.

    Given a predicted disambiguation `prediction` and a sample of true clusters `sample`, both represented as membership vectors, this functions returns the count of missing links for each true cluster. This is a pandas Series indexed by true cluster identifier and with values corresponding to the counts of missing links.

    Count of Missing Links
        For a given sampled cluster :math:`c` with records :math:`r \in c`, let :math:`B_r` be the set of records which are missing from the predicted cluster containing :math:`r`. That is, if :math:`\hat c(r)` is the predicted cluster containing :math:`r`, then :math:`B_r = c \backslash \hat c(r)`. Then the count of missing links for :math:`c` is :math:`\sum_{r\in c} \lvert B_r \rvert :math:`.

    Args:
        prediction (Series): Membership vector representing a predicted disambiguation.
        sample (Series): Membership vector representing a set of true clusters.

    Returns:
        Series: Pandas Series indexed by true cluster identifiers (unique values in `sample`) and with values corresponding to the count of extraneous links.

    Examples:
        >>> prediction = pd.Series(index=[1,2,3,4,5,6,7,8], data=[1,1,2,3,2,4,4,4])
        >>> sample = pd.Series(index=[1,2,3,4,5,8], data=["c1", "c1", "c1", "c2", "c2", "c4"])
        >>> count_missing_links(prediction, sample)
        sample
        c1    4
        c2    2
        c4    0
        Name: count_missing_links, dtype: int64
    """
    # Index of elements with a predicted cluster intersecting sampled clusters:
    I = prediction.isin(prediction[prediction.index.isin(sample.index)])
    relevant_predictions = prediction[I]

    outer = pd.concat(
        {"prediction": relevant_predictions, "sample": sample},
        axis=1,
        copy=False,
        join="outer",
    )

    def lambd(sample_cluster):
        p = pd.value_counts(sample_cluster)
        n = np.sum(p)

        n_links = comb(n, 2)
        if n_links == 0:
            return 0

        return np.sum(p * (n - p))

    result = outer.groupby("sample").agg(lambd).prediction
    result.rename("count_missing_links", inplace=True)

    return result


def expected_missing_links(prediction, sample):
    r"""
    Expected number of missing links to records in sampled clusters.

    Given a predicted disambiguation `prediction` and a sample of true clusters `sample`, both represented as membership vectors, this functions returns the expected number of missing links for each true cluster. This is a pandas Series indexed by true cluster identifier and with values corresponding to the expected number of missing links.

    Expected Number of Missing Links
        For a given sampled cluster  :math:`c` with records  :math:`r \in c`, let :math:`B_r` be the set of records which are missing from the predicted cluster containing :math:`r`. That is, if :math:`\hat c(r)` is the predicted cluster containing :math:`r`, then :math:`B_r = c \backslash \hat c(r)`. Then the expected number of missing links for :math:`c` is :math:`\sum_{r\in c} \lvert B_r \rvert / \lvert c \rvert :math:`.

    Args:
        prediction (Series): Membership vector representing a predicted disambiguation.
        sample (Series): Membership vector representing a set of true clusters.

    Returns:
        Series: Pandas Series indexed by true cluster identifiers (unique values in `sample`) and with values corresponding to the expected number of missing links.

    Examples:
        >>> prediction = pd.Series(index=[1,2,3,4,5,6,7,8], data=[1,1,2,3,2,4,4,4])
        >>> sample = pd.Series(index=[1,2,3,4,5,8], data=["c1", "c1", "c1", "c2", "c2", "c4"])
        >>> expected_missing_links(prediction, sample)
        sample
        c1    1.333333
        c2    1.000000
        c4    0.000000
        Name: expected_missing_links, dtype: float64
    """
    result = count_missing_links(prediction, sample)
    sizes = sample.groupby(sample).size()

    result = result / sizes
    result.rename("expected_missing_links", inplace=True)

    return result


def splitting_entropy(prediction, sample, alpha=1):
    r"""
    Splitting entropy of true clusters

    This function returns the splitting entropy, defined below, of each entity represented in the sampled clusters `sample`.

    Splitting Entropy:
        Let :math:`\hat{\mathcal{C}}` be a clustering of records :math:`\mathcal{R}` into **predicted** entities. For a given entity represented by a cluster :math:`c`, the splitting entropy is defined as the exponentiated Shannon entropy of the set of cluster sizes :math:`\{\lvert \hat c \cap c \rvert \mid \hat c \in \widehat{\mathcal{C}},\, \lvert \hat c \cap c \rvert > 0 \}`. That is, with using the convention that :math:`0 \cdot \log (0) = 0`, we have

        .. math::

            E_{\text{split}}(c) = \exp\left \{-\sum_{\hat c \in \widehat{\mathcal{C}}} \frac{\lvert\hat c \cap c \rvert}{\sum_{\hat c' \in \widehat{\mathcal{C}}} \lvert \hat c' \cap c \rvert } \log \left(\frac{\lvert\hat c \cap c \rvert}{\sum_{\hat c' \in \widehat{\mathcal{C}}} \lvert \hat c' \cap c \rvert }\right) \right \}.

    Args:
        prediction (Series): Membership vector representing a predicted disambiguation.
        sample (Series): Membership vector representing a set of true clusters.

    Returns:
        Series: Pandas Series indexed by true cluster identifiers (unique values in `sample`) and with values corresponding to the splitting entropy.

    Examples:
        >>> prediction = pd.Series(index=[1,2,3,4,5,6,7,8], data=[1,1,2,3,2,4,4,4])
        >>> sample = pd.Series(index=[1,2,3,4,5,8], data=["c1", "c1", "c1", "c2", "c2", "c4"])
        >>> splitting_entropy(prediction, sample)
        sample
        c1    1.889882
        c2    2.000000
        c4    1.000000
        Name: splitting_entropy_1, dtype: float64
    """

    # Index of elements with a predicted cluster intersecting sampled clusters:
    I = prediction.isin(prediction[prediction.index.isin(sample.index)])
    relevant_predictions = prediction[I]

    outer = pd.concat(
        {"prediction": relevant_predictions, "sample": sample},
        axis=1,
        copy=False,
        join="outer",
    )

    def lambd(sample_cluster):
        u = pd.value_counts(sample_cluster, normalize=True).values

        if len(u) <= 1:
            return 1
        if alpha == 0:
            return len(u)
        if alpha == 1:
            return np.exp(-np.sum(u * np.log(u)))

        return (np.sum(u**alpha)) ** (1 / (1 - alpha))

    result = outer.groupby("sample").agg(lambd).prediction
    result.rename(f"splitting_entropy_{alpha}", inplace=True)

    return result
