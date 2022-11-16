"""
Entity Resolution Error Analysis Tools

Copyright (C) 2022  Olivier Binette

This file is part of the ER-Evaluation Python package (er-evaluation).

er-evaluation is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import pandas as pd
import numpy as np
from scipy.special import comb


def count_extra_links(prediction, sample):
    """
    Count number of extraneous links to sampled clusters.

    Given a predicted disambiguation `prediction` and a sample of true clusters `sample`, both represented as membership vectors, this functions returns the count of extraneous links for each true cluster. This is a pandas Series indexed by true cluster identifier and with values corresponding to the counts of extraneous links.

    Count of Extraneous Links
        For a given sampled cluster $c$ with records $r \in c$, let $A_r$ be the set of records which are erroneously linked to $r$ in the predicted clustering. That is, if $\hat c(r)$ is the predicted cluster containing $r$, then $A_r = \hat c(r) \backslash c$ Then the count of extraneous links for $c$ is $\sum_{r\in c} \lvert A_r \rvert $.

    Args:
        prediction (Series): Membership vector representing a predicted disambiguation. 
        sample (Series): Membership vector representing a set of true clusters.

    Returns:
        Series: Pandas Series indexed by true cluster identifiers (unique values in `sample`) and with values corresponding to the count of extraneous links.
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

    result = outer.groupby("sample").apply(lambd)
    result.rename("count_extra_links", inplace=True)

    return result


def expected_extra_links(prediction, sample):
    """
    Expected number of extraneous links to records in sampled clusters.

    Given a predicted disambiguation `prediction` and a sample of true clusters `sample`, both represented as membership vectors, this functions returns the expected number of extraneous links for each true cluster. This is a pandas Series indexed by true cluster identifier and with values corresponding to the expected number of extraneous links.

    Expected Number of Extraneous Links
        For a given sampled cluster $c$ with records $r \in c$, let $A_r$ be the set of records which are erroneously linked to $r$ in the predicted clustering. That is, if $\hat c(r)$ is the predicted cluster containing $r$, then $A_r = \hat c(r) \backslash c$ Then the expected number of extraneous links for $c$ is $ \frac{1}{\lvert c \rvert}\sum_{r\in c} \lvert A_r \rvert $. This is the expected number of erroneous links to a random record $r \in c$.

    Args:
        prediction (Series): Membership vector representing a predicted disambiguation. 
        sample (Series): Membership vector representing a set of true clusters.

    Returns:
        Series: Pandas Series indexed by true cluster identifiers (unique values in `sample`) and with values corresponding to the expected number of extraneous links.
    """
    result = count_extra_links(prediction, sample)
    sizes = sample.groupby(sample).size()

    result = result / sizes
    result.rename("expected_extra_links", inplace=True)

    return result


def count_missing_links(prediction, sample):
    """
    Count number of missing links to sampled clusters.

    Given a predicted disambiguation `prediction` and a sample of true clusters `sample`, both represented as membership vectors, this functions returns the count of missing links for each true cluster. This is a pandas Series indexed by true cluster identifier and with values corresponding to the counts of missing links.

    Count of Missing Links
        For a given sampled cluster $c$ with records $r \in c$, let $B_r$ be the set of records which are missing from the predicted cluster containing $r$. That is, if $\hat c(r)$ is the predicted cluster containing $r$, then $B_r = c \backslash \hat c(r)$. Then the count of missing links for $c$ is $\sum_{r\in c} \lvert B_r \rvert $.

    Args:
        prediction (Series): Membership vector representing a predicted disambiguation. 
        sample (Series): Membership vector representing a set of true clusters.

    Returns:
        Series: Pandas Series indexed by true cluster identifiers (unique values in `sample`) and with values corresponding to the count of extraneous links.
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

    result = outer.groupby("sample").apply(lambd)
    result.rename("count_missing_links", inplace=True)

    return result


def expected_missing_links(prediction, sample):
    """
    Expected number of missing links to records in sampled clusters.

    Given a predicted disambiguation `prediction` and a sample of true clusters `sample`, both represented as membership vectors, this functions returns the expected number of missing links for each true cluster. This is a pandas Series indexed by true cluster identifier and with values corresponding to the expected number of missing links.

    Expected Number of Missing Links
        For a given sampled cluster $c$ with records $r \in c$, let $B_r$ be the set of records which are missing from the predicted cluster containing $r$. That is, if $\hat c(r)$ is the predicted cluster containing $r$, then $B_r = c \backslash \hat c(r)$. Then the expected number of missing links for $c$ is $\sum_{r\in c} \lvert B_r \rvert / \lvert c \rvert $.

    Args:
        prediction (Series): Membership vector representing a predicted disambiguation. 
        sample (Series): Membership vector representing a set of true clusters.

    Returns:
        Series: Pandas Series indexed by true cluster identifiers (unique values in `sample`) and with values corresponding to the expected number of missing links.
    """
    result = count_missing_links(prediction, sample)
    sizes = sample.groupby(sample).size()

    result = result / sizes
    result.rename("expected_missing_links", inplace=True)

    return result


def splitting_entropy(prediction, sample, alpha=1):
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

        return (np.sum(u ** alpha)) ** (1 / (1 - alpha))

    result = outer.groupby("sample").apply(lambd)
    result.rename(f"splitting_entropy_{alpha}", inplace=True)

    return result
