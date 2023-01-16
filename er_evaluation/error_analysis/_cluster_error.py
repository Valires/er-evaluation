import numpy as np
import pandas as pd
from scipy.special import comb

from er_evaluation.data_structures import MembershipVector
from er_evaluation.error_analysis._record_error import (
    error_metrics_from_table,
    expected_size_difference_from_table,
    record_error_table,
)
from er_evaluation.utils import relevant_prediction_subset


def error_metrics(prediction, sample):
    """
    Compute canonical set of error metrics from record error table.

    Error metrics included:

    * Expected extra links (see :meth:`er_evaluation.error_analysis.expected_extra_links`)
    * Expected relative extra links (see :meth:`er_evaluation.error_analysis.expected_relative_extra_links`)
    * Expected missing links (see :meth:`er_evaluation.error_analysis.expected_missing_links`)
    * Expected relative missing links (see :meth:`er_evaluation.error_analysis.expected_relative_missing_links`)
    * Error indicator (see :meth:`er_evaluation.error_analysis.error_indicator`)

    Args:
        prediction (Series): Membership vector representing a predicted disambiguation.
        sample (Series): Membership vector representing a set of true clusters.

    Returns:
        DataFrame: Dataframe indexed by cluster identifiers and with values corresponding to error metrics.

    Examples
        >>> prediction = pd.Series(index=[1,2,3,4,5,6,7,8], data=[1,1,2,3,2,4,4,4])
        >>> sample = pd.Series(index=[1,2,3,4,5,6,7, 8], data=["c1", "c1", "c1", "c2", "c2", "c3", "c3", "c3"])
        >>> error_metrics(prediction, sample)  # doctest: +SKIP
        expected_extra_links	expected_relative_extra_links	expected_missing_links	expected_relative_missing_links	error_indicator
        reference
        c1	0.333333	0.166667	1.333333	0.444444	1
        c2	0.500000	0.250000	1.000000	0.500000	1
        c3	1.000000	0.333333	0.000000	0.000000	0
    """
    error_table = record_error_table(prediction, sample)
    return error_metrics_from_table(error_table)


def count_extra_links(prediction, sample):
    r"""
    Count the number of extraneous links to sampled clusters.

    Given a predicted disambiguation ``prediction`` and a sample of true clusters ``sample``, both represented as membership vectors, this functions returns the count of extraneous links for each true cluster. This is a pandas Series indexed by true cluster identifier and with values corresponding to the counts of extraneous links.

    Count of Extraneous Links
        For a given sampled cluster :math:`c` with records :math:`r \in c`, let :math:`A_r` be the set of records which are erroneously linked to :math:`r` in the predicted clustering. That is, if :math:`\hat c(r)` is the predicted cluster containing :math:`r`, then :math:`A_r = \hat c(r) \backslash c` Then the count of extraneous links for :math:`c` is

        .. math::

            E_{\text{count_extra}}(c) = \sum_{r\in c} \lvert A_r \rvert.

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
    prediction = MembershipVector(prediction)
    sample = MembershipVector(sample)

    relevant_predictions = relevant_prediction_subset(prediction, sample)

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


def expected_size_difference(prediction, sample):
    r"""
    Expected size difference between predicted and sampled clusters.

    Expected Size Difference:
     For a given sampled cluster :math:`c` with records :math:`r \in c`, let :math:`\hat c(r)` be the predicted cluster containing :math:`r`. Then the expected size difference for :math:`c` is

     .. math::

        E_{\text{size}}(c)  = \frac{1}{\lvert c \rvert}\sum_{r\in c} \lvert \hat c(r) \rvert - \lvert c \rvert.

    Args:
        prediction (Series): Membership vector representing a predicted disambiguation.
        sample (Series): Membership vector representing a set of true clusters.

    Returns:
        Series: Pandas Series indexed by true cluster identifiers (unique values in `sample`) and with values corresponding to the expected size difference.

    Examples:
        >>> prediction = pd.Series(index=[1,2,3,4,5,6,7,8], data=[1,1,2,3,2,4,4,4])
        >>> sample = pd.Series(index=[1,2,3,4,5,6,7], data=["c1", "c1", "c1", "c2", "c2", "c3", "c3"])
        >>> expected_size_difference(prediction, sample)
        reference
        c1   -1.0
        c2   -0.5
        c3    1.0
        Name: expected_size_diff, dtype: float64
    """
    error_table = record_error_table(prediction, sample)

    return expected_size_difference_from_table(error_table)


def expected_extra_links(prediction, sample):
    r"""
    Expected number of extraneous links to records in sampled clusters.

    Given a predicted disambiguation ``prediction`` and a sample of true clusters ``sample``, both represented as membership vectors, this functions returns the expected number of extraneous links for each true cluster. This is a pandas Series indexed by true cluster identifier and with values corresponding to the expected number of extraneous links.

    Expected Number of Extraneous Links
        For a given sampled cluster :math:`c` with records :math:`r \in c`, let :math:`A_r` be the set of records which are erroneously linked to :math:`r` in the predicted clustering. That is, if :math:`\hat c(r)` is the predicted cluster containing :math:`r`, then :math:`A_r = \hat c(r) \backslash c` Then the expected number of extraneous links for :math:`c` is

        .. math::

            E_{\text{extra}}(c) = \frac{1}{\lvert c \rvert}\sum_{r\in c} \lvert A_r \rvert.

        This is the expected number of erroneous links to a random record :math:`r \in c`.

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
    prediction = MembershipVector(prediction)
    sample = MembershipVector(sample)

    result = count_extra_links(prediction, sample)
    sizes = sample.groupby(sample).size()

    result = result / sizes
    result.rename("expected_extra_links", inplace=True)

    return result


def expected_relative_extra_links(prediction, sample):
    r"""
    Expected relative number of extraneous links to records in sampled clusters.

    Given a predicted disambiguation ``prediction`` and a sample of true clusters ``sample``, both represented as membership vectors, this functions returns the expected number of relative extraneous links for each true cluster. This is a pandas Series indexed by true cluster identifier and with values corresponding to the expected number of relative extraneous links.

    Expected Relative Number of Extraneous Links
        For a given sampled cluster :math:`c` with records :math:`r \in c`, let :math:`A_r` be the set of records which are erroneously linked to :math:`r` in the predicted clustering. That is, if :math:`\hat c(r)` is the predicted cluster containing :math:`r`, then :math:`A_r = \hat c(r) \backslash c` Then the expected number of extraneous links for :math:`c` is

        .. math::

            E_{\text{rel_extra}}(c) = \frac{1}{\lvert c \rvert}\sum_{r\in c} \lvert A_r \rvert / \lvert \hat c(r) \rvert.

        This is the expected relative number of erroneous links to a random record :math:`r \in c`.

    Args:
        prediction (Series): Membership vector representing a predicted disambiguation.
        sample (Series): Membership vector representing a set of true clusters.

    Returns:
        Series: Pandas Series indexed by true cluster identifiers (unique values in `sample`) and with values corresponding to the expected number of extraneous links.

    Examples:
        >>> prediction = pd.Series(index=[1,2,3,4,5,6,7,8], data=[1,1,2,3,2,4,4,4])
        >>> sample = pd.Series(index=[1,2,3,4,5,8], data=["c1", "c1", "c1", "c2", "c2", "c4"])
        >>> expected_relative_extra_links(prediction, sample)
        sample
        c1    0.166667
        c2    0.250000
        c4    0.666667
        Name: expected_relative_extra_links, dtype: float64
    """
    prediction = MembershipVector(prediction)
    sample = MembershipVector(sample)

    relevant_predictions = relevant_prediction_subset(prediction, sample)

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

        return np.sum(p * (u - p) / u)

    outer.groupby("sample").agg(lambd)

    result = outer.groupby("sample").agg(lambd).prediction
    sizes = sample.groupby(sample).size()
    result = result / sizes
    result.rename("expected_relative_extra_links", inplace=True)

    return result


def count_missing_links(prediction, sample):
    r"""
    Count the number of missing links to sampled clusters.

    Given a predicted disambiguation ``prediction`` and a sample of true clusters ``sample``, both represented as membership vectors, this functions returns the count of missing links for each true cluster. This is a pandas Series indexed by true cluster identifier and with values corresponding to the counts of missing links.

    Count of Missing Links
        For a given sampled cluster :math:`c` with records :math:`r \in c`, let :math:`B_r` be the set of records which are missing from the predicted cluster containing :math:`r`. That is, if :math:`\hat c(r)` is the predicted cluster containing :math:`r`, then :math:`B_r = c \backslash \hat c(r)`. Then the count of missing links for :math:`c` is

        .. math::

            E_{\text{count_miss}}(c) = \sum_{r\in c} \lvert B_r \rvert.

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
    prediction = MembershipVector(prediction)
    sample = MembershipVector(sample)

    relevant_predictions = relevant_prediction_subset(prediction, sample)

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

    Given a predicted disambiguation ``prediction`` and a sample of true clusters ``sample``, both represented as membership vectors, this functions returns the expected relative number of missing links for each true cluster. This is a pandas Series indexed by true cluster identifier and with values corresponding to the expected number of missing links.

    Expected Number of Missing Links
        For a given sampled cluster  :math:`c` with records  :math:`r \in c`, let :math:`B_r` be the set of records which are missing from the predicted cluster containing :math:`r`. That is, if :math:`\hat c(r)` is the predicted cluster containing :math:`r`, then :math:`B_r = c \backslash \hat c(r)`. Then the expected number of missing links for

        .. math::

            E_{\text{miss}}(c) = \frac{1}{\lvert c \rvert}\sum_{r\in c} \lvert B_r \rvert.

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
    prediction = MembershipVector(prediction)
    sample = MembershipVector(sample)

    result = count_missing_links(prediction, sample)
    sizes = sample.groupby(sample).size()

    result = result / sizes
    result.rename("expected_missing_links", inplace=True)

    return result


def expected_relative_missing_links(prediction, sample):
    r"""
    Expected relative number of missing links to records in sampled clusters.

    Given a predicted disambiguation ``prediction`` and a sample of true clusters ``sample``, both represented as membership vectors, this functions returns the expected number of missing links for each true cluster. This is a pandas Series indexed by true cluster identifier and with values corresponding to the expected relative number of missing links.

    Expected Relative Number of Missing Links
        For a given sampled cluster  :math:`c` with records  :math:`r \in c`, let :math:`B_r` be the set of records which are missing from the predicted cluster containing :math:`r`. That is, if :math:`\hat c(r)` is the predicted cluster containing :math:`r`, then :math:`B_r = c \backslash \hat c(r)`. Then the expected number of missing links for :math:`c` is

        .. math::

            E_{\text{rel_miss}}(c) = \frac{1}{\lvert c \rvert}\sum_{r\in c} \lvert B_r \rvert / \lvert c \rvert.

    Args:
        prediction (Series): Membership vector representing a predicted disambiguation.
        sample (Series): Membership vector representing a set of true clusters.

    Returns:
        Series: Pandas Series indexed by true cluster identifiers (unique values in `sample`) and with values corresponding to the expected relative number of missing links.

    Examples:
        >>> prediction = pd.Series(index=[1,2,3,4,5,6,7,8], data=[1,1,2,3,2,4,4,4])
        >>> sample = pd.Series(index=[1,2,3,4,5,8], data=["c1", "c1", "c1", "c2", "c2", "c4"])
        >>> expected_relative_missing_links(prediction, sample)
        sample
        c1    0.444444
        c2    0.500000
        c4    0.000000
        Name: expected_relative_missing_links, dtype: float64
    """
    prediction = MembershipVector(prediction)
    sample = MembershipVector(sample)

    result = count_missing_links(prediction, sample)
    sizes = sample.groupby(sample).size()

    result = result / sizes**2
    result.rename("expected_relative_missing_links", inplace=True)

    return result


def error_indicator(prediction, sample):
    r"""
    Error indicator metric.

    Given a predicted disambiguation ``prediction`` and a sample of true clusters ``sample``, both represented as membership vectors, this functions returns an indicator whether each true cluster matches a predicted cluster. This is a pandas Series indexed by true cluster identifier and with values corresponding to 0 or 1, depending on whether or not the true cluster matches a predicted cluster.

    Args:
        prediction (Series): Membership vector representing a predicted disambiguation.
        sample (Series): Membership vector representing a set of true clusters.

    Returns:
        Series: Pandas Series indexed by true cluster identifiers (unique values in `sample`) and with values corresponding to the error indicator.

    Examples:
        >>> prediction = pd.Series(index=[1,2,3,4,5,6,7,8], data=[1,1,2,3,2,4,4,5])
        >>> sample = pd.Series(index=[1,2,3,4,5,8], data=["c1", "c1", "c1", "c2", "c2", "c4"])
        >>> error_indicator(prediction, sample)
        sample
        c1    1
        c2    1
        c4    0
        Name: error_indicator, dtype: int64

    """
    prediction = MembershipVector(prediction)
    sample = MembershipVector(sample)

    relevant_predictions = relevant_prediction_subset(prediction, sample)

    outer = pd.concat(
        {"prediction": relevant_predictions, "sample": sample},
        axis=1,
        copy=False,
        join="outer",
    )

    def lambd(sample_cluster):
        p = pd.value_counts(sample_cluster)
        u = outer.prediction.value_counts()[p.index].values

        if len(p) == 1 and p.values[0] == sum(u):
            return 0
        else:
            return 1

    result = outer.groupby("sample").agg(lambd).prediction
    result.rename("error_indicator", inplace=True)

    return result


def splitting_entropy(prediction, sample, alpha=1):
    r"""
    Splitting entropy of true clusters.

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
    prediction = MembershipVector(prediction)
    sample = MembershipVector(sample)

    relevant_predictions = relevant_prediction_subset(prediction, sample)

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
