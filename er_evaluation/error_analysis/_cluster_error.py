import numpy as np
import pandas as pd
from scipy.special import comb

from er_evaluation.data_structures import MembershipVector
from er_evaluation.error_analysis._record_error import (
    error_indicator_from_table, error_metrics_from_table,
    expected_extra_from_table, expected_missing_from_table,
    expected_relative_extra_from_table, expected_relative_missing_from_table,
    expected_size_difference_from_table, record_error_table)
from er_evaluation.utils import relevant_prediction_subset


def error_metrics(prediction, sample):
    """
    Compute canonical set of error metrics from record error table.

    Error metrics included:

    * Expected extra links (see :meth:`er_evaluation.error_analysis.expected_extra`)
    * Expected relative extra links (see :meth:`er_evaluation.error_analysis.expected_relative_extra`)
    * Expected missin elements (see :meth:`er_evaluation.error_analysis.expected_missing`)
    * Expected relative missin elements (see :meth:`er_evaluation.error_analysis.expected_relative_missing`)
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
        expected_extra	expected_relative_extra	expected_missing	expected_relative_missing	error_indicator
        reference
        c1	0.333333	0.166667	1.333333	0.444444	1
        c2	0.500000	0.250000	1.000000	0.500000	1
        c3	1.000000	0.333333	0.000000	0.000000	0

    Notes:
        The sample is restricted to the set of records which are present in the prediction.
    """
    prediction = MembershipVector(prediction, dropna=True)
    sample = MembershipVector(sample, dropna=True)

    sample = sample[sample.index.isin(prediction.index)]

    error_table = record_error_table(prediction, sample)
    return error_metrics_from_table(error_table)


def count_extra(prediction, sample):
    r"""
    Count the number of extraneous elements in sampled clusters.

    Given a predicted disambiguation ``prediction`` and a sample of true clusters ``sample``, both represented as membership vectors, this functions returns the count of extraneous elements for each true cluster. This is a pandas Series indexed by true cluster identifier and with values corresponding to the counts of extraneous elements.

    Count of extraneous elements
        For a given sampled cluster :math:`c` with records :math:`r \in c`, let :math:`A_r` be the set of records which are erroneously linked to :math:`r` in the predicted clustering. That is, if :math:`\hat c(r)` is the predicted cluster containing :math:`r`, then :math:`A_r = \hat c(r) \backslash c` Then the count of extraneous elements for :math:`c` is

        .. math::

            E_{\text{count_extra}}(c) = \sum_{r\in c} \lvert A_r \rvert.

    Args:
        prediction (Series): Membership vector representing a predicted disambiguation.
        sample (Series): Membership vector representing a set of true clusters.

    Returns:
        Series: Pandas Series indexed by true cluster identifiers (unique values in `sample`) and with values corresponding to the count of extraneous elements.

    Examples:
        >>> prediction = pd.Series(index=[1,2,3,4,5,6,7,8], data=[1,1,2,3,2,4,4,4])
        >>> sample = pd.Series(index=[1,2,3,4,5,8], data=["c1", "c1", "c1", "c2", "c2", "c4"])
        >>> count_extra(prediction, sample)
        reference
        c1    1
        c2    1
        c4    2
        Name: count_extra, dtype: int64

    Notes:
        The sample is restricted to the set of records which are present in the prediction.
    """
    prediction = MembershipVector(prediction, dropna=True)
    sample = MembershipVector(sample, dropna=True)

    sample = sample[sample.index.isin(prediction.index)]

    relevant_predictions = relevant_prediction_subset(prediction, sample)

    outer = pd.concat(
        {"prediction": relevant_predictions, "reference": sample},
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

    result = outer.groupby("reference").agg(lambd).prediction
    result.rename("count_extra", inplace=True)

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

    Notes:
        The sample is restricted to the set of records which are present in the prediction.
    """
    prediction = MembershipVector(prediction, dropna=True)
    sample = MembershipVector(sample, dropna=True)

    sample = sample[sample.index.isin(prediction.index)]

    error_table = record_error_table(prediction, sample)

    return expected_size_difference_from_table(error_table)


def expected_extra(prediction, sample):
    r"""
    Expected number of extraneous elements in sampled clusters.

    Given a predicted disambiguation ``prediction`` and a sample of true clusters ``sample``, both represented as membership vectors, this functions returns the expected number of extraneous elements for each true cluster. This is a pandas Series indexed by true cluster identifier and with values corresponding to the expected number of extraneous elements.

    Expected Number of extraneous elements
        For a given sampled cluster :math:`c` with records :math:`r \in c`, let :math:`A_r` be the set of records which are erroneously linked to :math:`r` in the predicted clustering. That is, if :math:`\hat c(r)` is the predicted cluster containing :math:`r`, then :math:`A_r = \hat c(r) \backslash c` Then the expected number of extraneous elements for :math:`c` is

        .. math::

            E_{\text{extra}}(c) = \frac{1}{\lvert c \rvert}\sum_{r\in c} \lvert A_r \rvert.

        This is the expected number of erroneous links to a random record :math:`r \in c`.

    Args:
        prediction (Series): Membership vector representing a predicted disambiguation.
        sample (Series): Membership vector representing a set of true clusters.

    Returns:
        Series: Pandas Series indexed by true cluster identifiers (unique values in `sample`) and with values corresponding to the expected number of extraneous elements.

    Examples:
        >>> prediction = pd.Series(index=[1,2,3,4,5,6,7,8], data=[1,1,2,3,2,4,4,4])
        >>> sample = pd.Series(index=[1,2,3,4,5,8], data=["c1", "c1", "c1", "c2", "c2", "c4"])
        >>> expected_extra(prediction, sample)
        reference
        c1    0.333333
        c2    0.500000
        c4    2.000000
        Name: expected_extra, dtype: float64

    Notes:
        The sample is restricted to the set of records which are present in the prediction.
    """
    prediction = MembershipVector(prediction, dropna=True)
    sample = MembershipVector(sample, dropna=True)

    error_table = record_error_table(prediction, sample)
    return expected_extra_from_table(error_table)


def expected_relative_extra(prediction, sample):
    r"""
    Expected relative number of extraneous elements in sampled clusters.

    Given a predicted disambiguation ``prediction`` and a sample of true clusters ``sample``, both represented as membership vectors, this functions returns the expected number of relative extraneous elements for each true cluster. This is a pandas Series indexed by true cluster identifier and with values corresponding to the expected number of relative extraneous elements.

    Expected Relative Number of extraneous elements
        For a given sampled cluster :math:`c` with records :math:`r \in c`, let :math:`A_r` be the set of records which are erroneously linked to :math:`r` in the predicted clustering. That is, if :math:`\hat c(r)` is the predicted cluster containing :math:`r`, then :math:`A_r = \hat c(r) \backslash c` Then the expected number of extraneous elements for :math:`c` is

        .. math::

            E_{\text{rel_extra}}(c) = \frac{1}{\lvert c \rvert}\sum_{r\in c} \lvert A_r \rvert / \lvert \hat c(r) \rvert.

        This is the expected relative number of erroneous links to a random record :math:`r \in c`.

    Args:
        prediction (Series): Membership vector representing a predicted disambiguation.
        sample (Series): Membership vector representing a set of true clusters.

    Returns:
        Series: Pandas Series indexed by true cluster identifiers (unique values in `sample`) and with values corresponding to the expected number of extraneous elements.

    Examples:
        >>> prediction = pd.Series(index=[1,2,3,4,5,6,7,8], data=[1,1,2,3,2,4,4,4])
        >>> sample = pd.Series(index=[1,2,3,4,5,8], data=["c1", "c1", "c1", "c2", "c2", "c4"])
        >>> expected_relative_extra(prediction, sample)
        reference
        c1    0.166667
        c2    0.250000
        c4    0.666667
        Name: expected_relative_extra, dtype: float64

    Notes:
        The sample is restricted to the set of records which are present in the prediction.
    """
    prediction = MembershipVector(prediction, dropna=True)
    sample = MembershipVector(sample, dropna=True)

    error_table = record_error_table(prediction, sample)
    return expected_relative_extra_from_table(error_table)


def count_missing(prediction, sample):
    r"""
    Count the number of missing elements in sampled clusters.

    Given a predicted disambiguation ``prediction`` and a sample of true clusters ``sample``, both represented as membership vectors, this functions returns the count of missin elements for each true cluster. This is a pandas Series indexed by true cluster identifier and with values corresponding to the counts of missin elements.

    Count of missin elements
        For a given sampled cluster :math:`c` with records :math:`r \in c`, let :math:`B_r` be the set of records which are missing from the predicted cluster containing :math:`r`. That is, if :math:`\hat c(r)` is the predicted cluster containing :math:`r`, then :math:`B_r = c \backslash \hat c(r)`. Then the count of missin elements for :math:`c` is

        .. math::

            E_{\text{count_miss}}(c) = \sum_{r\in c} \lvert B_r \rvert.

    Args:
        prediction (Series): Membership vector representing a predicted disambiguation.
        sample (Series): Membership vector representing a set of true clusters.

    Returns:
        Series: Pandas Series indexed by true cluster identifiers (unique values in `sample`) and with values corresponding to the count of extraneous elements.

    Examples:
        >>> prediction = pd.Series(index=[1,2,3,4,5,6,7,8], data=[1,1,2,3,2,4,4,4])
        >>> sample = pd.Series(index=[1,2,3,4,5,8], data=["c1", "c1", "c1", "c2", "c2", "c4"])
        >>> count_missing(prediction, sample)
        reference
        c1    4
        c2    2
        c4    0
        Name: count_missing, dtype: int64

    Notes:
        The sample is restricted to the set of records which are present in the prediction.
    """
    prediction = MembershipVector(prediction, dropna=True)
    sample = MembershipVector(sample, dropna=True)

    sample = sample[sample.index.isin(prediction.index)]

    relevant_predictions = relevant_prediction_subset(prediction, sample)

    outer = pd.concat(
        {"prediction": relevant_predictions, "reference": sample},
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

    result = outer.groupby("reference").agg(lambd).prediction
    result.rename("count_missing", inplace=True)

    return result


def expected_missing(prediction, sample):
    r"""
    Expected number of missing elements in sampled clusters.

    Given a predicted disambiguation ``prediction`` and a sample of true clusters ``sample``, both represented as membership vectors, this functions returns the expected relative number of missin elements for each true cluster. This is a pandas Series indexed by true cluster identifier and with values corresponding to the expected number of missin elements.

    Expected Number of missin elements
        For a given sampled cluster  :math:`c` with records  :math:`r \in c`, let :math:`B_r` be the set of records which are missing from the predicted cluster containing :math:`r`. That is, if :math:`\hat c(r)` is the predicted cluster containing :math:`r`, then :math:`B_r = c \backslash \hat c(r)`. Then the expected number of missin elements for

        .. math::

            E_{\text{miss}}(c) = \frac{1}{\lvert c \rvert}\sum_{r\in c} \lvert B_r \rvert.

    Args:
        prediction (Series): Membership vector representing a predicted disambiguation.
        sample (Series): Membership vector representing a set of true clusters.

    Returns:
        Series: Pandas Series indexed by true cluster identifiers (unique values in `sample`) and with values corresponding to the expected number of missin elements.

    Examples:
        >>> prediction = pd.Series(index=[1,2,3,4,5,6,7,8], data=[1,1,2,3,2,4,4,4])
        >>> sample = pd.Series(index=[1,2,3,4,5,8], data=["c1", "c1", "c1", "c2", "c2", "c4"])
        >>> expected_missing(prediction, sample)
        reference
        c1    1.333333
        c2    1.000000
        c4    0.000000
        Name: expected_missing, dtype: float64

    Notes:
        The sample is restricted to the set of records which are present in the prediction.
    """
    prediction = MembershipVector(prediction, dropna=True)
    sample = MembershipVector(sample, dropna=True)

    error_table = record_error_table(prediction, sample)
    return expected_missing_from_table(error_table)


def expected_relative_missing(prediction, sample):
    r"""
    Expected relative number of missing elements in sampled clusters.

    Given a predicted disambiguation ``prediction`` and a sample of true clusters ``sample``, both represented as membership vectors, this functions returns the expected number of missin elements for each true cluster. This is a pandas Series indexed by true cluster identifier and with values corresponding to the expected relative number of missin elements.

    Expected Relative Number of missin elements
        For a given sampled cluster  :math:`c` with records  :math:`r \in c`, let :math:`B_r` be the set of records which are missing from the predicted cluster containing :math:`r`. That is, if :math:`\hat c(r)` is the predicted cluster containing :math:`r`, then :math:`B_r = c \backslash \hat c(r)`. Then the expected number of missin elements for :math:`c` is

        .. math::

            E_{\text{rel_miss}}(c) = \frac{1}{\lvert c \rvert}\sum_{r\in c} \lvert B_r \rvert / \lvert c \rvert.

    Args:
        prediction (Series): Membership vector representing a predicted disambiguation.
        sample (Series): Membership vector representing a set of true clusters.

    Returns:
        Series: Pandas Series indexed by true cluster identifiers (unique values in `sample`) and with values corresponding to the expected relative number of missin elements.

    Examples:
        >>> prediction = pd.Series(index=[1,2,3,4,5,6,7,8], data=[1,1,2,3,2,4,4,4])
        >>> sample = pd.Series(index=[1,2,3,4,5,8], data=["c1", "c1", "c1", "c2", "c2", "c4"])
        >>> expected_relative_missing(prediction, sample)
        reference
        c1    0.444444
        c2    0.500000
        c4    0.000000
        Name: expected_relative_missing, dtype: float64

    Notes:
        The sample is restricted to the set of records which are present in the prediction.
    """
    prediction = MembershipVector(prediction, dropna=True)
    sample = MembershipVector(sample, dropna=True)

    error_table = record_error_table(prediction, sample)
    return expected_relative_missing_from_table(error_table)


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
        reference
        c1    1
        c2    1
        c4    0
        Name: error_indicator, dtype: int64

    Notes:
        The sample is restricted to the set of records which are present in the prediction.
    """
    prediction = MembershipVector(prediction, dropna=True)
    sample = MembershipVector(sample, dropna=True)

    error_table = record_error_table(prediction, sample)
    return error_indicator_from_table(error_table)


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
        reference
        c1    1.889882
        c2    2.000000
        c4    1.000000
        Name: splitting_entropy_1, dtype: float64

    Notes:
        The sample is restricted to the set of records which are present in the prediction.
    """
    prediction = MembershipVector(prediction, dropna=True)
    sample = MembershipVector(sample, dropna=True)

    sample = sample[sample.index.isin(prediction.index)]

    relevant_predictions = relevant_prediction_subset(prediction, sample)

    outer = pd.concat(
        {"prediction": relevant_predictions, "reference": sample},
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

    result = outer.groupby("reference").agg(lambd).prediction
    result.rename(f"splitting_entropy_{alpha}", inplace=True)

    return result
