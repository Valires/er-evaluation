"""
===========================
Analyse record-level errors
===========================

The **error_analysis.record_error** submodule provides a set of tools to analyze errors expressed at the record level.

For each record, it is assumed that we know the following quantities:

1. **pred_cluster_size**: The size of the predicted cluster which contains the record.
2. **ref_cluster_size**: The size of the true cluster which contains the record.
3. **extra_links**: The number of elements in the predicted cluster which are not in the true cluster.
4. **missing_links**: The number of elements in the true cluster which are not in the predicted cluster.

These four quantities, together with the record index, the predicted cluster ID, and the true cluster ID, are stored in what is called the **record error table**. The record error table can be computed using the :func:`record_error_table` function, given a sample of ground truth clusters and a prediction.

From the record error table, the same error metrics as in the :mod:`error_analysis.cluster_error` module can be computed. The functions :func:`expected_size_difference_from_table`, :func:`expected_extra_links_from_table`, :func:`expected_missing_links_from_table`, :func:`expected_relative_extra_links_from_table`, :func:`expected_relative_missing_links_from_table`, and :func:`error_indicator_from_table` compute the same metrics as in the :mod:`error_analysis.cluster_error` module, but from the record error table instead of from the prediction and the sample.

The key advantage of working with the record error table is that it allows sensitivity analyses to be performed. Since all cluster error metrics and representative performance estimators can be computed directly from the record error table, uncertainty regarding error rates can be propagated from the record error table into cluster error metrics and into performance estimates.
"""

import pandas as pd
import numpy as np
from scipy.special import comb

from er_evaluation.utils import relevant_prediction_subset
from er_evaluation.data_structures import membership_to_clusters


def record_error_table(prediction, sample):
    """
    Compute record error table.

    Args:
        prediction (Series): Membership vector representation of a clustering.
        sample (Series): Membership vector representation of a clustering.

    Returns:
        DataFrame: Record error table.

    Examples:
        >>> prediction = pd.Series(index=[1,2,3,4,5,6,7,8], data=[1,1,2,3,2,4,4,4])
        >>> sample = pd.Series(index=[1,2,3,4,5,6,7], data=["c1", "c1", "c1", "c2", "c2", "c3", "c3"])
        >>> record_error_table(prediction, sample)  # doctest: +NORMALIZE_WHITESPACE
                    prediction	reference	pred_cluster_size	ref_cluster_size	extra_links	    missing_links
        index
        1	    1	        c1	        2	                3.0	                0.0             1.0
        2	    1	        c1	        2	                3.0	                0.0	            1.0
        3	    2	        c1	        2	                3.0	                1.0	            2.0
        4	    3	        c2	        1	                2.0	                0.0	            1.0
        5	    2	        c2	        2	                2.0	                1.0	            1.0
        6	    4	        c3	        3	                2.0	                1.0	            0.0
        7	    4	        c3	        3	                2.0	                1.0	            0.0
        8	    4	        NaN	        3	                NaN	                NaN	            NaN
    """
    sample = sample[sample.index.isin(prediction.index)]
    prediction = relevant_prediction_subset(prediction, sample)

    pred_cluster_size = prediction.value_counts()
    pred_cluster_size.index.name = "prediction"
    pred_cluster_size = pred_cluster_size.reset_index(name="pred_cluster_size")

    ref_cluster_size = sample.value_counts()
    ref_cluster_size.index.name = "reference"
    ref_cluster_size = ref_cluster_size.reset_index(name="ref_cluster_size")

    prediction.index.name = "index"
    sample.index.name = "index"
    error_table = (
        prediction.reset_index(name="prediction")
        .merge(sample.reset_index(name="reference"), how="left")
        .merge(pred_cluster_size, on="prediction", how="left")
        .merge(ref_cluster_size, on="reference", how="left")
    )
    error_table.set_index("index", inplace=True)

    pred_clusters = membership_to_clusters(prediction)
    ref_clusters = membership_to_clusters(sample)

    def A_r(row):
        if pd.isna(row.reference):
            A_r = np.nan
        else:
            A_r = len(
                np.setdiff1d(
                    pred_clusters[row.prediction], ref_clusters[row.reference]
                )
            )
        return A_r

    def B_r(row):
        if pd.isna(row.reference):
            A_r = np.nan
        else:
            A_r = len(
                np.setdiff1d(
                    ref_clusters[row.reference], pred_clusters[row.prediction]
                )
            )
        return A_r

    error_table["extra_links"] = error_table.apply(A_r, axis=1)
    error_table["missing_links"] = error_table.apply(B_r, axis=1)

    return error_table


def expected_size_difference_from_table(error_table):
    """
    Compute expected size difference from record error table.

    Args:
        error_table (DataFrame): Record error table.

    Returns:
        Series: Expected size difference for each reference cluster.

    Examples:
        >>> prediction = pd.Series(index=[1,2,3,4,5,6,7,8], data=[1,1,2,3,2,4,4,4])
        >>> sample = pd.Series(index=[1,2,3,4,5,6,7], data=["c1", "c1", "c1", "c2", "c2", "c3", "c3"])
        >>> error_table = record_error_table(prediction, sample)
        >>> expected_size_difference_from_table(error_table)
        reference
        c1   -1.0
        c2   -0.5
        c3    1.0
        Name: expected_size_diff, dtype: float64
    """
    error_table = error_table.copy()
    error_table["expected_size_diff"] = (
        error_table["pred_cluster_size"] - error_table["ref_cluster_size"]
    )
    result = error_table.groupby("reference").agg(
        {"expected_size_diff": "mean"}
    )

    return result["expected_size_diff"]


def expected_extra_links_from_table(error_table):
    """
    Compute expected extra links from record error table.

    Args:
        error_table (DataFrame): Record error table.

    Returns:
        Series: Expected extra links for each reference cluster.

    Examples:
        >>> prediction = pd.Series(index=[1,2,3,4,5,6,7,8], data=[1,1,2,3,2,4,4,4])
        >>> sample = pd.Series(index=[1,2,3,4,5,6,7], data=["c1", "c1", "c1", "c2", "c2", "c3", "c3"])
        >>> error_table = record_error_table(prediction, sample)
        >>> expected_extra_links_from_table(error_table)
        reference
        c1    0.333333
        c2    0.500000
        c3    1.000000
        Name: expected_extra_links, dtype: float64
    """
    error_table = error_table.copy()
    result = error_table.groupby("reference").agg({"extra_links": "mean"})
    return result["extra_links"].rename("expected_extra_links")


def expected_missing_links_from_table(error_table):
    """
    Compute expected missing links from record error table.

    Args:
        error_table (DataFrame): Record error table.

    Returns:
        Series: Expected missing links for each reference cluster.

    Examples:
        >>> prediction = pd.Series(index=[1,2,3,4,5,6,7,8], data=[1,1,2,3,2,4,4,4])
        >>> sample = pd.Series(index=[1,2,3,4,5,6,7], data=["c1", "c1", "c1", "c2", "c2", "c3", "c3"])
        >>> error_table = record_error_table(prediction, sample)
        >>> expected_missing_links_from_table(error_table)
        reference
        c1    1.333333
        c2    1.000000
        c3    0.000000
        Name: expected_missing_links, dtype: float64
    """
    error_table = error_table.copy()
    result = error_table.groupby("reference").agg({"missing_links": "mean"})
    return result["missing_links"].rename("expected_missing_links")


def expected_relative_extra_links_from_table(error_table):
    """
    Compute expected relative extra links from record error table.

    Args:
        error_table (DataFrame): Record error table.

    Returns:
        Series: Expected relative extra links for each reference cluster.

    Examples:
        >>> prediction = pd.Series(index=[1,2,3,4,5,6,7,8], data=[1,1,2,3,2,4,4,4])
        >>> sample = pd.Series(index=[1,2,3,4,5,6,7], data=["c1", "c1", "c1", "c2", "c2", "c3", "c3"])
        >>> error_table = record_error_table(prediction, sample)
        >>> expected_relative_extra_links_from_table(error_table)
        reference
        c1    0.166667
        c2    0.250000
        c3    0.333333
        Name: expected_relative_extra_links, dtype: float64
    """
    error_table = error_table.copy()
    error_table["expected_relative_extra_links"] = (
        error_table["extra_links"] / error_table["pred_cluster_size"]
    )
    result = error_table.groupby("reference").agg(
        {"expected_relative_extra_links": "mean"}
    )
    return result["expected_relative_extra_links"]


def expected_relative_missing_links_from_table(error_table):
    """
    Compute expected relative missing links from record error table.

    Args:
        error_table (DataFrame): Record error table.

    Returns:
        Series: Expected relative missing links for each reference cluster.

    Examples:
        >>> prediction = pd.Series(index=[1,2,3,4,5,6,7,8], data=[1,1,2,3,2,4,4,4])
        >>> sample = pd.Series(index=[1,2,3,4,5,6,7], data=["c1", "c1", "c1", "c2", "c2", "c3", "c3"])
        >>> error_table = record_error_table(prediction, sample)
        >>> expected_relative_missing_links_from_table(error_table)
        reference
        c1    0.444444
        c2    0.500000
        c3    0.000000
        Name: expected_relative_missing_links, dtype: float64
    """
    error_table = error_table.copy()
    error_table["expected_relative_missing_links"] = (
        error_table["missing_links"] / error_table["ref_cluster_size"]
    )
    result = error_table.groupby("reference").agg(
        {"expected_relative_missing_links": "mean"}
    )
    return result["expected_relative_missing_links"]


def error_indicator_from_table(error_table):
    """
    Compute error indicator from record error table.

    Args:
        error_table (DataFrame): Record error table.

    Returns:
        Series: Error indicator for each reference cluster.

    Examples:
        >>> prediction = pd.Series(index=[1,2,3,4,5,6,7,8], data=[1,1,2,3,2,4,4,4])
        >>> sample = pd.Series(index=[1,2,3,4,5,6,7], data=["c1", "c1", "c1", "c2", "c2", "c3", "c3"])
        >>> error_table = record_error_table(prediction, sample)
        >>> error_indicator_from_table(error_table)
        reference
        c1    0.0
        c2    0.0
        c3    0.0
        Name: error_indicator, dtype: float64
    """
    error_table = error_table.copy()
    error_table["error_indicator"] = (error_table["extra_links"] == 0) & (
        error_table["missing_links"] == 0
    )
    result = error_table.groupby("reference").agg({"error_indicator": "mean"})
    return result["error_indicator"]


def cluster_sizes_from_table(error_table):
    """
    Compute cluster sizes from record error table.

    Args:
        error_table (DataFrame): Record error table.

    Returns:
        Series: Cluster sizes for each reference cluster.

    Examples:
        >>> prediction = pd.Series(index=[1,2,3,4,5,6,7,8], data=[1,1,2,3,2,4,4,4])
        >>> sample = pd.Series(index=[1,2,3,4,5,6,7], data=["c1", "c1", "c1", "c2", "c2", "c3", "c3"])
        >>> error_table = record_error_table(prediction, sample)
        >>> cluster_sizes_from_table(error_table)
        reference
        c1    3.0
        c2    2.0
        c3    2.0
        Name: ref_cluster_size, dtype: float64
    """
    error_table = error_table.copy()
    result = error_table.groupby("reference").agg({"ref_cluster_size": "mean"})
    return result["ref_cluster_size"]


def pred_cluster_sizes_from_table(error_table):
    """
    Compute predicted cluster sizes from record error table.

    Args:
        error_table (DataFrame): Record error table.

    Returns:
        Series: Predicted cluster sizes for each reference cluster.

    Examples:
        >>> prediction = pd.Series(index=[1,2,3,4,5,6,7,8], data=[1,1,2,3,2,4,4,4])
        >>> sample = pd.Series(index=[1,2,3,4,5,6,7], data=["c1", "c1", "c1", "c2", "c2", "c3", "c3"])
        >>> error_table = record_error_table(prediction, sample)
        >>> pred_cluster_sizes_from_table(error_table)
        reference
        c1    2.0
        c2    1.5
        c3    3.0
        Name: pred_cluster_size, dtype: float64
    """
    error_table = error_table.copy()
    result = error_table.groupby("reference").agg({"pred_cluster_size": "mean"})
    return result["pred_cluster_size"]
