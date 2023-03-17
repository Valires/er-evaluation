import pandas as pd

from er_evaluation.data_structures import MembershipVector
from er_evaluation.utils import relevant_prediction_subset


def error_metrics_from_table(error_table):
    """
    Compute canonical set of error metrics from record error table.

    Error metrics included:

    * Expected extra elements (see :meth:`er_evaluation.error_analysis.expected_extra`)
    * Expected relative extra elements (see :meth:`er_evaluation.error_analysis.expected_relative_extra`)
    * Expected missin elements (see :meth:`er_evaluation.error_analysis.expected_missing`)
    * Expected relative missin elements (see :meth:`er_evaluation.error_analysis.expected_relative_missing`)
    * Error indicator (see :meth:`er_evaluation.error_analysis.error_indicator`)

    Args:
        error_table (DataFrame): Record error table. See :meth:`er_evaluation.error_analysis.record_error_table`.

    Returns:
        DataFrame: Dataframe indexed by cluster identifiers and with values corresponding to error metrics.

    Examples
        >>> prediction = pd.Series(index=[1,2,3,4,5,6,7,8], data=[1,1,2,3,2,4,4,4])
        >>> sample = pd.Series(index=[1,2,3,4,5,6,7, 8], data=["c1", "c1", "c1", "c2", "c2", "c3", "c3", "c3"])
        >>> error_table = record_error_table(prediction, sample)
        >>> error_metrics_from_table(prediction, sample)  # doctest: +SKIP
        expected_extra	expected_relative_extra	expected_missing	expected_relative_missing	error_indicator
        reference
        c1	0.333333	0.166667	1.333333	0.444444	1
        c2	0.500000	0.250000	1.000000	0.500000	1
        c3	1.000000	0.333333	0.000000	0.000000	0
    """
    return pd.concat(
        [
            expected_extra_from_table(error_table),
            expected_relative_extra_from_table(error_table),
            expected_missing_from_table(error_table),
            expected_relative_missing_from_table(error_table),
            error_indicator_from_table(error_table),
        ],
        axis=1,
    )


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
        >>> record_error_table(prediction, sample)  # doctest: +SKIP
                    prediction	reference	pred_cluster_size	ref_cluster_size	extra	    missing
        index
        1	    1	        c1	        2	                3.0	                0.0             1.0
        2	    1	        c1	        2	                3.0	                0.0	            1.0
        3	    2	        c1	        2	                3.0	                1.0	            2.0
        4	    3	        c2	        1	                2.0	                0.0	            1.0
        5	    2	        c2	        2	                2.0	                1.0	            1.0
        6	    4	        c3	        3	                2.0	                1.0	            0.0
        7	    4	        c3	        3	                2.0	                1.0	            0.0

    Notes:
        sample is subsetted to only include indices present in prediction.
    """
    prediction = MembershipVector(prediction, dropna=True)
    sample = MembershipVector(sample, dropna=True)

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
    error_table.dropna(inplace=True)

    intersection_size = error_table.groupby(["prediction", "reference"]).size()
    intersection_size.name = "intersection_size"
    error_table = error_table.merge(intersection_size.reset_index(), on=["prediction", "reference"], how="left")
    error_table["extra"] = error_table["pred_cluster_size"] - error_table["intersection_size"]
    error_table["missing"] = error_table["ref_cluster_size"] - error_table["intersection_size"]

    error_table.drop(labels="intersection_size", axis=1, inplace=True)

    return error_table


def expected_size_difference_from_table(error_table):
    """
    Compute expected size difference from record error table.

    See :meth:`er_evaluation.error_analysis.expected_size_difference`.

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

        The result is the same as calling :meth:`er_evaluation.error_analysis.expected_size_difference` on prediction and sample:

        >>> from er_evaluation.error_analysis import expected_size_difference
        >>> expected_size_difference(prediction, sample)
        reference
        c1   -1.0
        c2   -0.5
        c3    1.0
        Name: expected_size_diff, dtype: float64
    """
    result = expected_extra_from_table(error_table) - expected_missing_from_table(error_table)
    result.name = "expected_size_diff"
    return result


def expected_extra_from_table(error_table):
    """
    Compute expected extra elements from record error table.

    See :meth:`er_evaluation.error_analysis.expected_extra`.

    Args:
        error_table (DataFrame): Record error table.

    Returns:
        Series: Expected extra elements for each reference cluster.

    Examples:
        >>> prediction = pd.Series(index=[1,2,3,4,5,6,7,8], data=[1,1,2,3,2,4,4,4])
        >>> sample = pd.Series(index=[1,2,3,4,5,6,7], data=["c1", "c1", "c1", "c2", "c2", "c3", "c3"])
        >>> error_table = record_error_table(prediction, sample)
        >>> expected_extra_from_table(error_table)
        reference
        c1    0.333333
        c2    0.500000
        c3    1.000000
        Name: expected_extra, dtype: float64

        The result is the same as calling :meth:`er_evaluation.error_analysis.expected_extra` directly on ``prediction`` and ``sample``:

        >>> from er_evaluation.error_analysis import expected_extra
        >>> expected_extra(prediction, sample)
        sample
        c1    0.333333
        c2    0.500000
        c3    1.000000
        Name: expected_extra, dtype: float64
    """
    error_table = error_table.copy()
    result = error_table.groupby("reference").agg({"extra": "mean"})
    return result["extra"].rename("expected_extra")


def expected_missing_from_table(error_table):
    """
    Compute expected missin elements from record error table.

    See :meth:`er_evaluation.error_analysis.expected_missing`.

    Args:
        error_table (DataFrame): Record error table.

    Returns:
        Series: Expected missin elements for each reference cluster.

    Examples:
        >>> prediction = pd.Series(index=[1,2,3,4,5,6,7,8], data=[1,1,2,3,2,4,4,4])
        >>> sample = pd.Series(index=[1,2,3,4,5,6,7], data=["c1", "c1", "c1", "c2", "c2", "c3", "c3"])
        >>> error_table = record_error_table(prediction, sample)
        >>> expected_missing_from_table(error_table)
        reference
        c1    1.333333
        c2    1.000000
        c3    0.000000
        Name: expected_missing, dtype: float64

        The result is the same as calling :meth:`er_evaluation.error_analysis.expected_missing` directly on ``prediction`` and ``sample``:

        >>> from er_evaluation.error_analysis import expected_missing
        >>> expected_missing(prediction, sample)
        sample
        c1    1.333333
        c2    1.000000
        c3    0.000000
        Name: expected_missing, dtype: float64
    """
    error_table = error_table.copy()
    result = error_table.groupby("reference").agg({"missing": "mean"})
    return result["missing"].rename("expected_missing")


def expected_relative_extra_from_table(error_table):
    """
    Compute expected relative extra elements from record error table.

    See :meth:`er_evaluation.error_analysis.expected_relative_extra`.

    Args:
        error_table (DataFrame): Record error table.

    Returns:
        Series: Expected relative extra elements for each reference cluster.

    Examples:
        >>> prediction = pd.Series(index=[1,2,3,4,5,6,7,8], data=[1,1,2,3,2,4,4,4])
        >>> sample = pd.Series(index=[1,2,3,4,5,6,7], data=["c1", "c1", "c1", "c2", "c2", "c3", "c3"])
        >>> error_table = record_error_table(prediction, sample)
        >>> expected_relative_extra_from_table(error_table)
        reference
        c1    0.166667
        c2    0.250000
        c3    0.333333
        Name: expected_relative_extra, dtype: float64

        The result is the same as calling :meth:`er_evaluation.error_analysis.expected_relative_extra` directly on ``prediction`` and ``sample``:

        >>> from er_evaluation.error_analysis import expected_relative_extra
        >>> expected_relative_extra(prediction, sample)
        sample
        c1    0.166667
        c2    0.250000
        c3    0.333333
        Name: expected_relative_extra, dtype: float64
    """
    error_table = error_table.copy()
    error_table["expected_relative_extra"] = error_table["extra"] / error_table["pred_cluster_size"]
    result = error_table.groupby("reference").agg({"expected_relative_extra": "mean"})
    return result["expected_relative_extra"]


def expected_relative_missing_from_table(error_table):
    """
    Compute expected relative missin elements from record error table.

    See :meth:`er_evaluation.error_analysis.expected_relative_missing`.

    Args:
        error_table (DataFrame): Record error table.

    Returns:
        Series: Expected relative missin elements for each reference cluster.

    Examples:
        >>> prediction = pd.Series(index=[1,2,3,4,5,6,7,8], data=[1,1,2,3,2,4,4,4])
        >>> sample = pd.Series(index=[1,2,3,4,5,6,7], data=["c1", "c1", "c1", "c2", "c2", "c3", "c3"])
        >>> error_table = record_error_table(prediction, sample)
        >>> expected_relative_missing_from_table(error_table)
        reference
        c1    0.444444
        c2    0.500000
        c3    0.000000
        Name: expected_relative_missing, dtype: float64

        The result is the same as calling :meth:`er_evaluation.error_analysis.expected_relative_missing` directly on ``prediction`` and ``sample``:

        >>> from er_evaluation.error_analysis import expected_relative_missing
        >>> expected_relative_missing(prediction, sample)
        reference
        c1    0.444444
        c2    0.500000
        c3    0.000000
        Name: expected_relative_missing, dtype: float64

    """
    error_table = error_table.copy()
    error_table["expected_relative_missing"] = error_table["missing"] / error_table["ref_cluster_size"]
    result = error_table.groupby("reference").agg({"expected_relative_missing": "mean"})
    return result["expected_relative_missing"]


def error_indicator_from_table(error_table):
    """
    Compute error indicator from record error table.

    See :meth:`er_evaluation.error_analysis.error_indicator`.

    Args:
        error_table (DataFrame): Record error table.

    Returns:
        Series: Error indicator for each reference cluster.

    Examples:
        >>> prediction = pd.Series(index=[1,2,3,4,5,6,7,8], data=[1,1,2,3,2,4,4,4])
        >>> sample = pd.Series(index=[1,2,3,4,5,6,7, 8], data=["c1", "c1", "c1", "c2", "c2", "c3", "c3", "c3"])
        >>> error_table = record_error_table(prediction, sample)
        >>> error_indicator_from_table(error_table)
        reference
        c1    1
        c2    1
        c3    0
        Name: error_indicator, dtype: int64

        The result is the same as calling :meth:`er_evaluation.error_analysis.error_indicator` directly on ``prediction`` and ``sample``:

        >>> from er_evaluation.error_analysis import error_indicator
        >>> error_indicator(prediction, sample)
        reference
        c1    1
        c2    1
        c3    0
        Name: error_indicator, dtype: int64
    """
    error_table = error_table.copy()
    error_table["error_indicator"] = (error_table["extra"] != 0) | (error_table["missing"] != 0)
    error_table["error_indicator"] = error_table["error_indicator"].astype(int)
    result = error_table.groupby("reference").agg({"error_indicator": "first"})
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
