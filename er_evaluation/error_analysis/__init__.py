r"""
==============
Error Analysis
==============

The **error_analysis** module provides tools to analyze errors, given a set of ground truth clusters. These ground truth clusters may correspond to a benchmark dataset which is *complete* (all of the entities within it are fully resolved and have no missing links), or to a probability sample of ground truth clusters.

The key assumptions used for this module are:

1. A *predicted* clustering is available as a membership vector (named  `prediction` throughout).
2. A set of ground truth clusters is available as a membership vector (named `sample` throughout).

Furthermore, two types of errors can be defined and analyzed:

1. Cluster-level errors are errors associated to each cluster.
2. Record-level errors are errors associated to each record.

Analyze cluster-level errors
----------------------------

**Toy Example**

Consider the following set of ground truth clusters and predicted clusters of records :math:`1,2,\dots, 8`::

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


Analyse record-level errors
---------------------------

We define errors at the record level through a **record error table**, which provides the following quantities for each sampled record:

1. **pred_cluster_size**: The size of the predicted cluster which contains the record.
2. **ref_cluster_size**: The size of the true cluster which contains the record.
3. **extra_links**: The number of elements in the predicted cluster which are not in the true cluster.
4. **missing_links**: The number of elements in the true cluster which are not in the predicted cluster.

These four quantities, together with the record index, the predicted cluster ID, and the true cluster ID, are stored in what is called the **record error table**. The record error table can be computed using the :func:`record_error_table` function, given a sample of ground truth clusters and a prediction.

From the record error table, the cluster error metrics can be computed. The functions :func:`expected_size_difference_from_table`, :func:`expected_extra_links_from_table`, :func:`expected_missing_links_from_table`, :func:`expected_relative_extra_links_from_table`, :func:`expected_relative_missing_links_from_table`, and :func:`error_indicator_from_table` compute cluster-level errors from the record error table rather than from the prediction and the sample.

The key advantage of working with the record error table is that it allows sensitivity analyses to be performed. Since all cluster error metrics and representative performance estimators can be computed directly from the record error table, uncertainty regarding error rates can be propagated from the record error table into cluster error metrics and into performance estimates.
"""
from er_evaluation.error_analysis._cluster_error import (
    count_extra_links,
    count_missing_links,
    error_indicator,
    expected_extra_links,
    expected_missing_links,
    expected_relative_extra_links,
    expected_relative_missing_links,
    expected_size_difference,
    splitting_entropy,
)
from er_evaluation.error_analysis._record_error import (
    cluster_sizes_from_table,
    error_indicator_from_table,
    expected_extra_links_from_table,
    expected_missing_links_from_table,
    expected_relative_extra_links_from_table,
    expected_relative_missing_links_from_table,
    expected_size_difference_from_table,
    pred_cluster_sizes_from_table,
    record_error_table,
)

__all__ = [
    "count_extra_links",
    "count_missing_links",
    "error_indicator",
    "expected_extra_links",
    "expected_missing_links",
    "expected_relative_extra_links",
    "expected_relative_missing_links",
    "expected_size_difference",
    "splitting_entropy",
    "cluster_sizes_from_table",
    "error_indicator_from_table",
    "expected_extra_links_from_table",
    "expected_missing_links_from_table",
    "expected_relative_extra_links_from_table",
    "expected_relative_missing_links_from_table",
    "expected_size_difference_from_table",
    "pred_cluster_sizes_from_table",
    "record_error_table",
]
