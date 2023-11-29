import numpy as np
import pandas as pd
import pytest

from er_evaluation.data_structures import (MembershipVector, ismembership,
                                           membership_to_clusters,
                                           membership_to_pairs)
from er_evaluation.datasets import load_pv_disambiguations

predictions, reference = load_pv_disambiguations()

valid_memberships = {
    "empty_series": pd.Series(),
    "series_with_all_nans": pd.Series(index=[1], data=[None]),
    "single_cluster": pd.Series(index=[1, 2, 3, 4, 5, 6, 7, 8], data=[1, 1, 1, 1, 1, 1, 1, pd.NA]),
    "singletons": pd.Series(index=[1, 2, 3, 4, 5, 6, 7, 8], data=[1, 2, 3, 4, 5, 6, 7, np.nan]),
    "standard_series": pd.Series(index=[1, 2, 3, 4, 5, 6, 7, 8], data=[1, 1, 2, 3, 2, 4, 4, 4]),
    "series_with_string_nan_indexes": pd.Series(
        index=["1", 2, 3, 4, "5", 6, 7, 8], data=[1, 1, None, 3, 2, None, "4", "4"]
    ),
    "pv_reference": reference,
    "pv_prediction": predictions[pd.Timestamp("2017-08-08")],
}

invalid_memberships = {
    "nan_in_index": pd.Series(index=[None], data=[1]),
    "duplicate_index": pd.Series(index=[1, 1], data=[1, 2]),
    "nan_in_index_2": pd.Series(index=[1, 2, 3, 4, None, 6, 7, 8], data=[1, 1, 2, 3, 2, 4, 4, 4]),
    "duplicate_index_2": pd.Series(index=["1", "1", 3, 4, "5", 6, 7, 8], data=[1, 1, None, 3, 2, None, "4", "4"]),
}


@pytest.mark.parametrize(
    "series, expected_output",
    [(series, True) for series in valid_memberships.values()]
    + [(series, False) for series in invalid_memberships.values()],
)
def test_ismembership(series, expected_output):
    assert ismembership(series) == expected_output


@pytest.mark.parametrize(
    "series, expected_output", [(series, MembershipVector(series)) for series in valid_memberships.values()]
)
def test_membershipvector_not_destructive(series, expected_output):
    assert series.equals(expected_output)


@pytest.mark.parametrize(
    "series_name, expected_output",
    [
        ("empty_series", (0, 2)),
        ("series_with_all_nans", (0, 2)),
        ("single_cluster", (21, 2)),
        ("singletons", (0, 2)),
        ("standard_series", (5, 2)),
        ("series_with_string_nan_indexes", (2, 2)),
        ("pv_reference", (1437465, 2)),
        ("pv_prediction", (7802637, 2)),
    ],
)
def test_membership_to_pairs_by_shape(series_name, expected_output):
    series = valid_memberships[series_name]
    output = membership_to_pairs(series)

    assert output.shape == expected_output


@pytest.mark.parametrize(
    "series_name, expected_output",
    [
        ("empty_series", 0),
        ("series_with_all_nans", 0),
        ("single_cluster", 1),
        ("singletons", 7),
        ("standard_series", 4),
        ("series_with_string_nan_indexes", 4),
        ("pv_reference", 401),
        ("pv_prediction", 11264),
    ],
)
def test_membership_to_clusters_by_len(series_name, expected_output):
    series = valid_memberships[series_name]
    output = len(membership_to_clusters(series))

    assert output == expected_output
