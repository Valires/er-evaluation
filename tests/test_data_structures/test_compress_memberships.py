import pandas as pd

from er_evaluation.data_structures import compress_memberships


def test_keep_na_values_in_index():
    series1 = pd.Series(index=[-1, 0, 4, 7], data=[pd.NA, 1, 2, 3])
    series2 = pd.Series(index=[1, 0, 4, 8], data=[1, pd.NA, 2, 3])
    cs1, cs2 = compress_memberships(series1, series2)

    assert cs1.isna().sum() == 3
    assert cs2.isna().sum() == 3
