import pandas as pd

from er_evaluation.error_analysis import (error_metrics_from_table,
                                          record_error_table)


def test_error_metrics_from_table():
    prediction = pd.Series(index=[1, 2, 3, 4, 5, 6, 7, 8], data=[1, 1, 2, 3, 2, 4, 4, 4])
    sample = pd.Series(index=[1, 2, 3, 4, 5, 6, 7, 8], data=["c1", "c1", "c1", "c2", "c2", "c3", "c3", "c3"])
    error_table = record_error_table(prediction, sample)

    expected_df = pd.DataFrame(
        {
            "expected_extra": [0.333333, 0.500000, 0.000000],
            "expected_relative_extra": [0.166667, 0.250000, 0.000000],
            "expected_missing": [1.333333, 1.000000, 0.000000],
            "expected_relative_missing": [0.444444, 0.500000, 0.000000],
            "error_indicator": [1, 1, 0],
        },
        index=pd.Index(["c1", "c2", "c3"], name="reference"),
    )

    result = error_metrics_from_table(error_table)

    pd.testing.assert_frame_equal(result, expected_df, check_dtype=False, atol=1e-4)
