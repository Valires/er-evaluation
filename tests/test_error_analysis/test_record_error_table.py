import pandas as pd
import numpy as np

from er_evaluation.error_analysis import record_error_table
from er_evaluation.datasets import load_rldata10000_disambiguations

predictions, reference = load_rldata10000_disambiguations()

pred_ref = {
    "empty_pred_ref": {
        "prediction": pd.Series(),
        "reference": pd.Series(),
    },
    "empty_ref_standard_pred": {
        "prediction": pd.Series(index=[1, 2, 3, 4, 5], data=[None, 1, 1, 1, 3]),
        "reference": pd.Series(),
    },
    "empty_pred_standard_ref": {
        "prediction": pd.Series(),
        "reference": pd.Series(index=[1, 2, 3, 4, 5, 8], data=["c1", "c1", "c1", "c2", "c2", "c4"]),
    },
    "standard_pred_ref": {
        "prediction": pd.Series(index=[1, 2, 3, 4, 5, 6, 7, 8], data=[1, 1, 2, 3, 2, 4, 4, 4]),
        "reference": pd.Series(index=[1, 2, 3, 4, 5, 8], data=["c1", "c1", "c1", "c2", "c2", "c4"]),
    },
    "equal_pred_ref": {
        "prediction": pd.Series(index=[1, 2, 3, 4, 5, 6, 7, 8], data=[1, 1, 2, 3, 2, 4, 4, 4]),
        "reference": pd.Series(index=[1, 2, 3, 4, 5, 6, 7, 8], data=[1, 1, 2, 3, 2, 4, 4, 4]),
    },
    "smaller_pref_than_ref": {
        "prediction": pd.Series(index=[1, 2, 3, 4, 5, 6, 7, 8], data=[1, 1, 2, 3, 2, 4, 4, 4]),
        "reference": pd.Series(index=[1, 2, 3, 4, 5, 8], data=["c1", "c1", "c1", "c2", "c2", "c4"]),
    },
}

expected_outputs = {
    "empty_pred_ref": pd.DataFrame(
        columns=["pred_cluster_size", "ref_cluster_size", "prediction", "reference", "extra", "missing"], dtype=np.int64
    ),
    "empty_ref_standard_pred": pd.DataFrame(
        columns=["pred_cluster_size", "ref_cluster_size", "prediction", "reference", "extra", "missing"], dtype=np.int64
    ),
    "empty_pred_standard_ref": pd.DataFrame(
        columns=["pred_cluster_size", "ref_cluster_size", "prediction", "reference", "extra", "missing"], dtype=np.int64
    ),
    "standard_pred_ref": pd.DataFrame(
        {
            "prediction": {0: 1, 1: 1, 2: 2, 3: 3, 4: 2, 5: 4},
            "reference": {0: "c1", 1: "c1", 2: "c1", 3: "c2", 4: "c2", 5: "c4"},
            "pred_cluster_size": {0: 2, 1: 2, 2: 2, 3: 1, 4: 2, 5: 3},
            "ref_cluster_size": {0: 3.0, 1: 3.0, 2: 3.0, 3: 2.0, 4: 2.0, 5: 1.0},
            "extra": {0: 0, 1: 0, 2: 1, 3: 0, 4: 1, 5: 2},
            "missing": {0: 1.0, 1: 1.0, 2: 2.0, 3: 1.0, 4: 1.0, 5: 0.0},
        }
    ),
    "equal_pred_ref": pd.DataFrame(
        {
            "prediction": {0: 1, 1: 1, 2: 2, 3: 3, 4: 2, 5: 4, 6: 4, 7: 4},
            "reference": {0: 1, 1: 1, 2: 2, 3: 3, 4: 2, 5: 4, 6: 4, 7: 4},
            "pred_cluster_size": {0: 2, 1: 2, 2: 2, 3: 1, 4: 2, 5: 3, 6: 3, 7: 3},
            "ref_cluster_size": {0: 2, 1: 2, 2: 2, 3: 1, 4: 2, 5: 3, 6: 3, 7: 3},
            "extra": {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0},
            "missing": {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0},
        }
    ),
    "smaller_pref_than_ref": pd.DataFrame(
        {
            "prediction": {0: 1, 1: 1, 2: 2, 3: 3, 4: 2, 5: 4},
            "reference": {0: "c1", 1: "c1", 2: "c1", 3: "c2", 4: "c2", 5: "c4"},
            "pred_cluster_size": {0: 2, 1: 2, 2: 2, 3: 1, 4: 2, 5: 3},
            "ref_cluster_size": {0: 3.0, 1: 3.0, 2: 3.0, 3: 2.0, 4: 2.0, 5: 1.0},
            "extra": {0: 0, 1: 0, 2: 1, 3: 0, 4: 1, 5: 2},
            "missing": {0: 1.0, 1: 1.0, 2: 2.0, 3: 1.0, 4: 1.0, 5: 0.0},
        }
    ),
}


def test_record_error_table():
    for key in pred_ref.keys():
        pred = pred_ref[key]["prediction"]
        ref = pred_ref[key]["reference"]
        expected = expected_outputs[key]
        output = record_error_table(pred, ref)

        pd.testing.assert_frame_equal(output, expected, check_dtype=False)
