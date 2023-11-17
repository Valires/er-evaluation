import pandas as pd
import pytest

from er_evaluation.data_structures import ismembership
from er_evaluation.datasets import (
    load_pv_data,
    load_pv_disambiguations,
    load_rldata500,
    load_rldata500_disambiguations,
    load_rldata10000,
    load_rldata10000_disambiguations,
)

pv_data = load_pv_data()
pv_predictions, pv_reference = load_pv_disambiguations()
rldata10000 = load_rldata10000()
rl10000_predictions, rl10000_reference = load_rldata10000_disambiguations()
rldata500 = load_rldata500()
rl500_predictions, rl500_reference = load_rldata500_disambiguations()


@pytest.mark.parametrize(
    "dataset, expected_shape",
    [
        (pv_data, (133541, 40)),
        (rldata10000, (10000, 7)),
        (rldata500, (500, 7)),
    ],
)
def test_dataset_shape(dataset, expected_shape):
    assert isinstance(dataset, pd.DataFrame)
    assert dataset.shape == expected_shape


@pytest.mark.parametrize(
    "series, expected_shape",
    [
        (pv_reference, (133541,)),
        (rl10000_reference, (10000,)),
        (rl500_reference, (500,)),
    ]
    + [(series, (133541,)) for series in pv_predictions.values()]
    + [(series, (10000,)) for series in rl10000_predictions.values()]
    + [(series, (500,)) for series in rl500_predictions.values()],
)
def test_disambiguations_shape(series, expected_shape):
    assert ismembership(series)
    assert series.shape == expected_shape
