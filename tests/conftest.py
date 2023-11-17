import pandas as pd
import pytest


@pytest.fixture
def pred_one():
    return pd.Series(index=[1, 2, 3], data=[1, 1, 1])


@pytest.fixture
def pred_singleton():
    return pd.Series(index=[1, 2, 3], data=[1, 2, 3])


@pytest.fixture
def names_unique():
    return pd.Series(index=[1, 2, 3], data=["a", "b", "c"])


@pytest.fixture
def names_common():
    return pd.Series(index=[1, 2, 3], data=["a", "a", "a"])
