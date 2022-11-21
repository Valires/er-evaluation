import pandas as pd
import itertools


def expand_grid(**kwargs):
    """Get all value combinations."""
    return pd.DataFrame.from_records(
        itertools.product(*kwargs.values()), columns=kwargs.keys()
    )
