import pandas as pd
import itertools


def expand_grid(**kwargs):
    """
    Create DataFrame from all combination of elements.

    Args:
        kwargs: Dictionary of elements to combine. Keys become column names.
    
    Returns:
        DataFrame: DataFrame with columns corresponding to argument names and rows for each combination of argument values.
    
    Examples:
        >>> expand_grid(col1=[1,2], col2=["a", "b"])
           col1 col2
        0     1    a
        1     1    b
        2     2    a
        3     2    b

        >>> expand_grid(col1={1:"something", 2:"something"}, col2=["a", "b"])
           col1 col2
        0     1    a
        1     1    b
        2     2    a
        3     2    b
    """
    return pd.DataFrame.from_records(
        itertools.product(*kwargs.values()), columns=kwargs.keys()
    )
