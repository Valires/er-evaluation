from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.impute import SimpleImputer

def fit_dt_regressor(X, y, numerical_features=None, categorical_features=None, sample_weights=None, random_state=0, **kwargs):
    """
    Fits a decision tree regressor model with optional preprocessing for numerical and categorical features.

    Args:
        X (numpy array or pandas DataFrame): The input features.
        y (numpy array or pandas Series): The target values.
        numerical_features (list of int or str, optional): The column indices or column names of numerical features. Default is None.
        categorical_features (list of int or str, optional): The column indices or column names of categorical features. Default is None.
        sample_weights (numpy array, optional): Individual weights for each sample. Default is None.
        random_state (int): Random state for the decision tree regressor.
        **kwargs: Additional keyword arguments passed to the DecisionTreeRegressor constructor.

    Returns:
        sklearn.pipeline.Pipeline: A fitted decision tree regressor model with preprocessing steps.

    Examples:
        >>> import numpy as np
        >>> import pandas as pd
        >>> X = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": ["a", "b", "a"]})
        >>> y = np.array([2, 4, 6])
        >>> numerical_features = ["A", "B"]
        >>> categorical_features = ["C"]
        >>> model = fit_dt_regressor(X, y, numerical_features, categorical_features)
        >>> isinstance(model, Pipeline)
        True
    """
    preprocess_steps = []
    if numerical_features is not None:
        preprocess_steps += [(
            "imputer", 
            SimpleImputer(strategy="constant", fill_value=-1),
            numerical_features
        )]
    if categorical_features is not None:
        preprocess_steps += [(
            "one_hot_encoder", 
            OneHotEncoder(),
            categorical_features
        )]
    
    model = Pipeline(steps=[
        ('preprocessor', ColumnTransformer(preprocess_steps)),
        ('regressor', DecisionTreeRegressor(**kwargs, random_state=random_state)),
    ])

    model.fit(X, y, regressor__sample_weight=sample_weights)

    return model