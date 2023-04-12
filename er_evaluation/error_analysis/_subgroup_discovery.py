import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree._splitter import Splitter
from sklearn.tree._criterion import Criterion
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.impute import SimpleImputer
from sklearn.tree import _tree
from sklearn.tree._tree import TREE_LEAF

def fit_dt_regressor(X, y, numerical_features=None, categorical_features=None, sample_weights=None, random_state=0, criterion="squared_error", **kwargs):
    """
    Fits a decision tree regressor model with optional preprocessing for numerical and categorical features.

    Args:
        X (numpy array or pandas DataFrame): The input features.
        y (numpy array or pandas Series): The target values.
        numerical_features (list of int or str, optional): The column indices or column names of numerical features. Default is None.
        categorical_features (list of int or str, optional): The column indices or column names of categorical features. Default is None.
        sample_weights (numpy array, optional): Individual weights for each sample. Default is None.
        random_state (int): Random state for the decision tree regressor.
        criterion (str): The function to measure the quality of a split. Supported criteria are "squared_error", "friedman_mse", "absolute_error", and "poisson". Default is "squared_error".
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
    clf = DecisionTreeRegressor(**kwargs, criterion=criterion, random_state=random_state)

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
        ('regressor', clf),
    ])

    model.fit(X, y, regressor__sample_weight=sample_weights)

    return model


class MeanDifferenceCriterion(Criterion):
    def __init__(self, n_outputs, n_samples):
        self.n_outputs = n_outputs
        self.n_samples = n_samples
        self.mean_left = 0
        self.mean_right = 0

    def init(self, y, sample_weight, weighted_n_samples, sample_indices, start, end):
        super().init(y, sample_weight, weighted_n_samples, sample_indices, start, end)

    def update(self, new_pos):
        super().update(new_pos)
        self.mean_left = self.sum_left[0] / self.weighted_n_left
        self.mean_right = self.sum_right[0] / self.weighted_n_right

    def node_impurity(self):
        return 0

    def node_value(self, dest):
        return 0

    def impurity_improvement(self, impurity_parent, impurity_left, impurity_right):
        if (self.weighted_n_left >= self.min_samples_leaf and
            self.weighted_n_right >= self.min_samples_leaf and
            self.weighted_n_node_samples >= self.min_samples_split):
            return np.abs(self.mean_left - self.mean_right)
        else:
            return 0

    def proxy_impurity_improvement(self):
        return self.impurity_improvement(0, 0, 0)