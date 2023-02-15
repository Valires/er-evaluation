import pandas as pd
from er_evaluation.estimators._utils import (
    ratio_estimator,
    std_dev,
    _prepare_args,
)
from er_evaluation.summary import cluster_sizes


def matching_rate_estimator(prediction, sample, weights):
    prediction, sample, weights = _prepare_args(prediction, sample, weights)

    cs = cluster_sizes(sample)
    N = cs * (cs > 1) * weights
    D = cs * weights

    return (ratio_estimator(N, D), std_dev(N, D))


def avg_cluster_size_estimator(prediction, sample, weights):
    prediction, sample, weights = _prepare_args(prediction, sample, weights)

    cs = cluster_sizes(sample)
    N = cs * weights
    D = weights

    return (ratio_estimator(N, D), std_dev(N, D))


def name_variation_estimator(names, sample, weights):
    names, sample, weights = _prepare_args(names, sample, weights)
    assert isinstance(names, pd.Series)
    assert all(sample.index.isin(names.index))

    inner = pd.concat({"names": names, "sample": sample}, join="inner", axis=1)

    N = (inner.groupby("sample").nunique() > 1) * weights
    D = weights

    return (ratio_estimator(N, D), std_dev(N, D))


def homonymy_rate_estimator(names, sample, weights):
    names, sample, weights = _prepare_args(names, sample, weights)
    assert isinstance(names, pd.Series)
    assert all(sample.index.isin(names.index))

    inner = pd.concat({"names": names, "sample": sample}, join="inner", axis=1)

    name_counts = (
        inner.groupby("sample")
        .value_counts()
        .reset_index(name="within_count")
        .merge(names.value_counts().rename("total_count"), left_on="names", right_index=True, how="left")
    )
    name_counts["diff"] = name_counts["total_count"] - name_counts["within_count"]

    N = (name_counts.groupby("sample").agg("max")["diff"] > 0) * weights
    D = weights

    return (ratio_estimator(N, D), std_dev(N, D))
