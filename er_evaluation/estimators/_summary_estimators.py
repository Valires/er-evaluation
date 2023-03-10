import pandas as pd

from er_evaluation.estimators._utils import (
    ratio_of_means_estimator,
    validate_prediction_sample,
    _parse_weights,
    validate_weights,
)
from er_evaluation.summary import cluster_sizes
from er_evaluation.data_structures import MembershipVector
from er_evaluation.utils import expand_grid


def summary_estimates_table(sample, weights, predictions, names=None):
    estimators = {
        "Matching Rate Estimate": matching_rate_estimator,
        "Avg Cluster Size Estimate": avg_cluster_size_estimator,
    }
    if names is not None:
        estimators["Name Variation Estimate"] = name_variation_estimator
        estimators["Homonymy Rate Estimate"] = homonymy_rate_estimator

    params = expand_grid(prediction=predictions, estimate=estimators)

    def lambd(pred_key, est_key):
        ests = estimators[est_key](
            sample,
            weights,
            prediction=predictions[pred_key],
            names=names,
        )

        return ests

    params[["value", "std"]] = pd.DataFrame(
        params.apply(
            lambda x: lambd(x["prediction"], x["estimate"]),
            axis=1,
        ).tolist(),
        index=params.index,
    )

    return params


def _prepare_summary_est_args(sample, weights, prediction=None, names=None):
    prediction = MembershipVector(prediction, dropna=True)
    sample = MembershipVector(sample, dropna=True)

    if prediction is not None:
        validate_prediction_sample(prediction, sample)
        sample = sample[sample.index.isin(prediction.index)]
    if names is not None:
        assert isinstance(names, pd.Series)
        assert all(sample.index.isin(names.index))

    weights = _parse_weights(sample, weights)
    validate_weights(sample, weights)
    weights = weights[weights.index.isin(sample.values)]

    return sample, weights, prediction, names


@ratio_of_means_estimator
def matching_rate_estimator(sample, weights, prediction=None, names=None):
    sample, weights, prediction, names = _prepare_summary_est_args(sample, weights, prediction, names)

    cs = cluster_sizes(sample)
    N = cs * (cs > 1) * weights
    D = cs * weights

    return N, D


@ratio_of_means_estimator
def avg_cluster_size_estimator(sample, weights, prediction=None, names=None):
    sample, weights, prediction, names = _prepare_summary_est_args(sample, weights, prediction, names)

    cs = cluster_sizes(sample)
    N = cs * weights
    D = weights

    return N, D


@ratio_of_means_estimator
def name_variation_estimator(sample, weights, prediction=None, names=None):
    sample, weights, prediction, names = _prepare_summary_est_args(sample, weights, prediction, names)

    inner = pd.concat({"names": names, "sample": sample}, join="inner", axis=1)

    N = (inner.groupby("sample").nunique()["names"] > 1) * weights
    D = weights

    return N, D


@ratio_of_means_estimator
def homonymy_rate_estimator(sample, weights, prediction=None, names=None):
    sample, weights, prediction, names = _prepare_summary_est_args(sample, weights, prediction, names)

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

    return N, D
