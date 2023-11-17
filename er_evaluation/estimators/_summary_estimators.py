import pandas as pd

from er_evaluation.data_structures import MembershipVector
from er_evaluation.estimators._utils import (_parse_weights,
                                             ratio_of_means_estimator,
                                             validate_prediction_sample,
                                             validate_weights)
from er_evaluation.summary import cluster_sizes
from er_evaluation.utils import expand_grid


def summary_estimates_table(sample, weights, predictions, names=None):
    """
    Generate a summary estimates table for the given sample, weights, predictions, and names.

    Args:
        sample (Series): Membership vector indexed by cluster elements and with values corresponding to associated cluster identifier.
        weights (Series): Pandas Series indexed by cluster identifier and with values corresponding to cluster sampling weights (e.g., inverse sampling probabilities). Can also be the string "uniform" for uniform sampling weights, or "cluster_size" for inverse cluster size sampling weights.
        predictions (Dict): Dictionary of membership vectors.
        names (pd.Series, optional): Series containing names associated with each cluster element. Used for Name Variation and Homonymy Rate Estimates. Defaults to None.

    Returns:
        DataFrame: Pandas DataFrame with columns "prediction", "estimate", "value", and "std", where value and std are the point estimate and standard deviation estimate for the estimator applied to the given sample, sampling weights, prediction, and names.
    """
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
    """
    Prepare and validate the arguments for summary estimators.

    Args:
        sample (Series): Membership vector indexed by cluster elements and with values corresponding to associated cluster identifier.
        weights (Series): Pandas Series indexed by cluster identifier and with values corresponding to cluster sampling weights (e.g., inverse sampling probabilities). Can also be the string "uniform" for uniform sampling weights, or "cluster_size" for inverse cluster size sampling weights.
        prediction (pd.Series, optional): Membership vector indexed by cluster elements and with values corresponding to associated cluster identifier. Defaults to None.
        names (pd.Series, optional): Series containing names associated with each cluster element. Used for Name Variation and Homonymy Rate Estimates. Defaults to None.

    Returns:
        tuple: Prepared and validated sample, weights, prediction, and names.
    """
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
    """
    Compute the matching rate estimator for the given sample, weights, prediction, and names.

    Matching rate:
        This is the proportion of elements belonging to clusters of size at least 2.

    Args:
        sample (Series): Membership vector indexed by cluster elements and with values corresponding to associated cluster identifier.
        weights (Series): Pandas Series indexed by cluster identifier and with values corresponding to cluster sampling weights (e.g., inverse sampling probabilities). Can also be the string "uniform" for uniform sampling weights, or "cluster_size" for inverse cluster size sampling weights.
        prediction (pd.Series, optional): Membership vector indexed by cluster elements and with values corresponding to associated cluster identifier. Defaults to None.
        names (pd.Series, optional): Series containing names associated with each cluster element. Used for Name Variation and Homonymy Rate Estimates. Defaults to None.

    Returns:
        tuple: Matching rate estimate and standard deviation estimate.
    """
    sample, weights, prediction, names = _prepare_summary_est_args(sample, weights, prediction, names)

    cs = cluster_sizes(sample)
    N = cs * (cs > 1) * weights
    D = cs * weights

    return N, D


@ratio_of_means_estimator
def avg_cluster_size_estimator(sample, weights, prediction=None, names=None):
    """
    Compute the average cluster size estimator for the given sample, weights, prediction, and names.

    Args:
        sample (Series): Membership vector indexed by cluster elements and with values corresponding to associated cluster identifier.
        weights (Series): Pandas Series indexed by cluster identifier and with values corresponding to cluster sampling weights (e.g., inverse sampling probabilities). Can also be the string "uniform" for uniform sampling weights, or "cluster_size" for inverse cluster size sampling weights.
        prediction (pd.Series, optional): Membership vector indexed by cluster elements and with values corresponding to associated cluster identifier. Defaults to None.
        names (pd.Series, optional): Series containing names associated with each cluster element. Used for Name Variation and Homonymy Rate Estimates. Defaults to None.

    Returns:
        tuple: Average cluster size estimate and standard deviation estimate.
    """
    sample, weights, prediction, names = _prepare_summary_est_args(sample, weights, prediction, names)

    cs = cluster_sizes(sample)
    N = cs * weights
    D = weights

    return N, D


@ratio_of_means_estimator
def name_variation_estimator(sample, weights, prediction=None, names=None):
    """
    Compute the name variation estimator for the given sample, weights, prediction, and names.

    Name variation rate:
        The name variation rate is the proportion of clusters with name variation within.

    Args:
        sample (Series): Membership vector indexed by cluster elements and with values corresponding to associated cluster identifier.
        weights (Series): Pandas Series indexed by cluster identifier and with values corresponding to cluster sampling weights (e.g., inverse sampling probabilities). Can also be the string "uniform" for uniform sampling weights, or "cluster_size" for inverse cluster size sampling weights.
        prediction (pd.Series, optional): Membership vector indexed by cluster elements and with values corresponding to associated cluster identifier. Defaults to None.
        names (pd.Series, optional): Series containing names associated with each cluster element. Used for Name Variation and Homonymy Rate Estimates. Defaults to None.

    Returns:
        tuple: Name variation estimate and standard deviation estimate.
    """
    sample, weights, prediction, names = _prepare_summary_est_args(sample, weights, prediction, names)

    inner = pd.concat({"names": names, "sample": sample}, join="inner", axis=1)

    N = (inner.groupby("sample").nunique()["names"] > 1) * weights
    D = weights

    return N, D


@ratio_of_means_estimator
def homonymy_rate_estimator(sample, weights, prediction=None, names=None):
    """
    Compute the homonymy rate estimator for the given sample, weights, prediction, and names.

    Homonymy rate:
        The homonymy rate is the proportion of clusters which share a name with another cluster.

    Args:
        sample (Series): Membership vector indexed by cluster elements and with values corresponding to associated cluster identifier.
        weights (Series): Pandas Series indexed by cluster identifier and with values corresponding to cluster sampling weights (e.g., inverse sampling probabilities). Can also be the string "uniform" for uniform sampling weights, or "cluster_size" for inverse cluster size sampling weights.
        prediction (pd.Series, optional): Membership vector indexed by cluster elements and with values corresponding to associated cluster identifier. Defaults to None.
        names (pd.Series, optional): Series containing names associated with each cluster element. Used for Name Variation and Homonymy Rate Estimates. Defaults to None.

    Returns:
        tuple: Homonymy rate estimate and standard deviation estimate.
    """
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
