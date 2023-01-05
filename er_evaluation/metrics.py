"""
Evaluation Metrics (Precision, Recall, etc)
"""

import pandas as pd
import numpy as np
from scipy.special import comb
import sklearn.metrics as sm
from functools import wraps

from er_evaluation.data_structures import ismembership
from er_evaluation.summary import number_of_links
from er_evaluation.utils import expand_grid
from er_evaluation.error_analysis import error_indicator


def _f_score(P, R, beta=1.0):
    """
    Compute the weighted F1 score for a given precision and recall.

    Args:
        P (float): Precision.
        R (float): Recall.
        beta (float): Weighting factor.

    Returns:
        float: Weighted F1 score.

    Examples:
        >>> _f_score(0.5, 0.5, beta=1.0)
        0.5
    """
    D = beta**2 * P + R
    if D == 0:
        return 0
    else:
        return (1 + beta**2) * P * R / D


def metrics_table(predictions, references, metrics):
    """
    Apply a set of metrics to all combinations of prediction and reference membership vectors.

    Args:
        predictions (Dict): Dictionary of membership vectors.
        references (Dict): Dictionary of membership vectors.
        metrics (Dict): Dictionary of metrics to apply to the prediction and reference pairs.

    Returns:
        DataFrame: Dataframe with columns "prediction", "reference", "metric", and "value", containing the value of the given metric applied to the corresponding prediction and reference membership vector.

    Examples:
        >>> predictions = {"prediction_1": pd.Series(index=[1,2,3,4,5,6,7,8], data=[1,1,2,3,2,4,4,4])}
        >>> references = {"reference_1": pd.Series(index=[1,2,3,4,5,6,7,8], data=["c1", "c1", "c1", "c2", "c2", "c3", "c3", "c4"])}
        >>> metrics = {"precision": pairwise_precision, "recall": pairwise_recall}
        >>> metrics_table(predictions, references, metrics) # doctest: +NORMALIZE_WHITESPACE
            prediction	    reference	metric	    value
        0	prediction_1	reference_1	precision	0.4
        1	prediction_1	reference_1	recall	    0.4
    """
    params = expand_grid(
        prediction=predictions, reference=references, metric=metrics
    )

    def lambd(pred_key, ref_key, metrics_key):
        return metrics[metrics_key](predictions[pred_key], references[ref_key])

    params["value"] = params.apply(
        lambda x: lambd(x["prediction"], x["reference"], x["metric"]), axis=1
    )

    return params


def pairwise_precision(prediction, reference):
    r"""
    Pairwise precision for the inner join of two clusterings.

    Pairwise precision:
        Consider two clusterings of a set of records, refered to as the *predicted* and *reference* clusterings. Let :math:`T` be the set of record pairs which appear in the same reference cluster, and let :math:`P` be the set of record pairs which appear in the same predicted clusters. Pairwise precision is then defined as

        .. math::

            P = \frac{\lvert T \cap P \rvert}{\lvert P \rvert}

        This is the proportion of correctly predicted links among all predicted links.

    Args:
        prediction (Series): Membership vector for the predicted clustering.
        reference (Series): Membership vector for the reference clustering.

    Returns:
        float: Pairwise precision for the inner join of `prediction` and `reference`.

    Examples:
        >>> prediction = pd.Series(index=[1,2,3,4,5,6,7,8], data=[1,1,2,3,2,4,4,4])
        >>> reference = pd.Series(index=[1,2,3,4,5,6,7,8], data=["c1", "c1", "c1", "c2", "c2", "c3", "c3", "c4"])
        >>> pairwise_precision(prediction, reference)
        0.4
    """
    assert ismembership(prediction) and ismembership(reference)

    inner = pd.concat(
        {"prediction": prediction, "reference": reference},
        axis=1,
        join="inner",
        copy=False,
    )
    TP_cluster_sizes = inner.groupby(["prediction", "reference"]).size().values

    TP = np.sum(comb(TP_cluster_sizes, 2))
    P = number_of_links(inner.prediction)

    if P == 0:
        return 1.0
    else:
        return TP / P


def pairwise_recall(prediction, reference):
    r"""
    Pairwise recall for the inner join of two clusterings.

    Pairwise recall:
        Consider two clusterings of a set of records, refered to as the *predicted* and *reference* clusterings. Let :math:`T` be the set of record pairs which appear in the same reference cluster, and let :math:`P` be the set of record pairs which appear in the same predicted clusters. Pairwise recall is then defined as

        .. math::

            R = \frac{\lvert T \cap P \rvert}{\lvert T \rvert}

        This is the proportion of correctly predicted links among all true links.

    Args:
        prediction (Series): Membership vector for the predicted clustering.
        reference (Series): Membership vector for the reference clustering.

    Returns:
        float: Pairwise recall computed on the inner join of `predicted` and `reference`.

    Examples:
        >>> prediction = pd.Series(index=[1,2,3,4,5,6,7,8], data=[1,1,2,3,2,4,4,4])
        >>> reference = pd.Series(index=[1,2,3,4,5,6,7,8], data=["c1", "c1", "c1", "c2", "c2", "c3", "c3", "c4"])
        >>> pairwise_recall(prediction, reference)
        0.4
    """
    return pairwise_precision(reference, prediction)


def pairwise_f(prediction, reference, beta=1.0):
    r"""
    Pairwise F score for the inner join of two clusterings.

    Pairwise F score:
        Pairwise F score is defined as the weighted harmonic mean of pairwise precision and pairwise recall: :math:`F_\beta = \frac{(1 + \beta^2)PR}{ \beta^2 P+R}`. The :math:`\beta` parameter controls the relative weight of precision and recall. When :math:`\beta = 1`, the F1 score is the harmonic mean of precision and recall. When :math:`\beta < 1`, the F1 score is weighted towards precision. When :math:`\beta > 1`, the F score is weighted towards recall.

    Args:
        prediction (Series): Membership vector for the predicted clustering.
        reference (Series): Membership vector for the reference clustering.
        beta (float): Weight of precision in the F score.

    Returns:
        float: Pairwise F score for the inner join of `prediction` and `reference`.

    Examples:
        >>> prediction = pd.Series(index=[1,2,3,4,5,6,7,8], data=[1,1,2,3,2,4,4,4])
        >>> reference = pd.Series(index=[1,2,3,4,5,6,7,8], data=["c1", "c1", "c1", "c2", "c2", "c3", "c3", "c4"])
        >>> pairwise_f(prediction, reference)
        0.4000000000000001
    """
    P = pairwise_precision(prediction, reference)
    R = pairwise_recall(prediction, reference)

    return _f_score(P, R, beta=beta)


def cluster_precision(prediction, reference):
    r"""
    Cluster precision for the inner join of two clusterings.

    Cluster precision:
        Consider two clusterings of a set of records, refered to as the *predicted* and *reference* clusterings. Let :math:`T` be the set of reference clusters, and let :math:`P` be the set of predicted clusters. Cluster precision is then defined as

        .. math::

            P = \frac{\lvert T \cap P \rvert}{\lvert P \rvert}

        This is the proportion of correctly predicted clusters among all predicted clusters.

    Args:
        prediction (Series): Membership vector for the predicted clustering.
        reference (Series): Membership vector for the reference clustering.

    Returns:
        float: Cluster precision for the inner join of `prediction` and `reference`.

    Examples:
        >>> prediction = pd.Series(index=[1,2,3,4,5,6,7,8], data=[1,1,2,3,2,4,4,5])
        >>> reference = pd.Series(index=[1,2,3,4,5,6,7,8], data=["c1", "c1", "c1", "c2", "c2", "c3", "c3", "c4"])
        >>> cluster_precision(prediction, reference)
        0.4
    """
    assert ismembership(prediction) and ismembership(reference)
    inner = pd.concat(
        {"prediction": prediction, "reference": reference},
        axis=1,
        join="inner",
        copy=False,
    )
    errors = error_indicator(inner.prediction, inner.reference)

    return (1 - errors).sum() / inner.prediction.nunique()


def cluster_recall(prediction, reference):
    r"""
    Cluster recall for the inner join of two clusterings.

    Cluster recall:
        Consider two clusterings of a set of records, refered to as the *predicted* and *reference* clusterings. Let :math:`T` be the set of reference clusters, and let :math:`P` be the set of predicted clusters. Cluster recall is then defined as

        .. math::

            R = \frac{\lvert T \cap P \rvert}{\lvert T \rvert}

        This is the proportion of correctly predicted clusters among all reference clusters.

    Args:
        prediction (Series): Membership vector for the predicted clustering.
        reference (Series): Membership vector for the reference clustering.

    Returns:
        float: Cluster recall for the inner join of `prediction` and `reference`.

    Examples:
        >>> prediction = pd.Series(index=[1,2,3,4,5,6,7,8], data=[1,1,2,3,2,4,4,5])
        >>> reference = pd.Series(index=[1,2,3,4,5,6,7,8], data=["c1", "c1", "c1", "c2", "c2", "c3", "c3", "c4"])
        >>> cluster_recall(prediction, reference)
        0.5
    """
    return cluster_precision(reference, prediction)


def cluster_f(prediction, reference, beta=1.0):
    r"""
    Cluster F score for the inner join of two clusterings.

    Cluster F score:
        Cluster F score is defined as the weighted harmonic mean of cluster precision and cluster recall: :math:`F_\beta = \frac{(1 + \beta^2)PR}{ \beta^2 P+R}`. The :math:`\beta` parameter controls the relative weight of precision and recall. When :math:`\beta = 1`, the F score is the harmonic mean of precision and recall. When :math:`\beta < 1`, the F score is weighted towards precision. When :math:`\beta > 1`, the F score is weighted towards recall.

    Args:
        prediction (Series): Membership vector for the predicted clustering.
        reference (Series): Membership vector for the reference clustering.
        beta (float): Weight of precision in the F score.

    Returns:
        float: Cluster F score for the inner join of `prediction` and `reference`.

    Examples:
        >>> prediction = pd.Series(index=[1,2,3,4,5,6,7,8], data=[1,1,2,3,2,4,4,5])
        >>> reference = pd.Series(index=[1,2,3,4,5,6,7,8], data=["c1", "c1", "c1", "c2", "c2", "c3", "c3", "c4"])
        >>> cluster_f(prediction, reference)
        0.4444444444444445
    """
    P = cluster_precision(prediction, reference)
    R = cluster_recall(prediction, reference)

    return _f_score(P, R, beta=beta)


def b_cubed_precision(prediction, reference):
    r"""
    B-cubed precision for the inner join of two clusterings.

    Args:
        prediction (Series): Membership vector for the predicted clustering.
        reference (Series): Membership vector for the reference clustering.

    Returns:
        float: B-cubed precision for the inner join of `prediction` and `reference`.

    Examples:
        >>> prediction = pd.Series(index=[1,2,3,4,5,6,7,8], data=[1,1,2,3,2,4,4,4])
        >>> reference = pd.Series(index=[1,2,3,4,5,6,7,8], data=["c1", "c1", "c1", "c2", "c2", "c3", "c3", "c4"])
        >>> b_cubed_precision(prediction, reference)
        0.6458333333333333
    """
    assert ismembership(prediction) and ismembership(reference)

    inner = pd.concat(
        {"prediction": prediction, "reference": reference},
        axis=1,
        join="inner",
        copy=False,
    )

    intersection_sizes = inner.value_counts()
    ref_cluster_sizes = inner.reference.value_counts()
    ref_cluster_sizes.index.name = "reference"
    pred_cluster_sizes = inner.prediction.value_counts()
    pred_cluster_sizes.index.name = "prediction"

    n_clusters = inner.reference.nunique()

    df = (
        inner.merge(
            intersection_sizes.reset_index(name="intersection_size"), how="left"
        )
        .merge(
            ref_cluster_sizes.reset_index(name="ref_cluster_size"), how="left"
        )
        .merge(
            pred_cluster_sizes.reset_index(name="pred_cluster_size"), how="left"
        )
    )
    df = df[df.intersection_size > 0]

    return (
        df.intersection_size
        / (df.ref_cluster_size * df.pred_cluster_size * n_clusters)
    ).sum()


def b_cubed_recall(prediction, reference):
    r"""
    B-cubed recall for the inner join of two clusterings.

    Args:
        prediction (Series): Membership vector for the predicted clustering.
        reference (Series): Membership vector for the reference clustering.

    Returns:
        float: B-cubed recall for the inner join of `prediction` and `reference`.

    Examples:
        >>> prediction = pd.Series(index=[1,2,3,4,5,6,7,8], data=[1,1,2,3,2,4,4,4])
        >>> reference = pd.Series(index=[1,2,3,4,5,6,7,8], data=["c1", "c1", "c1", "c2", "c2", "c3", "c3", "c4"])
        >>> b_cubed_recall(prediction, reference)
        0.7638888888888888
    """
    assert ismembership(prediction) and ismembership(reference)

    inner = pd.concat(
        {"prediction": prediction, "reference": reference},
        axis=1,
        join="inner",
        copy=False,
    )

    intersection_sizes = inner.value_counts()
    ref_cluster_sizes = inner.reference.value_counts()
    ref_cluster_sizes.index.name = "reference"
    pred_cluster_sizes = inner.prediction.value_counts()
    pred_cluster_sizes.index.name = "prediction"

    n_clusters = inner.reference.nunique()

    df = (
        inner.merge(
            intersection_sizes.reset_index(name="intersection_size"), how="left"
        )
        .merge(
            ref_cluster_sizes.reset_index(name="ref_cluster_size"), how="left"
        )
        .merge(
            pred_cluster_sizes.reset_index(name="pred_cluster_size"), how="left"
        )
    )
    df = df[df.intersection_size > 0]

    return (
        df.intersection_size / (df.ref_cluster_size**2 * n_clusters)
    ).sum()


def b_cubed_f(prediction, reference, beta=1.0):
    r"""
    B-cubed F score for the inner join of two clusterings.

    Args:
        prediction (Series): Membership vector for the predicted clustering.
        reference (Series): Membership vector for the reference clustering.
        beta (float): Weight of precision in the F score.

    Returns:
        float: B-cubed F score for the inner join of `prediction` and `reference`.

    Examples:
        >>> prediction = pd.Series(index=[1,2,3,4,5,6,7,8], data=[1,1,2,3,2,4,4,4])
        >>> reference = pd.Series(index=[1,2,3,4,5,6,7,8], data=["c1", "c1", "c1", "c2", "c2", "c3", "c3", "c4"])
        >>> b_cubed_f(prediction, reference)
        0.6999178981937603
    """
    P = b_cubed_precision(prediction, reference)
    R = b_cubed_recall(prediction, reference)

    return _f_score(P, R, beta=beta)


def wrap_sklearn_metric(sklearn_metric):
    """Generic function to wrap sklearn cluster metrics.

    Args:
        sklearn_metric (function): cluster metric to wrap.

    Notes:
        * The prediction and reference membership vectors are inner joined before this metric is computed.
    """

    @wraps(sklearn_metric)
    def func(prediction, reference, **kw):
        assert ismembership(prediction) and ismembership(reference)

        inner = pd.concat(
            {"prediction": prediction, "reference": reference},
            axis=1,
            join="inner",
            copy=False,
        )
        prediction_codes = pd.Categorical(inner.prediction).codes.astype(
            np.int64
        )
        reference_codes = pd.Categorical(inner.reference).codes.astype(np.int64)

        return sklearn_metric(reference_codes, prediction_codes, **kw)

    return func


def cluster_homogeneity(prediction, reference):
    """Cluster homogeneity score (based on conditional entropy).

    This wraps scikit-learn's `homogeneity score function <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.homogeneity_score.html>`.

    Args:
        prediction (Series):  membership vector for predicted clusters, i.e. a pandas Series indexed by mention ids and with values representing predicted cluster assignment.
        reference (Series):  membership vector for reference clusters, i.e. a pandas Series indexed by mention ids and with values representing reference cluster assignment.

    Returns:
        float: homogeneity score

    Notes:
        * The prediction and reference membership vectors are inner joined before this metric is computed.
    """
    return wrap_sklearn_metric(sm.homogeneity_score)(prediction, reference)


def cluster_completeness(prediction, reference):
    """Cluster completeness score (based on conditional entropy)

    This wraps scikit-learn's `completeness score function <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.completeness_score.html>`.

    Args:
        prediction (Series):  membership vector for predicted clusters, i.e. a pandas Series indexed by mention ids and with values representing predicted cluster assignment.
        reference (Series):  membership vector for reference clusters, i.e. a pandas Series indexed by mention ids and with values representing reference cluster assignment.

    Returns:
        float: completeness score

    Notes:
        * The prediction and reference membership vectors are inner joined before this metric is computed.
    """
    return wrap_sklearn_metric(sm.completeness_score)(prediction, reference)


def cluster_v_measure(prediction, reference, beta=1.0):
    """Compute the V-measure.

    This wraps scikit-learn's `V-measure function <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.v_measure_score.html#sklearn-metrics-v-measure-score>`.

    Args:
        prediction (Series):  membership vector for predicted clusters, i.e. a pandas Series indexed by mention ids and with values representing predicted cluster assignment.
        reference (Series):  membership vector for reference clusters, i.e. a pandas Series indexed by mention ids and with values representing reference cluster assignment.

    Returns:
        float: V-measure

    Notes:
        * The prediction and reference membership vectors are inner joined before this metric is computed.
    """

    return wrap_sklearn_metric(sm.v_measure_score)(
        prediction, reference, beta=beta
    )


def rand_score(prediction, reference):
    """Compute the Rand index.

    This wraps scikit-learn's `rand index function <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.rand_score.html#sklearn.metrics.rand_score>`.

    Args:
        prediction (Series):  membership vector for predicted clusters, i.e. a pandas Series indexed by mention ids and with values representing predicted cluster assignment.
        reference (Series):  membership vector for reference clusters, i.e. a pandas Series indexed by mention ids and with values representing reference cluster assignment.

    Returns:
        float: rand index

    Notes:
        * The prediction and reference membership vectors are inner joined before this metric is computed.
    """
    return wrap_sklearn_metric(sm.rand_score)(prediction, reference)


def adjusted_rand_score(prediction, reference):
    """Compute the adjusted Rand index.

    This wraps scikit-learn's `adjusted rand score function <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html#sklearn-metrics-adjusted-rand-score>`.

    Args:
        prediction (Series):  membership vector for predicted clusters, i.e. a pandas Series indexed by mention ids and with values representing predicted cluster assignment.
        reference (Series):  membership vector for reference clusters, i.e. a pandas Series indexed by mention ids and with values representing reference cluster assignment.

    Returns:
        float: adjusted rand index

    Notes:
        * The prediction and reference membership vectors are inner joined before this metric is computed.
    """

    return wrap_sklearn_metric(sm.adjusted_rand_score)(prediction, reference)
