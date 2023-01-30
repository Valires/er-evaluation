import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from er_evaluation.data_structures import MembershipVector
from er_evaluation.error_analysis import error_metrics
from er_evaluation.estimators import (
    cluster_precision_design_estimate,
    cluster_recall_design_estimate,
    estimates_table,
    pairwise_precision_design_estimate,
    pairwise_recall_design_estimate,
)
from er_evaluation.metrics import cluster_precision, cluster_recall, metrics_table, pairwise_precision, pairwise_recall
from er_evaluation.summary import cluster_hill_number, cluster_sizes_distribution, summary_statistics

DEFAULT_METRICS = {
    "Pairwise precision": pairwise_precision,
    "Pairwise recall": pairwise_recall,
    "Cluster precision": cluster_precision,
    "Cluster recall": cluster_recall,
}

DEFAULT_COMPARISON_METRICS = {
    "Pairwise precision": pairwise_precision,
    "Pairwise recall": pairwise_recall,
}

DEFAULT_ESTIMATORS = {
    "Pairwise precision": pairwise_precision_design_estimate,
    "Pairwise recall": pairwise_recall_design_estimate,
    "Cluster precision": cluster_precision_design_estimate,
    "Cluster recall": cluster_recall_design_estimate,
}


def compare_plots(*figs, names=None, marker="color", marker_values=None):
    r"""
    Combine multiple figures into one.
    """
    assert names is None or len(names) == len(figs)
    assert marker_values is None or len(marker_values) == len(figs)

    combined = go.Figure()
    for i, fig in enumerate(figs):
        if marker_values is not None:
            marker_val = marker_values[i]
        else:
            marker_val = i
        fig.update_traces(marker={marker: marker_val}, showlegend=True)
        if names is not None:
            fig.update_traces(name=names[i])
        elif fig.data[0].name == "":
            fig.update_traces(name=i)
        combined.add_traces(fig.data)

        combined.update_xaxes(range=fig.layout.xaxis.range)
        combined.update_yaxes(range=fig.layout.yaxis.range)

    return combined


def plot_cluster_sizes_distribution(membership, groupby=None, name=None, normalize=False):
    r"""
    Plot the cluster size distribution

    Args:
        membership (_type_): Membership vector.
        groupby (_type_, optional): Series to group by. Defaults to None.
        name (Series, optional): Name of the plot (useful when combining multiple plots together). Defaults to None.
        normalize: Wether or not to normalize

    Returns:
        Figure: Cluster size distribution plot.
    """
    membership = MembershipVector(membership, dropna=True)

    if groupby is not None:
        groupby = MembershipVector(groupby, dropna=True)
        if groupby.name is None:
            groupby.name = "groupby"
        membership = membership[membership.index.isin(groupby.index)]
        groupby = groupby[groupby.index.isin(membership.index)]

        cs_dist = membership.groupby(groupby).apply(cluster_sizes_distribution).reset_index(name="cs_size")
        if normalize:
            cs_dist.cs_size = cs_dist.cs_size / cs_dist.cs_size.sum()
            y_label = "proportion"
        else:
            y_label = "count"

        fig = px.bar(
            x=cs_dist.level_1,
            y=cs_dist.cs_size,
            color=cs_dist[groupby.name],
            labels={"x": "Cluster size", "y": y_label, "color": groupby.name},
            barmode="group",
        )
    else:
        cs_dist = cluster_sizes_distribution(membership)
        if normalize:
            cs_dist = cs_dist / cs_dist.sum()
            y_label = "proportion"
        else:
            y_label = "count"

        fig = px.bar(
            x=cs_dist.index,
            y=cs_dist.values,
            labels={"x": "Cluster size", "y": y_label},
            barmode="group",
        )

    if name is not None:
        fig.update_traces(name=name)

    return fig


def plot_entropy_curve(membership, q_range=None, groupby=None, name=None):
    r"""
    Plot the Hill number entropy curve

    Args:
        membership (_type_): Membership vector.
        groupby (_type_, optional): Series to group by. Defaults to None.
        name (Series, optional): Name of the plot (useful when combining multiple plots together). Defaults to None.

    Returns:
        Figure: Hill number entropy curve.
    """
    membership = MembershipVector(membership)

    if q_range is None:
        q_range = np.linspace(0, 2)

    if groupby is not None:
        groupby = MembershipVector(groupby, dropna=True)
        if groupby.name is None:
            groupby.name = "groupby"
        membership = membership[membership.index.isin(groupby.index)]
        groupby = groupby[groupby.index.isin(membership.index)]

        hill_numbers = (
            membership.groupby(groupby)
            .apply(lambda x: pd.Series([cluster_hill_number(x, q) for q in q_range], index=q_range))
            .reset_index(name="hill_numbers")
        )

        fig = px.line(
            x=hill_numbers.level_1,
            y=hill_numbers.hill_numbers,
            color=hill_numbers[groupby.name],
            title="Hill Numbers entropy curve",
            labels={"x": "q", "y": "Hill Number", "color": groupby.name},
        )
    else:
        hill_numbers = [cluster_hill_number(membership, q) for q in q_range]
        fig = px.line(
            x=q_range,
            y=hill_numbers,
            title="Hill Numbers entropy curve",
            labels={"x": "q", "y": "Hill Number"},
        )

    if name is not None:
        fig.update_traces(name=name)

    return fig


def plot_summaries(predictions, names=None, type="line", **kwargs):
    """
    Plot summary statistics

    Args:
        predictions (dict): Dictionary of predictions for which to plot the summary statistics.
        names (Series, optional): Series with cluster element names. Used to compute the homonymy rate and the name variation rate. Defaults to None.
        type (str, optional): One of "line" for a line plot, or "bar" for a bar plot. Defaults to "line".

    Returns:
        plotly Figure

    Examples:
        >>> from er_evaluation.datasets import load_pv_disambiguations
        >>> predictions, _ = load_pv_disambiguations()
        >>> fig = plot_summaries(predictions)
        >>> fig.show() # doctest: +SKIP
    """
    summaries = pd.DataFrame.from_records([summary_statistics(pred, names=names) for pred in predictions.values()])
    summaries["prediction"] = predictions.keys()

    if type == "line":
        plt = px.line
    elif type == "bar":
        plt = px.bar
    else:
        raise ValueError(f"Unknown plot type {type}. Should be one of ['line', 'bar'].")

    fig = make_subplots(
        rows=2 if names is None else 3,
        cols=2,
        subplot_titles=(
            "Average Cluster Size",
            "Matching Rate",
            "Number of distinct cluster sizes",
            "Entropy",
            "Name Variation Rate",
            "Homonymy Rate",
        ),
        shared_xaxes=True,
        **kwargs
    )

    plots = [["average_cluster_size", "matching_rate"], ["H0", "H1"]]
    for i, row in enumerate(plots):
        for j, col in enumerate(row):
            fig.add_trace(plt(summaries, x="prediction", y=col).data[0], row=i + 1, col=j + 1)

    if names is not None:
        extra_plots = ["name_variation_rate", "homonymy_rate"]
        for i, col in enumerate(extra_plots):
            fig.add_trace(plt(summaries, x="prediction", y=col).data[0], row=3, col=i + 1)

    fig.update_annotations(font_size=14)
    fig.update_layout(title_text="Disambiguation Summary Statistics")

    return fig


def plot_metrics(predictions, reference, metrics=DEFAULT_METRICS, type="line", **kwargs):
    """
    Plot performance metrics.

    Args:
        predictions (dict): Dictionary of predictions for which to plot the performance metrics.
        reference (Series): Reference membership vector representing the ground truth.
        metrics (dict, optional): Dictionary of metrics to display. Defaults to DEFAULT_METRICS.
        type (str, optional): One of "line" for a line plot or "bar" for a bar plot. Defaults to "line".

    Returns:
        plotly Figure

    Examples:
        >>> from er_evaluation.datasets import load_pv_disambiguations
        >>> predictions, reference = load_pv_disambiguations()
        >>> fig = plot_metrics(predictions, reference)
        >>> fig.show() # doctest: +SKIP
    """
    table = metrics_table(predictions, {"reference": reference}, metrics=metrics)

    if type == "line":
        fig = px.line(table, x="prediction", y="value", color="metric", **kwargs)
    elif type == "bar":
        fig = px.bar(table, x="metric", y="value", color="prediction", barmode="group", **kwargs)
    else:
        raise ValueError(f"Unknown plot type {type}. Should be one of ['line', 'bar'].")

    fig.update_layout(title_text="Performance metrics")

    return fig


def plot_estimates(predictions, sample_weights, estimators=DEFAULT_ESTIMATORS, type="line", **kwargs):
    """
    Plot representative performance estimates.

    Args:
        predictions (dict): Dictionary of predictions for which to plot the performance estimates.
        sample_weights (dict): Dictionary with an element named "sample" containing sampled clusters, and an element named "weights" containing the sampling weights.
        estimators (dict, optional): Dictionary of estimators to use. Defaults to DEFAULT_ESTIMATORS.
        type (str, optional): One of "line" for a line plot or "bar" for a bar plot. Defaults to "line".

    Returns:
        plotly Figure.

    Examples:
        >>> from er_evaluation.datasets import load_pv_disambiguations
        >>> predictions, reference = load_pv_disambiguations()
        >>> fig = plot_estimates(predictions, {"sample": reference, "weights": "cluster_size"})
        >>> fig.show() # doctest: +SKIP
    """
    table = estimates_table(predictions, samples_weights={"sample": sample_weights}, estimators=estimators)

    if type == "line":
        fig = px.line(table, x="prediction", y="value", color="estimator", error_y="std", **kwargs)
    elif type == "bar":
        fig = px.bar(table, x="estimator", y="value", color="prediction", barmode="group", error_y="std", **kwargs)
    else:
        raise ValueError(f"Unknown plot type {type}. Should be one of ['line', 'bar'].")

    fig.update_layout(title_text="Performance estimates")

    return fig


def plot_cluster_errors(
    prediction, reference, x="expected_relative_extra", y="expected_relative_missing", opacity=0.5, **kwargs
):
    """
    Scatter plot of two cluster-wise error metrics.

    Metrics that can be plotted are:

    * **expected_extra** (see :meth:`er_evaluation.error_analysis.expected_extra`)
    * **expected_relative_extra** (see :meth:`er_evaluation.error_analysis.expected_relative_extra`)
    * **expected_missing** (see :meth:`er_evaluation.error_analysis.expected_missing`)
    * **expected_relative_missing** (see :meth:`er_evaluation.error_analysis.expected_relative_missing`)
    * **error_indicator** (see :meth:`er_evaluation.error_analysis.error_indicator`)

    Args:
        prediction (Series): Predicted clustering.
        reference (Series): Reference clustering.
        x (str, optional): x-axis metric to plot. Defaults to "expected_relative_extra".
        y (str, optional): y-axis metric to plot. Defaults to "expected_relative_missing".
        opacity (float, optional): Opacity. Defaults to 0.5.

    Returns:
        plotly Figure
    """
    errors = error_metrics(prediction, reference)

    fig = px.scatter(errors, x=x, y=y, opacity=opacity, marginal_x="histogram", marginal_y="histogram", **kwargs)
    fig.update_layout(title_text="Cluster-Wise Error Metrics")

    return fig


def plot_comparison(predictions, metrics=DEFAULT_COMPARISON_METRICS, **kwargs):
    """
    Plot metrics computed for all prediction pairs.

    This is meant to help investigate the similarity and differences between a set of disambiguations.

    Args:
        predictions (dict): dictionary of membership vectors to compare.
        metrics (dict, optional): Dictionary of metrics to compute. Defaults to DEFAULT_COMPARISON_METRICS.

    Examples:
        >>> from er_evaluation.datasets import load_pv_disambiguations
        >>> predictions, _ = load_pv_disambiguations()
        >>> fig = plot_comparison(predictions)
        >>> fig.show() # doctest: +SKIP
    """

    def metrics_matrix(predictions, metrics):
        matrix = np.zeros((len(metrics), len(predictions), len(predictions)))
        for col, metric in enumerate(metrics.values()):
            for i, pred_i in enumerate(predictions.values()):
                for j, pred_j in enumerate(predictions.values()):
                    matrix[col, i, j] = metric(pred_i, pred_j)

        return matrix

    matrix = metrics_matrix(predictions, metrics)

    keys = list(predictions.keys())
    fig = px.imshow(matrix, x=keys, y=keys, facet_col=0, facet_col_wrap=min(len(metrics), 2), aspect="equal", **kwargs)
    fig.update_layout(title_text="Disambiguation Similarity")
    for i, name in enumerate(metrics.keys()):
        fig.layout.annotations[i].update(text=name, font_size=14)

    fig.update_xaxes(title="Prediction")
    fig.update_yaxes(title="Reference")

    return fig
