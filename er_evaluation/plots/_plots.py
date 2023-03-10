import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from er_evaluation.data_structures import MembershipVector
from er_evaluation.error_analysis import error_metrics
from er_evaluation.estimators import (
    b_cubed_precision_estimator,
    b_cubed_recall_estimator,
    estimates_table,
    pairwise_precision_estimator,
    pairwise_recall_estimator,
    summary_estimates_table,
    pairwise_f_estimator,
)
from er_evaluation.metrics import (
    cluster_precision,
    cluster_recall,
    metrics_table,
    pairwise_precision,
    pairwise_recall,
    pairwise_f,
)
from er_evaluation.summary import cluster_hill_number, cluster_sizes_distribution, summary_statistics

DEFAULT_METRICS = {
    "Pairwise precision": pairwise_precision,
    "Pairwise recall": pairwise_recall,
    "Cluster precision": cluster_precision,
    "Cluster recall": cluster_recall,
}

DEFAULT_COMPARISON_METRICS = {
    "Pairwise precision": pairwise_precision,
    "Pairwise F1": pairwise_f,
}

DEFAULT_ESTIMATORS = {
    "Pairwise precision": pairwise_precision_estimator,
    "Pairwise recall": pairwise_recall_estimator,
    "B3 precision": b_cubed_precision_estimator,
    "B3 recall": b_cubed_recall_estimator,
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


def plot_summaries(predictions, names=None, type="line", line_shape="spline", markers=True, **kwargs):
    """
    Plot summary statistics

    Args:
        predictions (dict): Dictionary of predictions for which to plot the summary statistics.
        names (Series, optional): Series with cluster element names. Used to compute the homonymy rate and the name variation rate. Defaults to None.
        type (str, optional): One of "line" for a line plot, or "bar" for a bar plot. Defaults to "line".
        **kwargs (optional): Additional arguments to pass to plotly express for plot creation.

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
        plt = lambda *args, **kwargs: px.line(*args, line_shape=line_shape, markers=markers, **kwargs)
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
        **kwargs,
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


def add_ests_to_summaries(fig, predictions, sample, weights, names=None):
    params = summary_estimates_table(sample, weights, predictions, names)

    plots = [
        (1, 1, "Avg Cluster Size Estimate"),
        (1, 2, "Matching Rate Estimate"),
    ]
    if names is not None:
        plots += [(3, 1, "Name Variation Estimate"), (3, 2, "Homonymy Rate Estimate")]

    fig.update_traces(legendgroup=0, name="summary")
    fig.update_traces(selector=0, showlegend=True)
    for k, (i, j, estimate) in enumerate(plots):
        dat = params.query(f"estimate == '{estimate}'")
        n = len(dat)
        fig.add_trace(
            go.Scatter(
                name="estimate",
                x=dat["prediction"],
                y=dat["value"],
                line=dict(color="black", dash="dot", width=1, shape="spline"),
                marker=dict(color="black", size=4),
                legendgroup=1,
                showlegend=(k == 0),
            ),
            row=i,
            col=j,
        )
        fig.add_trace(
            go.Scatter(
                name="estimate",
                x=dat["prediction"],
                y=dat["value"] - 2 * dat["std"],
                line=dict(color="black", dash="dot", width=0, shape="spline"),
                mode="lines",
                legendgroup=1,
                showlegend=False,
            ),
            row=i,
            col=j,
        )
        fig.add_trace(
            go.Scatter(
                name="estimate",
                x=dat["prediction"],
                y=dat["value"] + 2 * dat["std"],
                fill="tonexty",
                fillcolor="rgba(0, 0, 0, 0.1)",
                line=dict(color="black", dash="dot", width=0, shape="spline"),
                mode="lines",
                legendgroup=1,
                showlegend=False,
            ),
            row=i,
            col=j,
        )

    return fig


def plot_metrics(predictions, reference, metrics=DEFAULT_METRICS, type="line", **kwargs):
    """
    Plot performance metrics.

    Args:
        predictions (dict): Dictionary of predictions for which to plot the performance metrics.
        reference (Series): Reference membership vector representing the ground truth.
        metrics (dict, optional): Dictionary of metrics to display. Defaults to DEFAULT_METRICS.
        type (str, optional): One of "line" for a line plot or "bar" for a bar plot. Defaults to "line".
        **kwargs (optional): Additional arguments to pass to plotly express for plot creation.

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


def plot_estimates(
    predictions, sample_weights, estimators=DEFAULT_ESTIMATORS, type="line", line_shape="spline", **kwargs
):
    """
    Plot representative performance estimates.

    Args:
        predictions (dict): Dictionary of predictions for which to plot the performance estimates.
        sample_weights (dict): Dictionary with an element named "sample" containing sampled clusters, and an element named "weights" containing the sampling weights.
        estimators (dict, optional): Dictionary of estimators to use. Defaults to DEFAULT_ESTIMATORS.
        type (str, optional): One of "line" for a line plot or "bar" for a bar plot. Defaults to "line".
        **kwargs (optional): Additional arguments to pass to plotly express for plot creation.

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
        fig = px.line(
            table,
            x="prediction",
            y="value",
            color="estimator",
            symbol="estimator",
            error_y="std",
            line_shape=line_shape,
            **kwargs,
        )
    elif type == "bar":
        fig = px.bar(table, x="estimator", y="value", color="prediction", barmode="group", error_y="std", **kwargs)
    else:
        raise ValueError(f"Unknown plot type {type}. Should be one of ['line', 'bar'].")

    fig.update_layout(title_text="Performance estimates")

    return fig


def plot_cluster_errors(
    prediction,
    reference,
    x="expected_relative_extra",
    y="expected_relative_missing",
    groupby=None,
    weights=None,
    opacity=0.5,
    **kwargs,
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
        groupby (Series, optional): Optional Series with grouping values (corresponding to color elements). Should be indexed by cluster identifier, with values corresponding to group assignment.
        weights (Series, optional): Optional Series with cluster weights. Should be indexed by cluster identifier, with values corresponding to cluster weight.
        opacity (float, optional): Opacity. Defaults to 0.5.
        **kwargs (optional): Additional arguments to pass to plotly express for plot creation.

    Returns:
        plotly Figure

    Note:
        Weights are not accounted for in the marginal histograms.

    """
    errors = error_metrics(prediction, reference)
    size = None
    color = None
    if weights is not None:
        size = "weight"
        weights.name = "weight"
        errors = errors.merge(weights, left_index=True, right_index=True, how="left")
    if groupby is not None:
        color = "color"
        groupby.name = "color"
        errors = errors.merge(groupby, left_index=True, right_index=True, how="left")

    fig = px.scatter(
        errors,
        x=x,
        y=y,
        opacity=opacity,
        marginal_x="histogram",
        marginal_y="histogram",
        size=size,
        color=color,
        **kwargs,
    )
    fig.update_layout(title_text="Cluster-Wise Error Metrics")

    return fig


def plot_comparison(predictions, metrics=DEFAULT_COMPARISON_METRICS, **kwargs):
    """
    Plot metrics computed for all prediction pairs.

    This is meant to help investigate the similarity and differences between a set of disambiguations.

    Args:
        predictions (dict): dictionary of membership vectors to compare.
        metrics (dict, optional): Dictionary of metrics to compute. Defaults to DEFAULT_COMPARISON_METRICS.
        **kwargs (optional): Additional arguments to pass to plotly express for plot creation.

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
