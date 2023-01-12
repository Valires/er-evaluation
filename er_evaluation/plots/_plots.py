import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from er_evaluation.summary import cluster_hill_number, cluster_sizes_distribution


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
    if groupby is not None:
        assert isinstance(groupby, pd.Series)
        assert groupby.index.equals(membership.index)

        cs_dist = membership.groupby(groupby).apply(cluster_sizes_distribution).reset_index(name="cs_size")
        if normalize:
            cs_dist.cs_size = cs_dist.cs_size / cs_dist.cs_size.sum()
            y_label = "proportion"
        else:
            y_label = "count"

        fig = px.bar(
            x=cs_dist.level_1,
            y=cs_dist.cs_size,
            color=cs_dist.level_0,
            labels={"x": "Cluster size", "y": y_label},
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

    if q_range is None:
        q_range = np.linspace(0, 2)

    if groupby is not None:
        assert isinstance(groupby, pd.Series)
        assert groupby.index.equals(membership.index)

        hill_numbers = (
            membership.groupby(groupby)
            .apply(lambda x: pd.Series([cluster_hill_number(x, q) for q in q_range], index=q_range))
            .reset_index(name="hill_numbers")
        )

        fig = px.line(
            x=hill_numbers.level_1,
            y=hill_numbers.hill_numbers,
            color=hill_numbers.level_0,
            title="Hill Numbers entropy curve",
            labels={"x": "q", "y": "Hill Number"},
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
