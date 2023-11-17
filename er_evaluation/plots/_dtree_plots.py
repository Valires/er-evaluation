import numpy as np
import plotly.graph_objects as go

from er_evaluation.error_analysis import fit_dt_regressor
from er_evaluation.plots._dtree_data import build_sunburst_data, create_igraph_tree


def make_dt_regressor_plot(
    error_metrics,
    weights,
    features_df,
    numerical_features,
    categorical_features,
    type="sunburst",
    criterion="squared_error",
    **kwargs,
):
    """
    Fit a decision tree regressor to the data and create an interactive sunburst chart visualization of the resulting tree.

    Parameters:
    y (Series): Cluster-wise error metrics.
    weights (Series): The sample weights to use during model fitting.
    features_df (DataFrame): The input features for each cluster as a Pandas DataFrame.
    numerical_features (list): A list of column names corresponding to the numerical features in the DataFrame.
    categorical_features (list): A list of column names corresponding to the categorical features in the DataFrame.
    **kwargs: Additional keyword arguments to pass to the fit_dt_regressor function.

    Returns:
    plotly.graph_objs._sunburst.Sunburst: An interactive sunburst chart visualization of the fitted decision tree.

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> from er_evaluation.error_analysis import error_indicator
        >>> prediction = pd.Series([0, 1, 1])
        >>> reference = pd.Series([0, 1, 0])
        >>> y = error_indicator(prediction, reference)
        >>> weights = np.array([1, 1])
        >>> features_df = pd.DataFrame({'feature1': [1, 2], 'feature2': [4, 5]})
        >>> numerical_features = ['feature1']
        >>> categorical_features = ['feature2']
        >>> fig = make_dt_regressor_plot(y, weights, features_df, numerical_features, categorical_features)  # doctest: +SKIP
    """
    model = fit_dt_regressor(
        features_df,
        error_metrics,
        numerical_features,
        categorical_features,
        sample_weights=weights,
        criterion=criterion,
        **kwargs,
    )

    dt_regressor = model.named_steps["regressor"]
    preprocessor = model.named_steps["preprocessor"]
    feature_names = [x.split("__")[1] for x in preprocessor.get_feature_names_out()]
    X = preprocessor.transform(features_df)

    if type == "sunburst":
        return plot_dt_regressor_sunburst(
            dt_regressor, X, error_metrics, feature_names, weights=weights, label="Avg. Error"
        )
    elif type == "treemap":
        return plot_dt_regressor_treemap(
            dt_regressor, X, error_metrics, feature_names, weights=weights, label="Avg. Error"
        )
    elif type == "tree":
        return plot_dt_regressor_tree(dt_regressor, feature_names)
    else:
        raise ValueError("Argument 'type' should be one of 'sunburst' or 'treemap'.")


def plot_dt_regressor_tree(dt_regressor, feature_names):
    """
    Creates a tree plot of a decision tree regressor.

    Args:
        dt_regressor (DecisionTreeRegressor): A fitted decision tree regressor model.
        X (numpy array or pandas DataFrame): The input features used to fit the model.
        y (numpy array or pandas Series): The target values used to fit the model.
        feature_names (list of str): The names of the input features.
        weights (Series, optional): Sampling weights for y. Default is None.
        label (str, optional): The label for the color scale. Default is "Value".
        color_function (function, optional): A function applied to the subset of y values within each node to determine node color.If None, the predicted value for each node will be used as the color. Default is None.

    Returns:
        plotly.graph_objs.Figure: A tree plot of the decision tree regressor.

    Examples:
        >>> from sklearn.tree import DecisionTreeRegressor
        >>> import numpy as np
        >>> X = np.array([[1], [2], [3], [4], [5]])
        >>> y = np.array([0, 1, 0, 1, 0])
        >>> dt_regressor = DecisionTreeRegressor(max_depth=2)
        >>> dt_regressor.fit(X, y)  # doctest: +SKIP
        >>> feature_names = ['x']
        >>> plot_dt_regressor_tree(dt_regressor, feature_names)  # doctest: +SKIP
    """
    g, labels, node_sizes, colors = create_igraph_tree(dt_regressor, feature_names)
    layout = g.layout_reingold_tilford(mode="in", root=[0])

    edge_x = []
    edge_y = []
    for edge in g.es:
        source, target = edge.source, edge.target
        x0, y0 = layout[source]
        x1, y1 = layout[target]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(color="black", width=1), hoverinfo="none")

    node_x = [layout[i][0] for i in range(g.vcount())]
    node_y = [layout[i][1] for i in range(g.vcount())]
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers",
        marker=dict(
            size=400 * np.sqrt(node_sizes) / np.sum(np.sqrt(node_sizes)),
            color=colors,
            colorscale="Blues",
            showscale=True,
            colorbar=dict(title="Avg. Error"),
            line=dict(width=1, color="grey"),
        ),
        text=[f"Avg. Error: {value:.2f}<br>Size: {size}" for value, size in zip(colors, node_sizes)],
        hoverinfo="text",
        textfont=dict(size=10),
    )

    annotations = [
        go.layout.Annotation(
            x=layout[i][0],
            y=layout[i][1],
            text=labels[i],
            xanchor="center",
            yanchor="top" if i % 2 == 0 else "bottom",
            xref="x",
            yref="y",
            showarrow=False,
            font=dict(size=10),
            bgcolor="rgba(255, 255, 255, 0.75)",
            borderpad=1,
        )
        for i in range(g.vcount())
    ]

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor="white",
            annotations=annotations,
        ),
    )
    fig.update_layout(yaxis_autorange="reversed")
    fig.update_layout(margin=dict(t=0, l=0, r=0, b=0))

    return fig


def plot_dt_regressor_sunburst(dt_regressor, X, y, feature_names, weights=None, label="Value", color_function=None):
    """
    Creates a sunburst plot of a decision tree regressor.

    Args:
        dt_regressor (DecisionTreeRegressor): A fitted decision tree regressor model.
        X (numpy array or pandas DataFrame): The input features used to fit the model.
        y (numpy array or pandas Series): The target values used to fit the model.
        feature_names (list of str): The names of the input features.
        weights (Series, optional): Sampling weights for y. Default is None.
        label (str, optional): The label for the color scale. Default is "Value".
        color_function (function, optional): A function applied to the subset of y values within each node to determine node color.If None, the predicted value for each node will be used as the color. Default is None.

    Returns:
        plotly.graph_objs.Figure: A sunburst plot of the decision tree regressor.

    Examples:
        >>> from sklearn.tree import DecisionTreeRegressor
        >>> import numpy as np
        >>> X = np.array([[1], [2], [3], [4], [5]])
        >>> y = np.array([0, 1, 0, 1, 0])
        >>> dt_regressor = DecisionTreeRegressor(max_depth=2)
        >>> dt_regressor.fit(X, y)  # doctest: +SKIP
        >>> feature_names = ['x']
        >>> fig = plot_dt_regressor_sunburst(dt_regressor, X, y, feature_names)   # doctest: +SKIP
    """
    sunburst_data = build_sunburst_data(
        dt_regressor, feature_names, X, y, weights=weights, color_function=color_function, label=label
    )

    fig = go.Figure(
        go.Sunburst(
            ids=[item["id"] for item in sunburst_data],
            labels=[item["label"] for item in sunburst_data],
            parents=[item["parent"] for item in sunburst_data],
            values=[item["value"] for item in sunburst_data],
            marker=dict(
                colors=[item["color"] for item in sunburst_data], colorbar=dict(title=label), colorscale="Blues"
            ),
            hovertemplate="<b>"
            + label
            + ":</b> %{color:.2f}<br><b>Size:</b> %{value}<br><b>Path:</b><br>    %{customdata}<extra></extra>",
            branchvalues="total",
            customdata=["<br>    ".join(item["path"] + [item["label"]]) for item in sunburst_data],
        )
    )

    fig.update_layout(margin=dict(t=0, l=0, r=0, b=0))

    return fig


def plot_dt_regressor_treemap(dt_regressor, X, y, feature_names, weights=None, label="Value", color_function=None):
    """
    Creates a treemap plot of a decision tree regressor.

    Args:
        dt_regressor (DecisionTreeRegressor): A fitted decision tree regressor model.
        X (numpy array or pandas DataFrame): The input features used to fit the model.
        y (numpy array or pandas Series): The target values used to fit the model.
        feature_names (list of str): The names of the input features.
        weights (Series, optional): Sampling weights for y. Default is None.
        label (str, optional): The label for the color scale. Default is "Value".
        color_function (function, optional): A function applied to the subset of y values within each node to determine node color.If None, the predicted value for each node will be used as the color. Default is None.

    Returns:
        plotly.graph_objs.Figure: A treemap plot of the decision tree regressor.

    Examples:
        >>> from sklearn.tree import DecisionTreeRegressor
        >>> import numpy as np
        >>> X = np.array([[1], [2], [3], [4], [5]])
        >>> y = np.array([0, 1, 0, 1, 0])
        >>> dt_regressor = DecisionTreeRegressor(max_depth=2)
        >>> dt_regressor.fit(X, y)  # doctest: +SKIP
        >>> feature_names = ['x']
        >>> plot_dt_regressor_treemap(dt_regressor, X, y, feature_names)  # doctest: +SKIP
    """
    sunburst_data = build_sunburst_data(
        dt_regressor, feature_names, X, y, weights=weights, color_function=color_function, label=label
    )

    fig = go.Figure(
        go.Treemap(
            ids=[item["id"] for item in sunburst_data],
            labels=[item["label"] for item in sunburst_data],
            parents=[item["parent"] for item in sunburst_data],
            values=[item["value"] for item in sunburst_data],
            marker=dict(
                colors=[item["color"] for item in sunburst_data], colorbar=dict(title=label), colorscale="Blues"
            ),
            hovertemplate="<b>"
            + label
            + ":</b> %{color:.2f}<br><b>Size:</b> %{value}<br><b>Path:</b><br>    %{customdata}<extra></extra>",
            customdata=["<br>    ".join(item["path"] + [item["label"]]) for item in sunburst_data],
            branchvalues="total",
        )
    )

    fig.update_layout(margin=dict(t=0, l=0, r=0, b=0))

    return fig
