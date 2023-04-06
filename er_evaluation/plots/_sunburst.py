import plotly.graph_objects as go

def build_sunburst_data(dt_regressor, feature_names, X, y, color_function=None, label="Value"):
    """
    Recursively builds the sunburst data for a given decision tree regressor.

    Args:
        dt_regressor (sklearn.tree.DecisionTreeRegressor): Fitted decision tree regressor
        feature_names (list of str): The names of the input features.
        X (numpy array or pandas DataFrame): The input features used to fit the model.
        y (numpy array or pandas Series): The target values used to fit the model.
        label (str, optional): The label for the color scale. Default is "Value".
        color_function (function, optional): A function applied to the subset of y values within each node to determine node color.If None, the predicted value for each node will be used as the color. Default is None.

    Returns:
        list of dict: A list of dictionaries containing the sunburst data for the current node and its children.
    
    Examples:
        >>> from sklearn.tree import DecisionTreeRegressor
        >>> import numpy as np
        >>> X = np.array([[1], [2], [3], [4], [5]])
        >>> y = np.array([2, 4, 6, 8, 10])
        >>> dt_regressor = DecisionTreeRegressor(max_depth=2)
        >>> dt_regressor.fit(X, y)
        >>> feature_names = ['x']
        >>> sunburst_data = build_sunburst_data(dt_regressor, feature_names, X, y)
        >>> len(sunburst_data)
            7
        >>> sunburst_data[6]  # doctest: +SKIP
            {'id': 'node_6',
            'parent': 'node_4',
            'label': 'Value: 9.00',
            'value': 2,
            'color': 9.0,
            'path': ['Root', 'x > 2.50', 'Value: 9.00']}
    """
    def build_sunburst_data_recursive(node_id, tree, feature_names, X, y, decision_paths, parent=None, parent_feature=None, parent_threshold=None, decision=None, path=[]):
        """
        Args:
            node_id (int): The current node's id.
            tree (sklearn.tree._tree.Tree): The tree object from the decision tree regressor.
            decision_paths (numpy array): A boolean array indicating the decision path for each sample.
            parent (int, optional): The parent node's id. Default is None.
            parent_feature (int, optional): The parent node's feature index. Default is None.
            parent_threshold (float, optional): The parent node's threshold. Default is None.
            decision (str, optional): The decision made in the parent node. Either 'True' or 'False'. Default is None.
            path (str, optional): The decision path leading to the current node. Default is an empty string.
        """
        node_data = []

        node_sample_indices = decision_paths[:, node_id].toarray().flatten().astype(bool)
        node_y_values = y[node_sample_indices]
        if color_function is None:
            node_color = tree.value[node_id][0][0]
        else:
            node_color = color_function(node_y_values)

        node_label = "" if parent is None else f"{feature_names[parent_feature]} {'≤' if decision == 'True' else '>'} {parent_threshold:.2f}"

        node_data.append({
            "id": f"node_{node_id}",
            "parent": f"node_{parent}" if parent is not None else "",
            "label": node_label,
            "value": tree.n_node_samples[node_id],
            "color": node_color,
            "path": path
        })

        if tree.children_left[node_id] != -1:
            left_child_id = tree.children_left[node_id]
            right_child_id = tree.children_right[node_id]

            new_path = path + [node_label] if len(node_label) > 0 else path
            node_data.extend(build_sunburst_data_recursive(left_child_id, tree, feature_names, X, y, decision_paths, node_id, tree.feature[node_id], tree.threshold[node_id], 'True', new_path))
            node_data.extend(build_sunburst_data_recursive(right_child_id, tree, feature_names, X, y, decision_paths, node_id, tree.feature[node_id], tree.threshold[node_id], 'False', new_path))

        return node_data
    
    tree = dt_regressor.tree_
    decision_paths = dt_regressor.decision_path(X)
    return build_sunburst_data_recursive(0, tree, feature_names, X, y, decision_paths)


def plot_dt_regressor_sunburst(dt_regressor, X, y, feature_names, label="Value", color_function=None):
    """
    Creates a sunburst plot of a decision tree regressor.

    Args:
        dt_regressor (DecisionTreeRegressor): A fitted decision tree regressor model.
        X (numpy array or pandas DataFrame): The input features used to fit the model.
        y (numpy array or pandas Series): The target values used to fit the model.
        feature_names (list of str): The names of the input features.
        label_children (bool, optional): If True, labels are placed in the children nodes instead of parent nodes. Default is False.
        label (str, optional): The label for the color scale. Default is "Value".
        color_function (function, optional): A function applied to the subset of y values within each node to determine node color.If None, the predicted value for each node will be used as the color. Default is None.

    Returns:
        plotly.graph_objs.Figure: A sunburst plot of the decision tree regressor.
    
    Example:
        >>> from sklearn.tree import DecisionTreeRegressor
        >>> import numpy as np
        >>> X = np.array([[1], [2], [3], [4], [5]])
        >>> y = np.array([2, 4, 6, 8, 10])
        >>> dt_regressor = DecisionTreeRegressor(max_depth=2)
        >>> dt_regressor.fit(X, y)
        >>> feature_names = ['x']
        >>> fig = plot_dt_regressor_sunburst(dt_regressor, X, y, feature_names)
        >>> isinstance(fig, go.Figure)
            True
        >>> fig.data[0]  # doctest: +SKIP
            Sunburst({
                'branchvalues': 'total',
                'hovertemplate': '<b>%{label}</b><br>Size: %{value}<br>Value: %{color:.2f}<extra></extra>',
                'ids': [node_0, node_1, node_2, node_3, node_4, node_5, node_6],
                'labels': [Root, x ≤ 2.50, Value: 2.00, Value: 4.00, x > 2.50, Value: 6.00,
                        Value: 9.00],
                'marker': {'colorbar': {'title': {'text': 'Value'}}, 'colors': [6.0, 3.0, 2.0, 4.0, 8.0, 6.0, 9.0]},
                'parents': [, node_0, node_1, node_1, node_0, node_4, node_4],
                'values': [5, 2, 1, 1, 3, 1, 2]
            })
    """
    sunburst_data = build_sunburst_data(dt_regressor, feature_names, X, y, color_function=color_function, label=label)

    fig = go.Figure(go.Sunburst(
        ids=[item["id"] for item in sunburst_data],
        labels=[item["label"] for item in sunburst_data],
        parents=[item["parent"] for item in sunburst_data],
        values=[item["value"] for item in sunburst_data],
        marker=dict(
            colors=[item["color"] for item in sunburst_data],
            colorbar=dict(title=label)
        ),
        hovertemplate='<b>'+label+':</b> %{color:.2f}<br><b>Size:</b> %{value}<br><b>Path:</b><br>    %{customdata}<extra></extra>',
        branchvalues="total",
        customdata=["<br>    ".join(item["path"] + [item["label"]]) for item in sunburst_data],
    ))

    fig.update_layout(margin=dict(t=0, l=0, r=0, b=0))

    return fig
