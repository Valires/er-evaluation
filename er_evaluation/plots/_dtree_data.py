import numpy as np
from igraph import Graph

def create_igraph_tree(dt_regressor, feature_names):
    """
    Recursively builds an igraph Graph for a given decision tree regressor.

    Args:
        dt_regressor (sklearn.tree.DecisionTreeRegressor): Fitted decision tree regressor
        feature_names (list of str): The names of the input features.
    
    Returns:
        igraph.Graph: 
            An igraph tree representation of the decision tree regressor.
        list of str:
            A list of node labels containing the decision rules.
        list of int:
            A list of node sizes, proportional to the number of weighted samples in each node.
        list of float:
            A list of colors representing the predicted value at each node.
    
    Examples:
    >>> from sklearn.tree import DecisionTreeRegressor
    >>> import numpy as np

    >>> # Generate some example data
    >>> X = np.random.randn(100, 2)
    >>> y = np.random.randn(100)
    >>> feature_names = ['feature1', 'feature2']
    
    >>> # Fit a decision tree regressor
    >>> dt_regressor = DecisionTreeRegressor()
    >>> dt_regressor.fit(X, y)
    
    >>> # Create an igraph tree representation
    >>> g, labels, node_sizes, colors = create_igraph_tree(dt_regressor, feature_names)
    """
    n_nodes = dt_regressor.tree_.node_count
    children_left = dt_regressor.tree_.children_left
    children_right = dt_regressor.tree_.children_right
    feature = dt_regressor.tree_.feature
    threshold = dt_regressor.tree_.threshold
    value = dt_regressor.tree_.value

    g = Graph()
    g.add_vertices(n_nodes)
    labels = [""] * n_nodes  # Initialize an empty label list of the same size as the number of nodes
    node_sizes = []
    colors = []

    def add_nodes(node_id, parent_decision=None):
        if node_id == -1:
            return

        left_child_id = children_left[node_id]
        right_child_id = children_right[node_id]

        if feature[node_id] != -2:
            decision_true = f'{feature_names[feature[node_id]]} <= {threshold[node_id]:.2f}'
            decision_false = f'{feature_names[feature[node_id]]} > {threshold[node_id]:.2f}'

        # Assign the parent's decision to the current node label
        if parent_decision is not None:
            labels[node_id] = parent_decision

        node_sizes.append(dt_regressor.tree_.weighted_n_node_samples[node_id])
        colors.append(value[node_id][0][0])

        if left_child_id != -1:
            g.add_edge(node_id, left_child_id)
            add_nodes(left_child_id, decision_true)

        if right_child_id != -1:
            g.add_edge(node_id, right_child_id)
            add_nodes(right_child_id, decision_false)

    add_nodes(0)
    return g, labels, node_sizes, colors


def build_sunburst_data(dt_regressor, feature_names, X, y, weights=None, color_function=None, label="Value"):
    """
    Recursively builds the sunburst data for a given decision tree regressor.

    Args:
        dt_regressor (sklearn.tree.DecisionTreeRegressor): Fitted decision tree regressor
        feature_names (list of str): The names of the input features.
        X (numpy array or pandas DataFrame): The input features used to fit the model.
        y (numpy array or pandas Series): The target values used to fit the model.
        weights (Series): Sampling weights for y.
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
    if weights is None:
        weights = np.ones_like(y)

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
        node_weights = weights[node_sample_indices]

        if color_function is None:
            node_color = tree.value[node_id][0][0]
        else:
            node_color = color_function(node_y_values)

        node_label = "" if parent is None else f"{feature_names[parent_feature]} {'≤' if decision == 'True' else '>'} {parent_threshold:.2f}"

        node_data.append({
            "id": f"node_{node_id}",
            "parent": f"node_{parent}" if parent is not None else "",
            "label": node_label,
            "value": np.sum(node_weights),
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
