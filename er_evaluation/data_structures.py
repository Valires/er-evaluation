r"""
==========================
Clustering Data Structures
==========================

This module contains utilities defining clustering data structures (graph, membership vector, clusters dictionary, and pairwise links list) and allowing transformation between them.

-----------
Definitions
-----------

A clustering of a set of elements :math:`E` is a partition of :math:`E` into a set of disjoint clusters :math:`C`. For example, the following diagram represents a clustering of elements :math:`\{0,1,2,3,4,5\}` into the three clusters "c1", "c2", and "c3"::

    ┌───────┐  ┌─────┐  ┌───┐
    │ 0   1 │  │  3  │  │   │
    │       │  │     │  │ 5 │
    │   2   │  │  4  │  │   │
    └───────┘  └─────┘  └───┘
       c1        c2      c3

We use the following data structures to represent clusterings:

Membership vector
    A membership vector is a pandas :py:class:`Series` indexed by the elements of :math:`E` and with values corresponding to cluster identifiers. That is, the memebership vector maps elements to clusters. Example::

        >>> pd.Series(["c1", "c1", "c1", "c2", "c2", "c3"], index=[0,1,2,3,4,5])
        0    c1
        1    c1
        2    c1
        3    c2
        4    c2
        5    c3
        dtype: object

Clusters dictionary
    A clusters dictionary is a Python :py:class:`dict` with keys corresponding to cluster identifiers and values being list of cluster elements. Example::

        {'c1': array([0, 1, 2]), 'c2': array([3, 4]), 'c3': array([5])}

Pairwise links list
    A pairwise links list is an array of pairwise links between elements of the clustering, where each element of a cluster is linked to every other element of the same cluster. Note that clusters are unnamed in pairwise links lists. Example::

        array([[0, 1],
               [0, 2],
               [1, 2],
               [3, 4]])

Graph
    A graph is an igraph :py:class:`Graph` object with vertices representing clustering elements and with edges between all elements belonging to the same cluster. Note that clusters are unnamed in graphs. Example::

        1───2       4
        │   │       │       6
        └─3─┘       5

--------
Contents
--------

* Check that objects satisfy representation definitions using::
    
    ismembership()
    isclusters()
    ispairs()
    isgraph()

* Transform between representations using the following functions::

    membership_to_clusters()
    membership_to_pairs()
    membership_to_graph()
    clusters_to_membership()
    clusters_to_pairs()
    clusters_to_graph()
    pairs_to_membership()
    pairs_to_clusters()
    pairs_to_graph()
    graph_to_membership()
    graph_to_clusters()
    graph_to_pairs()
"""

import numpy as np
import pandas as pd
from igraph import Graph


def isgraph(obj):
    r"""
    Check if given object is an iGraph :py:class:`Graph`.

    Graph:
        A graph is an igraph :py:class:`Graph` object with vertices representing clustering elements and with edges between all elements belonging to the same cluster. Note that clusters are unnamed in graphs. Example::

            1───2       4
            │   │       │       6
            └─3─┘       5

    Returns:
        bool: True if Graph, False otherwise.

    Examples:
        >>> import igraph
        >>> g = igraph.Graph()
        >>> isgraph(g)
        True
    """
    return isinstance(obj, Graph)


def ismembership(obj):
    r"""
    Check if given object is a membership vector.

    Membership vector:
        A membership vector is a pandas :py:class:`Series` indexed by the elements of :math:`E` and with values corresponding to cluster identifiers. That is, the memebership vector maps elements to clusters. Example::

            >>> pd.Series(["c1", "c1", "c1", "c2", "c2", "c3"], index=[0,1,2,3,4,5])
            0    c1
            1    c1
            2    c1
            3    c2
            4    c2
            5    c3
            dtype: object

    Returns:
        bool: True if membership vector, False otherwise.

    Examples:
        >>> import pandas as pd
        >>> obj = pd.Series(index=[1,2,3,4,5,6,7,8], data=[1,1,2,3,2,4,4,4])
        >>> ismembership(obj)
        True

        >>> ismembership([1,1,2,3,2,4,4,4])
        False
    """
    if isinstance(obj, pd.Series):
        return all(
            [
                obj.index.has_duplicates == False,
                obj.index.hasnans == False,
            ]
        )
    else:
        return False


def isclusters(obj):
    r"""
    Check if given object is a clusters dictionary.

    Clusters dictionary:
        A clusters dictionary is a Python :py:class:`dict` with keys corresponding to cluster identifiers and values being list of cluster elements. Example::

            {'c1': array([0, 1, 2]), 'c2': array([3, 4]), 'c3': array([5])}

    Returns:
        bool: True if clusters dictionary, False otherwise.

    Examples:
        >>> from numpy import array
        >>> obj = {'c1': array([0, 1, 2]), 'c2': array([3, 4]), 'c3': array([5])}
        >>> isclusters(obj)
        True

        Dictionary values should be numpy arrays:

        >>> obj = {'c1': [0, 1, 2], 'c2': [3, 4], 'c3': [5]}
        >>> isclusters(obj)
        False

        ⚠️ Warning: Clustering validity is not checked.

        >>> obj = {'c1': array([pd.NA]), 'c2': array([pd.NA])}
        >>> isclusters(obj)
        True

    Notes:
        * This function does not verify that clusters are non-overlapping with unique non-NaN elements.
    """
    return all(
        [
            isinstance(obj, dict),
            all(isinstance(value, np.ndarray) for value in obj.values()),
        ]
    )


def ispairs(obj):
    r"""
    Check if given object is a pairs list.

    A pairwise links list is an array of pairwise links between elements of the clustering, where each element of a cluster is linked to every other element of the same cluster. Note that clusters are unnamed in pairwise links lists. Example::

        array([[0, 1],
               [0, 2],
               [1, 2],
               [3, 4]])

    Returns:
        bool: True if a pairs list, False otherwise.

    Examples:
        >>> from numpy import array
        >>> obj = array([[0, 1], [0, 2], [1, 2], [3, 4]])
        >>> ispairs(obj)
        True

        >>> obj = [[0, 1], [0, 2], [1, 2], [3, 4]]
        >>> ispairs(obj)
        False
    """
    if not isinstance(obj, np.ndarray):
        return False
    shape = obj.shape
    if not len(shape) == 2:
        return False
    return shape[1] == 2


def membership_to_clusters(membership):
    r"""
    Transform membership vector into clusters dictionary.

    Args:
        membership (Series): Membership vector.

    Returns:
        Cluters dictionary.

    Examples:
        >>> membership = pd.Series(index=[1,2,3,4,5,6,7,8], data=[1,1,2,3,2,4,4,4])
        >>> membership_to_clusters(membership)
        {1: array([1, 2]), 2: array([3, 5]), 3: array([4]), 4: array([6, 7, 8])}
    """
    assert ismembership(membership)

    return {
        k: np.array(v) for k, v in membership.groupby(membership).groups.items()
    }


def membership_to_pairs(membership):
    r"""
    Transform membership vector into pairs list.

    Args:
        membership (Series): Membership vector.

    Returns:
        Pairs list.

    Examples:
        >>> membership = pd.Series(index=[1,2,3,4,5,6,7,8], data=[1,1,2,3,2,4,4,4])
        >>> membership_to_pairs(membership)
        array([[1, 2],
               [3, 5],
               [6, 7],
               [6, 8],
               [7, 8]])
    """
    assert ismembership(membership)

    clusters = membership_to_clusters(membership)
    return clusters_to_pairs(clusters)


def membership_to_graph(membership):
    r"""
    Transform membership vector into Graph.

    Args:
        membership (Series): Membership vector.

    Returns:
        Graph, with all elements converted to string.

    Note:
        All elements are converted to string before creating the graph.

    Examples:
        >>> membership = pd.Series(index=[1,2,3,4,5,6,7,8], data=[1,1,2,3,2,4,4,4])
        >>> graph = membership_to_graph(membership)

    """
    assert ismembership(membership)

    return pairs_to_graph(
        membership_to_pairs(membership), membership.index.values
    )


def clusters_to_pairs(clusters):
    r"""
    Transform clusters dictionary into pairs list.

    Args:
        clusters (dictionary): Dictionary mapping cluster identifiers to numpy array of cluster elements.

    Returns:
        Pairs list.

    Examples:
        >>> from numpy import array
        >>> clusters = {1: array([1, 2]), 2: array([3, 5]), 3: array([4]), 4: array([6, 7, 8])}
        >>> clusters_to_pairs(clusters)
        array([[1, 2],
               [3, 5],
               [6, 7],
               [6, 8],
               [7, 8]])
    """
    assert isclusters(clusters)

    def single_cluster_to_pairs(c):
        """
        References:
            - Carlos Gameiro (2021) Fast pairwise combinations in NumPy.
                Accessed online on November 1, 2022.
                https://carlostgameiro.medium.com/fast-pairwise-combinations-in-numpy-c29b977c33e2
        """
        I = np.stack(np.triu_indices(len(c), k=1), axis=-1)
        return c[I]

    return np.row_stack([single_cluster_to_pairs(c) for c in clusters.values()])


def clusters_to_membership(clusters):
    r"""
    Transform clusters dictionary into membership vector.

    Args:
        clusters (dictionary): Dictionary mapping cluster identifiers to numpy array of cluster elements.

    Returns:
        Membership vector.

    Examples:
        >>> from numpy import array
        >>> clusters = {1: array([1, 2]), 2: array([3, 5]), 3: array([4]), 4: array([6, 7, 8])}
        >>> clusters_to_membership(clusters)
        1    1
        2    1
        3    2
        5    2
        4    3
        6    4
        7    4
        8    4
        dtype: int64
    """
    assert isclusters(clusters)

    return pd.concat(
        [pd.Series(value, index=indices) for value, indices in clusters.items()]
    )


def clusters_to_graph(clusters):
    r"""
    Transform clusters dictionary into Graph.

    Args:
        clusters (dictionary): Dictionary mapping cluster identifiers to numpy array of cluster elements.

    Returns:
        Membership vector.

    Examples:
        >>> from numpy import array
        >>> clusters = {1: array([1, 2]), 2: array([3, 5]), 3: array([4]), 4: array([6, 7, 8])}
        >>> graph = clusters_to_graph(clusters)
    """
    assert isclusters(clusters)

    indices = np.concatenate(list(clusters.values()))

    return pairs_to_graph(clusters_to_pairs(clusters), indices)


def pairs_to_membership(pairs, indices):
    r"""Transform pairs list into membership vector.

    Args:
        pairs (ndarray): array of paired elements.
        indices (ndarray): flat array of all elements to consider (paired and non-paired), including singletons.

    Returns:
        Membership vector

    Examples:
        >>> from numpy import array
        >>> pairs = array([[1, 2], [3, 5], [6, 7], [6, 8], [7, 8]])
        >>> indices = array([1,2,3,4,5,6,7,8])
        >>> pairs_to_membership(pairs, indices)
        1    0
        2    0
        3    1
        4    2
        5    1
        6    3
        7    3
        8    3
        dtype: int64
    """
    assert ispairs(pairs)
    assert all(np.isin(pairs.flatten(), indices))

    return graph_to_membership(pairs_to_graph(pairs, indices))


def pairs_to_clusters(pairs, indices):
    r"""Transform pairs list into clusters dictionary.

    Args:
        pairs (ndarray): array of paired elements.
        indices (ndarray): flat array of all elements to consider (paired and non-paired), including singletons.
    """
    assert ispairs(pairs)
    assert all(np.isin(pairs.flatten(), indices))

    return membership_to_clusters(pairs_to_membership(pairs, indices))


def pairs_to_graph(pairs, indices):
    r"""
    Transform pairs list into Graph.

    Args:
        pairs (ndarray): array of paired elements.
        indices (ndarray): flat array of all elements to consider (paired and non-paired), including singletons.

    Returns:
        Graph corresponding to the pairs list with given indices as vertices. Note that all elements are converted to string before creating the graph.

    Note:
        All elements are converted to string before creating the graph.

    Examples:
        >>> from numpy import array
        >>> pairs = array([[1, 2], [3, 5], [6, 7], [6, 8], [7, 8]])
        >>> indices = array([1,2,3,4,5,6,7,8])
        >>> graph = pairs_to_graph(pairs, indices)
    """
    assert ispairs(pairs)
    assert all(np.isin(pairs.flatten(), indices))

    g = Graph()
    g.add_vertices(indices.astype(str))
    g.add_edges(pairs.astype(str))

    return g


def graph_to_membership(graph):
    r"""
    Transform Graph into membership vector.

    Args:
        graph (Graph): igraph Graph object.

    Returns:
        Membership vector

    Examples:
        >>> from numpy import array
        >>> membership = pd.Series(index=[1,2,3,4,5,6,7,8], data=[1,1,2,3,2,4,4,4])
        >>> graph = membership_to_graph(membership)
        >>> graph_to_membership(graph) # Note that cluster identifiers are arbitrary.
        1    0
        2    0
        3    1
        4    2
        5    1
        6    3
        7    3
        8    3
        dtype: int64
    """
    assert isgraph(graph)

    return pd.Series(
        index=graph.get_vertex_dataframe().name.values,
        data=graph.connected_components().membership,
    )


def graph_to_clusters(graph):
    r"""
    Transform Graph into clusters dictionary.

    Args:
        graph (Graph): igraph Graph object.

    Returns:
        Membership vector

    Examples:
        >>> from numpy import array
        >>> clusters = {1: array([1, 2]), 2: array([3, 5]), 3: array([4]), 4: array([6, 7, 8])}
        >>> graph = clusters_to_graph(clusters)
        >>> graph_to_clusters(graph) # doctest: +NORMALIZE_WHITESPACE
        {0: array(['1', '2'], dtype=object),
         1: array(['3', '5'], dtype=object),
         2: array(['4'], dtype=object),
         3: array(['6', '7', '8'], dtype=object)}
    """
    assert isgraph(graph)

    return membership_to_clusters(graph_to_membership(graph))


def graph_to_pairs(graph):
    r"""
    Transform Graph into pairs list.

    Args:
        graph (Graph): igraph Graph object.

    Returns:
        Membership vector

    Examples:
        >>> from numpy import array
        >>> pairs = array([[1, 2], [3, 5], [6, 7], [6, 8], [7, 8]])
        >>> indices = array([1,2,3,4,5,6,7,8])
        >>> graph = pairs_to_graph(pairs, indices)
        >>> graph_to_pairs(graph)
        array([['1', '2'],
               ['3', '5'],
               ['6', '7'],
               ['6', '8'],
               ['7', '8']], dtype='<U1')
    """
    assert isgraph(graph)

    names = graph.get_vertex_dataframe().name.values
    edges = graph.get_edgelist()

    return np.array([[names[e[0]], names[e[1]]] for e in edges])
