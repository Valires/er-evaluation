"""
Entity Resolution Clustering Data Structures

This module contains utilities defining clustering data structures (graph, membership vector, clusters dictionary, and pairwise links list) and allowing transformation between them.

-----------
Definitions
-----------

A clustering of a set of elements $E$ is a partition of $E$ into a set of disjoint clusters $C$. For example, the following diagram represents a clustering of elements $\{0,1,2,3,4,5\}$ into the three clusters "c1", "c2", and "c3"::

    ┌───────┐  ┌─────┐  ┌───┐
    │ 0   1 │  │  3  │  │   │
    │       │  │     │  │ 5 │
    │   2   │  │  4  │  │   │
    └───────┘  └─────┘  └───┘
       c1        c2      c3

We use the following data structures to represent clusterings:

Membership vector
    A membership vector is a pandas :py:class:`Series` indexed by the elements of $E$ and with values corresponding to cluster identifiers. That is, the memebership vector maps elements to clusters. Example::

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


Copyright (C) 2022  Olivier Binette

This file is part of the ER-Evaluation Python package (er-evaluation).

er-evaluation is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import numpy as np
import pandas as pd
from igraph import Graph


def isgraph(obj):
    """
    Check if given object is an iGraph :py:class:`Graph`.

    A graph is an igraph :py:class:`Graph` object with vertices representing clustering elements and with edges between all elements belonging to the same cluster. Note that clusters are unnamed in graphs. Example::

        1───2       4
        │   │       │       6
        └─3─┘       5

    Returns:
        bool: True if Graph, False otherwise.
    """
    return isinstance(obj, Graph)


def ismembership(obj):
    """
    Check if given object is a membership vector.

    A membership vector is a pandas :py:class:`Series` indexed by the elements of $E$ and with values corresponding to cluster identifiers. That is, the memebership vector maps elements to clusters. Example::

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
    """
    return all(
        [
            isinstance(obj, pd.Series),
            obj.index.has_duplicates == False,
            obj.index.hasnans == False,
        ]
    )


def isclusters(obj):
    """
    Check if given object is a clusters dictionary.

    A clusters dictionary is a Python :py:class:`dict` with keys corresponding to cluster identifiers and values being list of cluster elements. Example::

        {'c1': array([0, 1, 2]), 'c2': array([3, 4]), 'c3': array([5])}

    Returns:
        bool: True if clusters dictionary, False otherwise.

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
    """
    Check if given object is a pairs list.

    A pairwise links list is an array of pairwise links between elements of the clustering, where each element of a cluster is linked to every other element of the same cluster. Note that clusters are unnamed in pairwise links lists. Example::

        array([[0, 1],
               [0, 2],
               [1, 2],
               [3, 4]])

    Returns:
        bool: True if membership vector, False otherwise.
    """
    if not isinstance(obj, np.ndarray):
        return False
    shape = obj.shape
    return all([len(shape) == 2, shape[1] == 2])


def membership_to_clusters(membership):
    """
    Transform membership vector into clusters dictionary.
    """
    assert ismembership(membership)

    return membership.groupby(membership).indices


def membership_to_pairs(membership):
    """Transform membership vector into pairs list."""
    assert ismembership(membership)

    clusters = membership_to_clusters(membership)
    return clusters_to_pairs(clusters)


def membership_to_graph(membership):
    """Transform membership vector into Graph."""
    assert ismembership(membership)

    return pairs_to_graph(membership_to_pairs(membership), membership.index)


def clusters_to_pairs(clusters):
    """Transform clusters dictionary into pairs list."""
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
    """Transform clusters dictionary into membership vector."""
    assert isclusters(clusters)

    return pd.concat(
        [pd.Series(value, index=indices) for value, indices in clusters.items()]
    )


def clusters_to_graph(clusters):
    """Transform clusters dictionary into Graph."""
    assert isclusters(clusters)

    return pairs_to_graph(clusters_to_pairs(clusters))


def pairs_to_membership(pairs, indices):
    """Transform pairs list into membership vector.

    Args:
        pairs (ndarray): array of paired elements.
        indices (ndarray): flat array of all elements to consider (paired and non-paired), including singletons.
    """
    assert ispairs(pairs)
    assert all(np.isin(pairs.flatten(), indices))

    return graph_to_membership(pairs_to_graph(pairs, indices))


def pairs_to_clusters(pairs, indices):
    """Transform pairs list into clusters dictionary.

    Args:
        pairs (ndarray): array of paired elements.
        indices (ndarray): flat array of all elements to consider (paired and non-paired), including singletons.
    """
    assert ispairs(pairs)
    assert all(np.isin(pairs.flatten(), indices))

    return membership_to_clusters(pairs_to_membership(pairs, indices))


def pairs_to_graph(pairs, indices):
    """
    Transform pairs list into Graph.

    Args:
        pairs (ndarray): array of paired elements.
        indices (ndarray): flat array of all elements to consider (paired and non-paired), including singletons.
    """
    assert ispairs(pairs)
    assert all(np.isin(pairs.flatten(), indices))

    g = Graph()
    g.add_vertices(indices)
    g.add_edges(pairs)

    return g


def graph_to_membership(graph):
    """Transform Graph into membership vector."""
    assert isgraph(graph)

    return graph.connected_components().membership


def graph_to_clusters(graph):
    """Transform Graph into clusters dictionary."""
    assert isgraph(graph)

    return membership_to_clusters(graph_to_membership(graph))


def graph_to_pairs(graph):
    """Transform Graph into pairs list."""
    assert isgraph(graph)

    return np.array(graph.get_edgelist())
