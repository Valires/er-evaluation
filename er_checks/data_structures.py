"""
Clustering data structures: graph, membership vector, clusters list,
pairwise links.
"""

import numpy as np
import pandas as pd
from igraph import Graph


def isgraph(obj):
    return isinstance(obj, Graph)


def ismembership(obj):
    return all(
        [
            isinstance(obj, pd.Series),
            obj.index.has_duplicates == False,
            obj.index.hasnans == False,
        ]
    )


def isclusters(obj):
    return all(
        [
            isinstance(obj, dict),
            all(isinstance(value, np.array) for value in obj.values()),
            ismembership(clusters_to_membership(obj)),  # Check for nans and duplicates
        ]
    )


def ispairs(obj):
    if not isinstance(obj, np.array):
        return False
    shape = obj.shape()
    return all([len(shape) == 2, shape[1] == 2])


def membership_to_clusters(membership):
    assert ismembership(membership)

    return membership.groupby(membership).indices


def membership_to_pairs(membership):
    assert ismembership(membership)

    clusters = membership_to_clusters(membership)
    return clusters_to_pairs(clusters)


def membership_to_graph(membership):
    assert ismembership(membership)

    return pairs_to_graph(membership_to_pairs(membership))


def clusters_to_pairs(clusters):
    assert isclusters(clusters)

    def single_cluster_to_pairs(c):
        """
        References:
            - Carlos Gameiro (2021) Fast pairwise combinations in NumPy. Accessed online on November 1, 2022. https://carlostgameiro.medium.com/fast-pairwise-combinations-in-numpy-c29b977c33e2
        """
        I = np.stack(np.triu_indices(len(c), k=1), axis=-1)
        return c[I]

    return np.row_stack([single_cluster_to_pairs(c) for c in clusters.values()])


def clusters_to_membership(clusters):
    assert isclusters(clusters)

    return pd.concat(
        [pd.Series(value, index=indices) for value, indices in clusters.items()]
    )


def clusters_to_graph(clusters):
    assert isclusters(clusters)

    return pairs_to_graph(clusters_to_pairs(clusters))


def pairs_to_membership(pairs, indices):
    assert ispairs(pairs)
    assert all(np.isin(pairs.flatten(), indices))

    return graph_to_membership(pairs_to_graph(pairs, indices))


def pairs_to_clusters(pairs, indices):
    assert ispairs(pairs)
    assert all(np.isin(pairs.flatten(), indices))

    return membership_to_clusters(pairs_to_membership(pairs, indices))


def pairs_to_graph(pairs, indices):
    assert ispairs(pairs)
    assert all(np.isin(pairs.flatten(), indices))

    g = Graph()
    g.add_vertices(indices)
    g.add_edges(pairs)

    return g


def graph_to_membership(graph):
    assert isgraph(graph)

    return graph.connected_components().membership


def graph_to_clusters(graph):
    assert isgraph(graph)

    return membership_to_clusters(graph_to_membership(graph))


def graph_to_pairs(graph):
    assert isgraph(graph)

    return np.array(graph.get_edgelist())
