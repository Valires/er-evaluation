r"""
==========================
Clustering Data Structures
==========================

The **data_structures** module contains utilities defining clustering data structures (graph, membership vector, clusters dictionary, and pairwise links list) and allowing transformation between them.

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
    A membership vector is a pandas :py:class:`Series` indexed by the elements of :math:`E` and with values corresponding to cluster identifiers. That is, the membership vector maps elements to clusters. Example::

        >>> import pandas as pd
        >>> pd.Series(["c1", "c1", "c1", "c2", "c2", "c3"], index=[0,1,2,3,4,5])
        0    c1
        1    c1
        2    c1
        3    c2
        4    c2
        5    c3
        dtype: object
    
    Note that using integer indices and values for membership vectors will lead to significantly faster computation. See :py:meth:`er_evaluation.data_structures.compress_memberships`.

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
"""
from er_evaluation.data_structures._data_structures import (
    MembershipVector,
    clusters_to_graph,
    clusters_to_membership,
    clusters_to_pairs,
    compress_memberships,
    graph_to_clusters,
    graph_to_membership,
    graph_to_pairs,
    isclusters,
    isgraph,
    ismembership,
    ispairs,
    membership_to_clusters,
    membership_to_graph,
    membership_to_pairs,
    pairs_to_clusters,
    pairs_to_graph,
    pairs_to_membership,
)

__all__ = [
    "compress_memberships",
    "clusters_to_graph",
    "clusters_to_membership",
    "clusters_to_pairs",
    "graph_to_clusters",
    "graph_to_membership",
    "graph_to_pairs",
    "isclusters",
    "isgraph",
    "ismembership",
    "ispairs",
    "membership_to_clusters",
    "membership_to_graph",
    "membership_to_pairs",
    "pairs_to_clusters",
    "pairs_to_graph",
    "pairs_to_membership",
    "MembershipVector",
]
