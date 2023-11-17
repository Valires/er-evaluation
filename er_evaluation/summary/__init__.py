"""
Clustering Summary Statistics
=============================


The **summary** submodule provides disambiguation summary statistics that aim to describe properties of any given disambiguation result, as well as differences between successive or competing disambiguations. The goal of these statistics is not to assess performance but to provide key indicators that can be tracked to understand and monitor disambiguation results throughout the lifetime of the entity resolution system. They are simple and easily interpretable statistics that can help explain properties of the clustering. Additionally, these statistics act as quality assurance indicators that can be automatically monitored to identify potential bugs and errors.

The following statistics are provided:

* **Average Cluster Size**: The average cluster size is defined as the expected value of the size of a random cluster.
* **Matching Rate**: The matching rate is the proportion of elements belonging to clusters of size at least 2.
* **Cluster Entropy Curve**: The cluster entropy curve is a set of Hill numbers, which provide fine-grained cluster size distribution statistics. Hill numbers are based on the Rényi entropy of the cluster size distribution and provide interpretable statistics, such as the number of unique cluster sizes, the exponential Shannon entropy and the inverse prevalence of the most common cluster size.
* **Homonymy Rate**: The proportion of clusters where at least one individual name is shared with another cluster.
* **Name Variation Rate**: The proportion of clusters with name variation within them.

Example
-------

Get summary statistics for a given disambiguation:

.. code::

    import pandas as pd
    from er_evaluation.summary import summary_statistics

    membership = pd.Series(index=[1,2,3,4,5,6,7,8], data=[1,1,2,3,2,4,4,4])
    summary_statistics(membership)
    # {'number_of_clusters': 4, 
    #  'average_cluster_size': 2.0, 
    #  'matching_rate': 0.875, 
    #  'H0': 3, 
    #  'H1': 2.82842712474619, 
    #  'H2': 2.6666666666666665}

Additionally, you can provide a set of names associated with each cluster element in order to compute the homonymy rate and name variation rate:

.. code::

    names = pd.Series(index=[1,2,3,4,5,6,7,8], data=["a1", "a1", "b1", "a1", "b2", "c1", "c2", "c1"])
    summary_statistics(membership, names)
    # {'number_of_clusters': 4,
    #  'average_cluster_size': 2.0,
    #  'matching_rate': 0.875,
    #  'H0': 3,
    #  'H1': 2.82842712474619,
    #  'H2': 2.6666666666666665,
    #  'homonymy_rate': 0.5,
    #  'name_variation_rate': 0.5}
"""
from er_evaluation.summary._summary import (average_cluster_size,
                                            cluster_hill_number, cluster_sizes,
                                            cluster_sizes_distribution,
                                            homonymy_rate, matching_rate,
                                            name_variation_rate,
                                            number_of_clusters,
                                            number_of_links,
                                            summary_statistics)

__all__ = [
    "average_cluster_size",
    "cluster_hill_number",
    "cluster_sizes",
    "cluster_sizes_distribution",
    "homonymy_rate",
    "matching_rate",
    "name_variation_rate",
    "number_of_clusters",
    "number_of_links",
    "summary_statistics",
]
