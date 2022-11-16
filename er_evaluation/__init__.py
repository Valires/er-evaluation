"""
er_evaluation Module

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

from .data_structures import (
    ismembership,
    isclusters,
    ispairs,
    isgraph,
    membership_to_clusters,
    membership_to_pairs,
    membership_to_graph,
    clusters_to_membership,
    clusters_to_pairs,
    clusters_to_graph,
    pairs_to_membership,
    pairs_to_clusters,
    pairs_to_graph,
    graph_to_membership,
    graph_to_clusters,
    graph_to_pairs,
)

from .error_analysis import (
    expected_extra_links,
    expected_missing_links,
    count_extra_links,
    count_missing_links,
    splitting_entropy,
)

from .estimators import (
    pairwise_precision_design_estimate,
    pairwise_recall_design_estimate,
)

from .metrics import (
    pairwise_precision,
    pairwise_recall,
)

from .plots import (
    compare_plots,
    plot_cluster_sizes_distribution,
    plot_entropy_curve,
)

from .summary import (
    number_of_clusters,
    number_of_links,
    cluster_hill_number,
    cluster_sizes,
    cluster_sizes_distribution,
    homonimy_rate,
    name_variation_rate,
)

__all__ = [
    # data_structures
    "ismembership",
    "isclusters",
    "ispairs",
    "isgraph",
    "membership_to_clusters",
    "membership_to_pairs",
    "membership_to_graph",
    "clusters_to_membership",
    "clusters_to_pairs",
    "clusters_to_graph",
    "pairs_to_membership",
    "pairs_to_clusters",
    "pairs_to_graph",
    "graph_to_membership",
    "graph_to_clusters",
    "graph_to_pairs",
    # error_analysis
    "expected_extra_links",
    "expected_missing_links",
    "count_extra_links",
    "count_missing_links",
    "splitting_entropy",
    # estimators
    "pairwise_precision_design_estimate",
    "pairwise_recall_design_estimate",
    # metrics
    "pairwise_precision",
    "pairwise_recall",
    # plots
    "compare_plots",
    "plot_cluster_sizes_distribution",
    "plot_entropy_curve",
    # summary
    "number_of_clusters",
    "number_of_links",
    "cluster_hill_number",
    "cluster_sizes",
    "cluster_sizes_distribution",
    "homonimy_rate",
    "name_variation_rate",
]
