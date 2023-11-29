"""
====================================
Example datasets and disambiguations
====================================

The **datasets** module contains toy datasets used to test and demonstrate the functionality of the ER-Evaluation package.

For example, the :py:meth:`load_pv_disambiguations` function returns the tuple `(predictions, reference)`, where `predictions` is a dictionary containing PatentsView's disambiguation history (indexed by pandas Datetime objects), and where `reference` is *Binette's 2022 inventors benchmark* that contains 401 disambiguated inventors.

The :py:meth:`load_pv_data` function returns a dataframe containing features for a small set of inventor mention.

The :py:meth:`load_rldata10000_disambiguations` and :py:meth:`load_rldata10000` return ground truth disambiguation, toy predicted disambiguations, and the full RLdata1000 dataframe.
"""

from er_evaluation.datasets.patentsview import (load_pv_data,
                                                load_pv_disambiguations)
from er_evaluation.datasets.rldata import (load_rldata500,
                                           load_rldata500_disambiguations,
                                           load_rldata10000,
                                           load_rldata10000_disambiguations)

__all__ = [
    "load_pv_data",
    "load_pv_disambiguations",
    "load_rldata500",
    "load_rldata500_disambiguations",
    "load_rldata10000",
    "load_rldata10000_disambiguations",
]
