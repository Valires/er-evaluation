"""Datasets"""

from er_evaluation.datasets.patentsview import (
    load_pv_data,
    load_pv_disambiguations,
)

from er_evaluation.datasets.rldata import (
    load_rldata500,
    load_rldata500_disambiguations,
    load_rldata10000,
    load_rldata10000_disambiguations,
)

__all__ = [
    "load_pv_data",
    "load_pv_disambiguations",
    "load_rldata500",
    "load_rldata500_disambiguations",
    "load_rldata10000",
    "load_rldata10000_disambiguations",
]
