"""
================
Search Utilities
================

Search tools to assist with data labeling and information retrieval.

The ElasticSearch class provides a simple interface to query specific fields of an elasticsearch API, with meaningful defaults. It also provides the option to aggregate results by a given ID field, which is useful to retrieve entity clusters based on matching entity mentions.
"""

from er_evaluation.search.elasticsearch import ElasticSearch

__all__ = ["ElasticSearch"]
