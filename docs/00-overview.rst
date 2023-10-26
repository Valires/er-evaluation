------------
Introduction
------------

.. _introduction:

ER-Evaluation is a Python 3.7+ package designed for evaluating the performance of entity resolution (ER) systems. The package implements all components of a comprehensive evaluation process, from understanding characteristics of your ER system and analyzing errors, to estimating and visualizing key performance metrics.

Central to the package are **principled statistical estimators** for performance metrics and summary statistics, taking into account sampling processes and biases. Without those, performance metrics are generally over-optimistic and not representative of real-world performance, which can lead to performance rank reversals and poor system design. See `our blog post <https://www.valires.com/post/common-pitfalls-to-avoid-when-estimating-er-performance-metrics>`_ on the topic for more information.

Our evaluation tools employ an **entity-centric approach,** using a sample of fully-resolved entities as the starting point of analysis. To use the package, you need:

- **Prediction(s):** Predicted clusters for a set of records or entity mentions, usually the main output of an ER system.

  - You can have a single output or you can compare multiple clustering results.
- **Reference/benchmark data:** A sample of fully-resolved entities, usually from a trusted benchmark dataset or from data labeling.

  - Each entity/cluster in the reference dataset should be *complete*, i.e. it should be fully resolved and it should not exclude any record or mention.
  - In our experience, benchmark datasets including 200 to 400 resolved entities are generally sufficient.

.. admonition:: Example

    Consider a customer relationship management (CRM) system with individual customer records, including duplicate entries. An entity resolution system consolidates customer records, producing **predicted clusters**. Each predicted cluster is meant to represent a single customer. To assess accuracy, a set of verified customer record clusters (a **reference disambiguation**) is needed.

.. note::

    - Consult our `data labeling guide <06-data-labeling.html>`_ for strategies to create reference disambiguations.
    - Without reference data, only summary statistics can be monitored, and accuracy cannot be determined.
    - Ensure reference data is representative of the entire population or apply sampling weights for accurate performance metrics.

With predictions and reference data available, you can install the package and begin evaluation.

---------------------------
Terminology and Definitions
---------------------------

Throughout this user guide and the documentation of the package, we use the following terminology:

- An **entity** is a real-world object, e.g. a person, a company, or a product. 
- A **record** is a set of attributes describing an entity, e.g. a customer record in a CRM system. The term **record** is used interchangeably with the term **entity mention**.
- A **database** or **file** is a collection of records/mentions.
- **Entity resolution,** also called **disambiguation,** **deduplication,** or **record linkage** aims to create a clustering of records/mentions according to the entity which they represent.
- The output of an entity resolution system is a **predicted clustering,** i.e. an attempt at correctly clustering records/mentions according to the entity to which they refer. There may be errors in the predicted clustering, e.g. records/mentions may be incorrectly clustered together or split into multiple clusters.
- A **reference** dataset, or a set of **ground truth** clusters, is a clustering of mentions/records that is assumed to be correct.

For more information on entity resolution, we refer the reader to [Binette & Steorts (2022)](https://www.science.org/doi/10.1126/sciadv.abi8021) and [Christophides et al. (2019)](https://arxiv.org/abs/1905.06397).

We recommend `Splink <https://github.com/moj-analytical-services/splink>`_ as a state-of-the-art large-scale entity resolution software. The splink team provides a large list of `tutorials <https://moj-analytical-services.github.io/splink/demos/tutorials/00_Tutorial_Introduction.html>`_ and `training materials <https://moj-analytical-services.github.io/splink/topic_guides/topic_guides_index.html>`_ on their website. The book `"Hands-On Entity Resolution" <https://www.oreilly.com/library/view/hands-on-entity-resolution/9781098148478/>`_ provides an introduction to entity resolution with Splink.

------------
Installation
------------

.. _installation:

You can install the ER-Evaluation package `from PyPI <https://pypi.org/project/ER-Evaluation/>`_ using:

.. code::

    pip install er-evaluation


----------
Next Steps
----------

.. _next-steps:

With the package installed, the next step will be to `prepare your data <01-dataprep.html>`_ for evaluation.

If you would rather skim the guide for an overview of features, we recommend jumping ahead to pages describing:

1. `Summary Statistic monitoring <02-summary_statistics.html>`_
2. `Perfomance estimation <03-estimating_performance.html>`_
3. `Error Analysis <04-error_analysis.html>`_
4. `Visualization Functions <visualizations.html>`_

The `Advanced Topics <05-advanced_topics.html>`_ page briefly discusses computational speedups, error modeling, and sensitivity analyses.

------------
Other Topics
------------

.. _other-topics:

Portability
-----------

.. _portability:

**ER-Evaluation** is written in pure Python with a simple functional architecture and standard relational algebra operators for data manipulation. Functions can easily be ported to languages and frameworks such as SQL and PySpark.

Contributing and Support
------------------------

.. _contributing:

Users can submit issues or feature requests on the `GitHub repository <https://github.com/Valires/er-evaluation>`_. We encourage contributions through pull requests or sharing ideas and suggestions with the community.