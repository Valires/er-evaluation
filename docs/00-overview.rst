------------
Introduction
------------

.. _introduction:

ER-Evaluation is a Python 3.7+ package designed for evaluating the performance of entity resolution (ER) systems. The package implements all components of a comprehensive evaluation process, from understanding characteristics of your ER system and analyzing errors, to estimating and visualizing key performance metrics.

Central to the package are **principled statistical estimators** for performance metrics and summary statistics, taking into account sampling processes and biases. Without those, performance metrics are over-optimistic and not representative of real-world performance, leading to poor system design and poor performance. See `our blog post <https://www.valires.com/post/common-pitfalls-to-avoid-when-estimating-er-performance-metrics>`_ on the topic for more information.

Our evaluation tools employ an **entity-centric** approach, using disambiguated entity clusters as the starting point of analysis.

To use the package, you need:

1. **Prediction(s)**: Predicted entity clusters for a set of records or entity mentions, usually the main output of an ER system.
2. **Reference/benchmark data**: A trusted benchmark dataset or reference disambiguation for "ground truth" data, typically containing 200 to 400 disambiguated entities.

.. admonition:: Example

    Consider a customer relationship management (CRM) system with individual customer records, including duplicate entries. An entity resolution system consolidates customer records, producing **predicted clusters**. To assess accuracy, a set of verified customer record clusters (a **reference disambiguation**) is needed.

.. note::

    - Consult our `data labeling guide <06-data-labeling.html>`_ for strategies to create reference disambiguations.
    - Without reference data, only summary statistics can be monitored, and accuracy cannot be determined.
    - Ensure reference data is representative of the entire population or apply sampling weights for accurate performance metrics.


With predictions and reference data available, you can install the package and begin evaluation.

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

The `Advanced Topics <05-advanced_topics>`_ page discusses computational speedups, inverse probability weighting using estimated propensity scores, error modeling, sensitivity analyses, and the structure of the statistical estimation framework on which the package is based.

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