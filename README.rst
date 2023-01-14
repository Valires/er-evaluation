
.. image:: https://github.com/OlivierBinette/er-evaluation/actions/workflows/python-package.yaml/badge.svg
        :target: https://github.com/OlivierBinette/er-evaluation/actions/workflows/python-package.yaml

.. image:: https://badge.fury.io/py/er-evaluation.svg
        :target: https://badge.fury.io/py/er-evaluation

.. image:: https://readthedocs.org/projects/er-evaluation/badge/?version=latest
        :target: https://er-evaluation.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status


🔍 ER-Evaluation: An End-to-End Evaluation Framework for Entity Resolution Systems
==================================================================================

**ER-Evaluation** is a Python package for the evaluation of entity resolution (ER) systems. It provides data structure definitions, summary statistics, visualizations, error analysis tools, and statistically principled performance estimators.

Installation
------------

Install the released version from PyPI using:

.. code:: bash

    pip install er-evaluation


Documentation
-------------

Please refer to the documentation website `er-evaluation.readthedocs.io <https://er-evaluation.readthedocs.io/en/latest>`_.

Development Philosophy
----------------------

**ER-Evaluation** is designed to be a unified source of evaluation tools for entity resolution systems, adhering to the Unix philosophy of simplicity, modularity, and composability. The package contains Python functions that take standard data structures such as pandas Series and DataFrames as input, making it easy to integrate into existing workflows. By importing the necessary functions and calling them on your data, you can easily use ER-Evaluation to evaluate your entity resolution system without worrying about custom data structures or complex architectures.

Publications
------------

- `Binette, Olivier, Sokhna A York, Emma Hickerson, Youngsoo Baek, Sarvo Madhavan, Christina Jones. (2022). Estimating the Performance of Entity Resolution Algorithms: Lessons Learned Through PatentsView.org. arXiv e-prints: arxiv:2210.01230 <https://arxiv.org/abs/2210.01230>`_

- Upcoming: "A Statistical Evaluation Framework for Black-Box Entity Resolution Systems With Application to Inventor Name Disambiguation"

Acknowledgements
----------------

**ER-Evaluation** is an extension of the `PatentsView/PatentsView-Evaluation <https://github.com/PatentsView/PatentsView-Evaluation/>`_ project sponsored by the American Institutes for Research and the U.S. Patents and Trademarks Office. We aim to provide a unified source of evaluation tools for entity resolution systems which are maintained as an open source academic project.

Funding
^^^^^^^

This project was made possible through support from the Natural Sciences and Engineering Research Council of Canada, Fonds de Recherche du Québec - Nature et Technologies, Duke University, the American Institutes for Research, and the U.S. Patents and Trademarks Office.

Citation
--------

Please acknowledge the above publications as well as the ER-Evaluation Python package:

- Binette, Olivier. (2022). ER-Evaluation: An End-to-End Evaluation Framework for Entity Resolution Systems. Available online at https://github.com/OlivierBinette/ER-Evaluation

License
-------

* `GNU Affero General Public License v3 <https://www.gnu.org/licenses/agpl-3.0.en.html>`_