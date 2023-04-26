---
title: 'ER-Evaluation: An End-to-End Evaluation Framework for Entity Resolution Systems'
tags:
  - Python
  - Entity Resolution
  - Evaluation
authors:
  - name: Olivier Binette
    orcid: 0000-0001-6009-5206
    equal-contrib: true
    corresponding: true
    affiliation: "1" # (Multiple affiliations must be quoted)
affiliations:
 - name: Duke University, USA
   index: 1
date: April 26, 2023
bibliography: paper.bib
---

# Summary

Entity resolution (ER), also refered to as record linkage and deduplication, is the process of identifying and matching distinct representations of real-world entities across diverse data sources. It is used for data management, cleaning and integration, with numerous important applications including assessing the accuracy of the decennial census, detecting fraud, linking patient data in health care, and extracting relationships in structured and unstructured data. As ER techniques continue to evolve and improve, it is essential to have an efficient and comprehensive evaluation framework to measure their performance and compare different approaches. However, despite the growth of ER research, there is still a need for a unified evaluation framework that can address the challenges associated with the evaluation of ER systems, including accounting for sampling biases and managing class imbalance. Without the use of principled evaluation methodology, the naive use of clustering metrics and toy benchmark datasets has been shown to lead to over-optimistic results, performance rank reversals, and poor system design.

ER-Evaluation is a Python 3.7+ package addressing these challenges by implementing all components of a principled evaluation framework for ER systems. It incorporates principled statistical estimators for key performance metrics and summary statistics, error analysis tools, and data labeling tools, and data visualizations. Adopting an entity-centric approach, ER-Evaluation uses disambiguated entity clusters as the foundation for analysis. The package is written in Python with a simple architecture to ensure straightforward portability to other languages and frameworks when needed.


# Statement of need

Entity resolution is a microclustering problem: a particular type of clustering where clusters tend to be small and numerous (up to millions or billions of clusters). As such, researchers typically evaluate the performance of entity resolution systems using performance metrics (precision, recall, b-cubed metrics) on relatively small benchmark datasets. However, this process has been shown to lead to highly biased and over-optimistic performance assessments in ER, leading to preformance rank reversals and poor system design. To address this issue, new entity-centric methodology has been proposed in [XX] to obtain accurate performance metric estimates based on small and possibly biased benchmark datasets. The ER-Evaluation package implements this methodology as well as numerous extensions for a comprehensive, end-to-end evaluation framework. It aims to simplify the comparison of various ER techniques, evaluate their accuracy, and ultimately expedite the development and adoption of high-performing ER systems. Integrating essential components like data preprocessing, error analysis, performance estimation, and visualization functions, ER-Evaluation presents a user-friendly, modular, and expandable interface for researchers and practitioners.

The software is used by PatentsView.org for the evaluation of patent inventor name disambiguation. The original methodology has been published in [XX] and extended methodology is currently being developed in an upcoming article titled "An End-to-End Evaluation Framework for Entity Resolution Systems With Application to Inventor Name Disambiguation".

# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Acknowledgements

We acknowledge financial support from the National Sciences and Engineering Research Council of Canada and from the Fonds de Recherche du Québec - Nature et Technologies.

# References
