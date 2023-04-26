---
title: 'ER-Evaluation: End-to-End Evaluation of Entity Resolution Systems'
tags:
  - Python
  - Entity Resolution
  - Evaluation
authors:
  - name: Olivier Binette
    orcid: 0000-0001-6009-5206
    corresponding: true
    affiliation: "1"
affiliations:
 - name: Duke University, USA
   index: 1
date: April 26, 2023
bibliography: paper.bib
---

# Summary

Entity resolution (ER), also referred to as record linkage and deduplication, is the process of identifying and matching distinct representations of real-world entities across diverse data sources. It plays a crucial role in data management, cleaning, and integration, with applications such as assessing the accuracy of the decennial census, detecting fraud, linking patient data in healthcare, and extracting relationships in structured and unstructured data [@christen2012; christophides2019; @papadakis2021; @binette2022a].

As ER techniques continue to evolve and improve, it is essential to have an efficient and comprehensive evaluation framework to measure their performance and compare different approaches. Despite the growth of ER research, there remains a need for a unified evaluation framework that can address challenges associated with ER system evaluation, including accounting for sampling biases and managing class imbalances. Otherwise, using naive clustering metrics and toy benchmark datasets without a principled evaluation methodology leads to over-optimistic results, performance rank reversals, and poor system design [@wang2022; @binette2022b].

ER-Evaluation is a Python 3.7+ package designed to address these challenges by implementing all components of a principled evaluation framework for ER systems. It incorporates principled statistical estimators for key performance metrics and summary statistics, error analysis tools, data labeling tools, and data visualizations. The package is written in Python with a simple architecture, ensuring straightforward portability to other languages and frameworks when necessary.

Additionally, ER-Evaluation adopts a novel entity-centric approach that uses disambiguated entity clusters as the foundation for analysis. This contrasts with traditional evaluation methods based on labeling record pairs [@marchant2017]. The entity-centric approach streamlines the utilization of existing benchmark datasets and the labeling of new datasets without necessitating complex sampling schemes. Furthermore, it enables the reuse of benchmark datasets at all stages of the evaluation process, including for cluster-level error analysis.

# Statement of need

Entity resolution is a clustering problem characterized by small and numerous clusters (up to millions or billions of clusters). Researchers commonly evaluate the performance of entity resolution systems by computing performance metrics (precision, recall, cluster metrics) on relatively small benchmark datasets. However, this process has been shown to yield highly biased and over-optimistic performance assessments in ER, leading to performance rank reversals and poor system design.

To address this issue, a new entity-centric methodology has been proposed in @binette2022b for obtaining accurate performance metric estimates based on small and potentially biased benchmark datasets. The ER-Evaluation package implements this methodology and numerous extensions to create a comprehensive, end-to-end evaluation framework. It aims to streamline the comparison of diverse ER techniques, assess their accuracy, and ultimately accelerate the development and adoption of high-performing ER systems. By integrating essential components such as data preprocessing, error analysis, performance estimation, and visualization functions, ER-Evaluation offers a user-friendly, modular, and extensible interface for researchers and practitioners.

The software is currently being used by PatentsView.org for the evaluation of patent inventor name disambiguation [@binette2022c]. The original methodology has been published in [@binette2022b], and extended methodology is under development in an upcoming article titled "An End-to-End Evaluation Framework for Entity Resolution Systems With Application to Inventor Name Disambiguation."

# Acknowledgements

We acknowledge financial support from the National Sciences and Engineering Research Council of Canada and from the Fonds de Recherche du Québec - Nature et Technologies.

# References
