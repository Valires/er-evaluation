Data Labeling Guide
===================

Data labeling is a critical step in the development and evaluation of machine learning algorithms, particularly in entity resolution and disambiguation tasks. A high-quality labeled dataset is essential for training and evaluating models, as well as providing insights into features and patterns that can improve the performance of the disambiguation algorithm.

This guide aims to provide a generic overview of the data labeling process, focusing on creating benchmark datasets that can be used with **ER-Evaluation** to evaluate the performance of disambiguation algorithms.

Before You Get Started
----------------------

Before getting started with data labeling, make sure you have a clear operational entity definition. For example, if you are disambiguating company names, then what is your definition of a company? Does it include subsidiaries? You need a clear definition for what constitutes an entity, how to handle variations in entity names, and how to handle cases where there is ambiguity.

Next, make sure to identify a time period (and/or regional period) that you will focus on. Your data labeling should be done with respect to a well-defined scope.

Finally, our data labeling methodology has **two dependencies**:

1. You need predicted disambiguation to be used as a starting point. This disambiguation does not have to be perfect, but it will facilitate disambiguation by providing candidate matches to annotators.
2. You need a search tool to help identify missing elements from disambiguated clusters. The search tool should be able to search entity mentions and aggregate results by disambiguated cluster ID. This can easily be set up using ElasticSearch. Our experimental `search 'er_evaluation.search`_ module provides utilities to this end.

Data Labeling Methodology
-------------------------

Benchmark datasets used with **ER-Evaluation** need to satisfy two criteria:

1. Entity clusters should be **complete**: Ensure that no relevant entity mention is omitted within the targeted time period. This means all mentions of a specific entity are included in the dataset.
2. Associated sampling weights: Assign weights to the entity clusters, ensuring the representativeness of the data. This can be done using random sampling or probabilities proportional to cluster sizes.

To achieve these two goals, we recommend the methodology described below.

1. Randomly sample an entity mention.
2. Identify all predicted clusters that contain matches to the sampled entity mention. This can be done using the pre-requisite search tool.
3. Review the contents of the identified predicted clusters to remove all non-matching elements.
4. The result of step (3) is the "true" cluster for the sampled entity mention. Repeat step (1) - (3) until the desired sample size is achieved.

.. note::

    You can use probability sampling in step (1). In practice, we find that sampling entity mentions uniformly at random (i.e., sampling clusters with probability proportional to their size) is cost-efficient for most applications.

Quality Control
^^^^^^^^^^^^^^^

To validate the quality of hand-disambiguations, you can use the following quality control measures:

- Verify that the sampled entity mention is part of the cluster resulting from step (3).
- Confirm that all mention IDs in the hand-disambiguated cluster are valid.
- Review the list of unique entity names in the disambiguated clusters. Any name that obviously do not match the sampled mention name should be identified and flagged as an error.

Additional Considerations
-------------------------

Define Clear Guidelines and Provide Training
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Before starting the data labeling process, it is important to define clear guidelines for how to label entities. This should include definitions for what constitutes an entity, how to handle variations in entity names, and how to handle cases where there is ambiguity. These guidelines should be clear and unambiguous, and should be followed consistently throughout the labeling process. Annotators should be provided with training to ensure that they have the necessary knowledge and skills make decisions with confidence when facing ambiguous cases. Individual assistance should be readily available when needed to unblock annotators.

Use Multiple Annotators
^^^^^^^^^^^^^^^^^^^^^^^

To improve accuracy in data labeling, using multiple annotators can help reduce individual bias and increase consistency. Optimizing efficiency and maintaining quality requires considering how best to utilize them. For example, annotators can label the same data independently or different subsets of the data. Collaboration and communication among annotators can also improve accuracy, such as by sharing insights and discussing ambiguous cases. Spreading the workload can prevent fatigue and maintain quality over time.

Use Validation Data and Quality Control
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Validation data is a subset of the dataset that has already been labeled and is considered to be highly reliable. Using validation data can help identify errors and inconsistencies in the labeling process. It can also help identify cases where there is disagreement among annotators, and improve the overall accuracy of the labeling process. You can also validate the quality of the labeled data through quality control measures, such as reviewing labeled data for obvious errors.

Monitor Inter-Annotator Agreement
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Inter-annotator agreement is a measure of the consistency of the annotations provided by multiple annotators. Monitoring inter-annotator agreement can help identify cases where there is disagreement among annotators, and improve the overall accuracy of the labeling process. It can also help identify cases where the guidelines may need to be updated or clarified.
