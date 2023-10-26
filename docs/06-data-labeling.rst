Data Labeling Guide
===================

Data labeling is a critical step in the development and evaluation of machine learning algorithms, particularly in entity resolution and disambiguation tasks. A high-quality labeled dataset is essential for training and evaluating models, as well as providing insights into features and patterns that can improve the performance of the disambiguation algorithm.

This guide aims to provide a generic overview of the data labeling process, focusing on creating benchmark datasets that can be used with **ER-Evaluation** to evaluate the performance of disambiguation algorithms.

Before You Get Started
----------------------

Before getting started with data labeling, make sure you have a clear operational entity definition. For example, if you are disambiguating company names, then what is your definition of a company? Does it include subsidiaries? You need a clear definition for what constitutes an entity, how to handle variations in entity names, and how to handle cases where there is ambiguity.

Next, make sure to identify a time period (and/or regional period) that you will focus on. Your data labeling should be done with respect to a well-defined scope.

Data Labeling Methodology
-------------------------

Benchmark datasets used with **ER-Evaluation** need to satisfy one main criteria:

- Entity clusters should be **complete**: Ensure that no relevant entity mention is omitted. This means all mentions of a specific entity are included in the dataset.

To achieve this goal, we recommend the methodology described below.

1. Randomly sample a single record.
2. Manually identify all other records that match the sampled record. We recommend doing this using a search tool, or by using a spreadsheet containing data for the block to which the sampled record belongs.
3. The result of step (2) is the "true" cluster for the sampled entity mention. Repeat step (1) - (2) until the desired sample size is achieved.

.. note::
    Sampling records at random leads to clusters sampled with **probability proportional to size**. Use the `weights="cluster_size"` argument when prompted to provide appropriate sampling weights.

.. note::
    More sophisticated methodology can be used to speed up the data labeling process:

    - You can use ElasticSearch to help identify records that match the sampled record.
    - You can use a predicted clustering to help identify match candidates for the sampled record.
    - You can use a probabilistic matching algorithm to estimate match probabilities for each record in the block, and then rank records by their match probabilities to help identify match candidates for the sampled record.
    - You can use probability sampling, such as stratified sampling, to improve the efficiency of the data labeling process in view of the performance metrics that you want to estimate.


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

Validation data is a subset of the dataset that has already been labeled and is considered to be highly reliable. Using validation data can help identify errors and inconsistencies in the labeling process. It can also help identify cases where there is disagreement among annotators, and improve the overall accuracy of the labeling process. You can also validate the quality of the labeled data through quality control measures, such as reviewing labeled data for obvious errors and obvious non-matches that may have been introduced by mistake.

Monitor Inter-Annotator Agreement
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Inter-annotator agreement is a measure of the consistency of the annotations provided by multiple annotators. Monitoring inter-annotator agreement can help identify cases where there is disagreement among annotators, and improve the overall accuracy of the labeling process. It can also help identify cases where the guidelines may need to be updated or clarified.
