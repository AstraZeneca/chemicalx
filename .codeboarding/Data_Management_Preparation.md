```mermaid

graph LR

    Data_Loaders["Data Loaders"]

    Feature_Management["Feature Management"]

    Interaction_Data_Labeled_Triples_["Interaction Data (Labeled Triples)"]

    Batching_Data_Preparation["Batching & Data Preparation"]

    DrugPairBatch["DrugPairBatch"]

    Data_Utilities["Data Utilities"]

    Data_Loaders -- "Populates" --> Feature_Management

    Data_Loaders -- "Populates" --> Interaction_Data_Labeled_Triples_

    Feature_Management -- "Populated by" --> Data_Loaders

    Feature_Management -- "Queried by" --> Batching_Data_Preparation

    Interaction_Data_Labeled_Triples_ -- "Populated by" --> Data_Loaders

    Interaction_Data_Labeled_Triples_ -- "Iterated by" --> Batching_Data_Preparation

    Batching_Data_Preparation -- "Queries" --> Feature_Management

    Batching_Data_Preparation -- "Iterates over" --> Interaction_Data_Labeled_Triples_

    Batching_Data_Preparation -- "Constructs" --> DrugPairBatch

    DrugPairBatch -- "Constructed by" --> Batching_Data_Preparation

    Data_Utilities -- "Used by" --> Data_Loaders

    Data_Utilities -- "Used by" --> Feature_Management

    Data_Utilities -- "Used by" --> Interaction_Data_Labeled_Triples_

```



[![CodeBoarding](https://img.shields.io/badge/Generated%20by-CodeBoarding-9cf?style=flat-square)](https://github.com/CodeBoarding/GeneratedOnBoardings)[![Demo](https://img.shields.io/badge/Try%20our-Demo-blue?style=flat-square)](https://www.codeboarding.org/demo)[![Contact](https://img.shields.io/badge/Contact%20us%20-%20contact@codeboarding.org-lightgrey?style=flat-square)](mailto:contact@codeboarding.org)



## Details



Final Component Overview for Data Management & Preparation within the chemicalx project



### Data Loaders

This component is the primary interface for acquiring raw chemical and biological datasets. It abstracts the complexities of fetching data from various sources (remote or local) and specific public datasets. It's responsible for populating the initial `DrugFeatureSet`, `ContextFeatureSet`, and `LabeledTriples` objects. This aligns with the "Data Loaders" pattern, providing a unified API for diverse data sources.





**Related Classes/Methods**:



- <a href="https://github.com/AstraZeneca/chemicalx/blob/main/chemicalx/data/datasetloader.py#L1-L1" target="_blank" rel="noopener noreferrer">`chemicalx.data.datasetloader` (1:1)</a>





### Feature Management

This component defines and manages the structured representation of chemical features for drugs (`DrugFeatureSet`) and contextual information (`ContextFeatureSet`). These classes encapsulate the feature data, ensuring consistency and providing methods for accessing and manipulating them. They serve as the organized repositories for all input features required by the models. This aligns with "Feature Processors" and "Datasets" patterns.





**Related Classes/Methods**:



- <a href="https://github.com/AstraZeneca/chemicalx/blob/main/chemicalx/data/drugfeatureset.py#L1-L1" target="_blank" rel="noopener noreferrer">`chemicalx.data.drugfeatureset` (1:1)</a>

- <a href="https://github.com/AstraZeneca/chemicalx/blob/main/chemicalx/data/contextfeatureset.py#L1-L1" target="_blank" rel="noopener noreferrer">`chemicalx.data.contextfeatureset` (1:1)</a>





### Interaction Data (Labeled Triples)

This component represents the core interaction data, typically in the form of (drug1, drug2, label) tuples. It provides functionalities for managing these triples, such as splitting data into training and testing sets, and querying counts of positive/negative interactions. It is the ground truth dataset for model training and evaluation. This is a fundamental "Dataset" component.





**Related Classes/Methods**:



- <a href="https://github.com/AstraZeneca/chemicalx/blob/main/chemicalx/data/labeledtriples.py#L1-L1" target="_blank" rel="noopener noreferrer">`chemicalx.data.labeledtriples` (1:1)</a>





### Batching & Data Preparation

This component is responsible for transforming the raw feature sets and labeled triples into optimized mini-batches (`DrugPairBatch`) suitable for efficient consumption by deep learning models. It iterates through the interaction data, retrieves corresponding features, and constructs the final batch objects. This is a critical part of the "Training Pipeline" and "Data Preprocessing Scripts".





**Related Classes/Methods**:



- <a href="https://github.com/AstraZeneca/chemicalx/blob/main/chemicalx/data/batchgenerator.py#L1-L1" target="_blank" rel="noopener noreferrer">`chemicalx.data.batchgenerator` (1:1)</a>





### DrugPairBatch

This is a specialized data structure designed to hold a single mini-batch of drug pair interaction data, including drug features, context features, and labels, in a format optimized for PyTorch models. It acts as the standardized input for the deep learning models, ensuring efficient data flow during training and inference. This is a key "Data Structure" component.





**Related Classes/Methods**:



- <a href="https://github.com/AstraZeneca/chemicalx/blob/main/chemicalx/data/drugpairbatch.py#L1-L1" target="_blank" rel="noopener noreferrer">`chemicalx.data.drugpairbatch` (1:1)</a>





### Data Utilities

This component provides a collection of general helper functions for common data-related tasks, such as writing data to JSON files or extracting features. These utilities support the overall data preparation and management processes, promoting code reusability and simplifying various data handling operations across the subsystem.





**Related Classes/Methods**:



- <a href="https://github.com/AstraZeneca/chemicalx/blob/main/chemicalx/data/utils.py#L1-L1" target="_blank" rel="noopener noreferrer">`chemicalx.data.utils` (1:1)</a>









### [FAQ](https://github.com/CodeBoarding/GeneratedOnBoardings/tree/main?tab=readme-ov-file#faq)