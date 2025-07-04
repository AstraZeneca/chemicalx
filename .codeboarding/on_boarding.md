```mermaid

graph LR

    Data_Management_Preparation["Data Management & Preparation"]

    Model_Core_Architectures["Model Core & Architectures"]

    Training_Evaluation_Pipeline["Training & Evaluation Pipeline"]

    Utilities["Utilities"]

    Data_Management_Preparation -- "provides prepared data batches to" --> Training_Evaluation_Pipeline

    Training_Evaluation_Pipeline -- "interacts with" --> Model_Core_Architectures

    Training_Evaluation_Pipeline -- "leverages functions from" --> Utilities

    click Data_Management_Preparation href "https://github.com/AstraZeneca/chemicalx/blob/main/.codeboarding//Data_Management_Preparation.md" "Details"

    click Model_Core_Architectures href "https://github.com/AstraZeneca/chemicalx/blob/main/.codeboarding//Model_Core_Architectures.md" "Details"

    click Training_Evaluation_Pipeline href "https://github.com/AstraZeneca/chemicalx/blob/main/.codeboarding//Training_Evaluation_Pipeline.md" "Details"

```



[![CodeBoarding](https://img.shields.io/badge/Generated%20by-CodeBoarding-9cf?style=flat-square)](https://github.com/CodeBoarding/GeneratedOnBoardings)[![Demo](https://img.shields.io/badge/Try%20our-Demo-blue?style=flat-square)](https://www.codeboarding.org/demo)[![Contact](https://img.shields.io/badge/Contact%20us%20-%20contact@codeboarding.org-lightgrey?style=flat-square)](mailto:contact@codeboarding.org)



## Details



Abstract Components Overview and Relationships for ChemicalX library.



### Data Management & Preparation [[Expand]](./Data_Management_Preparation.md)

This component is responsible for the entire lifecycle of data, from loading raw chemical and biological datasets (drug features, context features, interaction triples) from various sources (remote/local) to structuring and transforming them into optimized batches (`DrugPairBatch`) suitable for efficient model consumption during training and inference. It manages feature sets and ensures data integrity and flow, including utility functions for data handling.





**Related Classes/Methods**:



- <a href="https://github.com/AstraZeneca/chemicalx/blob/main/chemicalx/data/datasetloader.py#L1-L1" target="_blank" rel="noopener noreferrer">`chemicalx.data.datasetloader` (1:1)</a>

- <a href="https://github.com/AstraZeneca/chemicalx/blob/main/chemicalx/data/contextfeatureset.py#L1-L1" target="_blank" rel="noopener noreferrer">`chemicalx.data.contextfeatureset` (1:1)</a>

- <a href="https://github.com/AstraZeneca/chemicalx/blob/main/chemicalx/data/drugfeatureset.py#L1-L1" target="_blank" rel="noopener noreferrer">`chemicalx.data.drugfeatureset` (1:1)</a>

- <a href="https://github.com/AstraZeneca/chemicalx/blob/main/chemicalx/data/labeledtriples.py#L1-L1" target="_blank" rel="noopener noreferrer">`chemicalx.data.labeledtriples` (1:1)</a>

- <a href="https://github.com/AstraZeneca/chemicalx/blob/main/chemicalx/data/batchgenerator.py#L1-L1" target="_blank" rel="noopener noreferrer">`chemicalx.data.batchgenerator` (1:1)</a>

- <a href="https://github.com/AstraZeneca/chemicalx/blob/main/chemicalx/data/drugpairbatch.py#L1-L1" target="_blank" rel="noopener noreferrer">`chemicalx.data.drugpairbatch` (1:1)</a>

- <a href="https://github.com/AstraZeneca/chemicalx/blob/main/chemicalx/data/utils.py#L1-L1" target="_blank" rel="noopener noreferrer">`chemicalx.data.utils` (1:1)</a>





### Model Core & Architectures [[Expand]](./Model_Core_Architectures.md)

This component defines the foundational abstract interface (`chemicalx.models.base.Model`) for all deep learning models within the library. It also includes the concrete implementations of various high-level model architectures (e.g., `CASTER`, `DeepDDI`, `MHCADDI`, `GCNBMP`, `SSIDDI`) tailored for specific tasks like drug-drug interaction or synergy prediction, encapsulating their overall neural network designs.





**Related Classes/Methods**:



- <a href="https://github.com/AstraZeneca/chemicalx/blob/main/chemicalx/models/base.py#L1-L1" target="_blank" rel="noopener noreferrer">`chemicalx.models.base` (1:1)</a>

- <a href="https://github.com/AstraZeneca/chemicalx/blob/main/chemicalx/models/caster.py#L1-L1" target="_blank" rel="noopener noreferrer">`chemicalx.models.caster` (1:1)</a>

- <a href="https://github.com/AstraZeneca/chemicalx/blob/main/chemicalx/models/deepddi.py#L1-L1" target="_blank" rel="noopener noreferrer">`chemicalx.models.deepddi` (1:1)</a>

- <a href="https://github.com/AstraZeneca/chemicalx/blob/main/chemicalx/models/deepdds.py#L1-L1" target="_blank" rel="noopener noreferrer">`chemicalx.models.deepdds` (1:1)</a>

- <a href="https://github.com/AstraZeneca/chemicalx/blob/main/chemicalx/models/deepdrug.py#L1-L1" target="_blank" rel="noopener noreferrer">`chemicalx.models.deepdrug` (1:1)</a>

- <a href="https://github.com/AstraZeneca/chemicalx/blob/main/chemicalx/models/deepsynergy.py#L1-L1" target="_blank" rel="noopener noreferrer">`chemicalx.models.deepsynergy` (1:1)</a>

- <a href="https://github.com/AstraZeneca/chemicalx/blob/main/chemicalx/models/epgcnds.py#L1-L1" target="_blank" rel="noopener noreferrer">`chemicalx.models.epgcnds` (1:1)</a>

- <a href="https://github.com/AstraZeneca/chemicalx/blob/main/chemicalx/models/gcnbmp.py#L1-L1" target="_blank" rel="noopener noreferrer">`chemicalx.models.gcnbmp` (1:1)</a>

- <a href="https://github.com/AstraZeneca/chemicalx/blob/main/chemicalx/models/matchmaker.py#L1-L1" target="_blank" rel="noopener noreferrer">`chemicalx.models.matchmaker` (1:1)</a>

- <a href="https://github.com/AstraZeneca/chemicalx/blob/main/chemicalx/models/mhcaddi.py#L1-L1" target="_blank" rel="noopener noreferrer">`chemicalx.models.mhcaddi` (1:1)</a>

- <a href="https://github.com/AstraZeneca/chemicalx/blob/main/chemicalx/models/mrgnn.py#L1-L1" target="_blank" rel="noopener noreferrer">`chemicalx.models.mrgnn` (1:1)</a>

- <a href="https://github.com/AstraZeneca/chemicalx/blob/main/chemicalx/models/ssiddi.py#L1-L1" target="_blank" rel="noopener noreferrer">`chemicalx.models.ssiddi` (1:1)</a>





### Training & Evaluation Pipeline [[Expand]](./Training_Evaluation_Pipeline.md)

This component orchestrates the end-to-end workflow for training, validating, and evaluating deep learning models. It manages the training loop, handles device placement (CPU/GPU), and processes the results generated by the models, providing a structured and reproducible way to run experiments.





**Related Classes/Methods**:



- <a href="https://github.com/AstraZeneca/chemicalx/blob/main/chemicalx/pipeline.py#L72-L201" target="_blank" rel="noopener noreferrer">`chemicalx.pipeline.pipeline` (72:201)</a>

- <a href="https://github.com/AstraZeneca/chemicalx/blob/main/chemicalx/pipeline.py#L36-L69" target="_blank" rel="noopener noreferrer">`chemicalx.pipeline.Result` (36:69)</a>





### Utilities

This component offers a collection of general-purpose helper functions and mathematical operations that support various parts of the library. This includes tensor manipulation functions (e.g., segment operations for sparse tensors) and system-level utilities like device resolution.





**Related Classes/Methods**:



- <a href="https://github.com/AstraZeneca/chemicalx/blob/main/chemicalx/utils.py#L1-L1" target="_blank" rel="noopener noreferrer">`chemicalx.utils` (1:1)</a>









### [FAQ](https://github.com/CodeBoarding/GeneratedOnBoardings/tree/main?tab=readme-ov-file#faq)