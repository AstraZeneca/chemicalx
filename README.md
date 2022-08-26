[pypi-image]: https://badge.fury.io/py/chemicalx.svg
[pypi-url]: https://pypi.python.org/pypi/chemicalx
[size-image]: https://img.shields.io/github/repo-size/AstraZeneca/chemicalx.svg
[size-url]: https://github.com/AstraZeneca/chemicalx/archive/main.zip
[build-image]: https://github.com/AstraZeneca/chemicalx/workflows/CI/badge.svg
[build-url]: https://github.com/AstraZeneca/chemicalx/actions?query=workflow%3ACI
[docs-image]: https://readthedocs.org/projects/chemicalx/badge/?version=latest
[docs-url]: https://chemicalx.readthedocs.io/en/latest/?badge=latest
[coverage-image]: https://codecov.io/gh/AstraZeneca/chemicalx/branch/main/graph/badge.svg
[coverage-url]: https://codecov.io/github/AstraZeneca/chemicalx?branch=main

<p align="center">
  <img width="90%" src="https://github.com/AstraZeneca/chemicalx/blob/main/images/chemicalx_logo.jpg?sanitize=true" />
</p>

--------------------------------------------------------------------------------

[![PyPI Version][pypi-image]][pypi-url]
[![Docs Status][docs-image]][docs-url]
[![Code Coverage][coverage-image]][coverage-url]
[![Build Status][build-image]][build-url]
[![DOI](https://img.shields.io/badge/DOI-10.1145/3534678.3539023-blue.svg)](https://bioregistry.io/doi:10.1145/3534678.3539023)

**[Documentation](https://chemicalx.readthedocs.io)** | **[External Resources](https://chemicalx.readthedocs.io/en/latest/notes/resources.html)** | **[Datasets](https://chemicalx.readthedocs.io/en/latest/notes/introduction.html#datasets)** | **[Examples](https://github.com/AstraZeneca/chemicalx/tree/main/examples)** 

*ChemicalX* is a deep learning library for drug-drug interaction, polypharmacy side effect, and synergy prediction. The library consists of data loaders and integrated benchmark datasets. It also includes state-of-the-art deep neural network architectures that solve the [drug pair scoring task](https://arxiv.org/pdf/2111.02916v4.pdf). Implemented methods cover traditional SMILES string based techniques and neural message passing based models.

--------------------------------------------------------------------------------

**Citing**


If you find *ChemicalX* and the new datasets useful in your research, please consider adding the following citation:

```bibtex
@inproceedings{10.1145/3534678.3539023,
  author = {Rozemberczki, Benedek and Hoyt, Charles Tapley and Gogleva, Anna and Grabowski, Piotr and Karis, Klas and Lamov, Andrej and Nikolov, Andriy and Nilsson, Sebastian and Ughetto, Michael and Wang, Yu and Derr, Tyler and Gyori, Benjamin M.},
  title = {ChemicalX: A Deep Learning Library for Drug Pair Scoring},
  year = {2022},
  isbn = {9781450393850},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  url = {https://doi.org/10.1145/3534678.3539023},
  doi = {10.1145/3534678.3539023},
  booktitle = {Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  pages = {3819–3828},
  numpages = {10},
  keywords = {chemistry, neural networks, deep learning},
  location = {Washington DC, USA},
  series = {KDD '22}
}
```
--------------------------------------------------------------------------------

**Drug Pair Scoring Explained**

Our framework solves the [drug pair scoring task](https://arxiv.org/abs/2111.02916) of computational chemistry. In this task a machine learning model has to predict the outcome of administering two drugs together in a biological or chemical context. Deep learning models which solve this task have an architecture with two distinctive parts:

1. A drug encoder layer which takes a pair of drugs as an input (blue and red drugs below).
2. A head layer which outputs scores in the administration context - polypharmacy in our explanatory figure.

<p align="center">
  <img width="90%" src="https://github.com/AstraZeneca/chemicalx/blob/main/images/pair_scoring.jpg?sanitize=true" />
</p>


**Getting Started**

The API of `chemicalx` provides a high-level function for training and evaluating models
that's heavily influenced by the [PyKEEN](https://github.com/pykeen/pykeen/)
training and evaluation pipeline:

```python
from chemicalx import pipeline
from chemicalx.models import DeepSynergy
from chemicalx.data import DrugCombDB

model = DeepSynergy(context_channels=112, drug_channels=256)
dataset = DrugCombDB()

results = pipeline(
    dataset=dataset,
    model=model,
    # Data arguments
    batch_size=5120,
    context_features=True,
    drug_features=True,
    drug_molecules=False,
    # Training arguments
    epochs=100,
)

# Outputs information about the AUC-ROC, etc. to the console.
results.summarize()

# Save the model, losses, evaluation, and other metadata.
results.save("~/test_results/")
```

--------------------------------------------------------------------------------

**Case Study Tutorials**

We provide in-depth case study like tutorials in the [Documentation](https://chemicalx.readthedocs.io/en/latest/), each covers an aspect of ChemicalX’s functionality.

--------------------------------------------------------------------------------

**Methods Included**

In detail, the following drug pair scoring models were implemented.

**2018**

* **[DeepDDI](https://chemicalx.readthedocs.io/en/latest/modules/root.html#chemicalx.models.deepddi.DeepDDI)** from [Deep Learning Improves Prediction of Drug–Drug and Drug–Food Interactions](https://www.pnas.org/content/115/18/E4304) (PNAS)

* **[DeepSynergy](https://chemicalx.readthedocs.io/en/latest/modules/root.html#chemicalx.models.deepsynergy.DeepSynergy)** from [DeepSynergy: Predicting Anti-Cancer Drug Synergy with Deep Learning](https://academic.oup.com/bioinformatics/article/34/9/1538/4747884) (Bioinformatics)

**2019**

* **[MR-GNN](https://chemicalx.readthedocs.io/en/latest/modules/root.html#chemicalx.models.mrgnn.MRGNN)** from [MR-GNN: Multi-Resolution and Dual Graph Neural Network for Predicting Structured Entity Interactions](https://arxiv.org/abs/1905.09558) (IJCAI)

* **[MHCADDI](https://chemicalx.readthedocs.io/en/latest/modules/root.html#chemicalx.models.mhcaddi.MHCADDI)** from [Drug-Drug Adverse Effect Prediction with Graph Co-Attention](https://arxiv.org/pdf/1905.00534v1.pdf) (ICML)

**2020**

* **[CASTER](https://chemicalx.readthedocs.io/en/latest/modules/root.html#chemicalx.models.caster.CASTER)** from [CASTER: Predicting Drug Interactions with Chemical Substructure Representation](https://arxiv.org/abs/1911.06446) (AAAI)

* **[SSI-DDI](https://chemicalx.readthedocs.io/en/latest/modules/root.html#chemicalx.models.ssiddi.SSIDDI)** from [SSI–DDI: Substructure–Substructure Interactions for Drug–Drug Interaction Prediction](https://academic.oup.com/bib/article-abstract/22/6/bbab133/6265181) (Briefings in Bioinformatics)

* **[EPGCN-DS](https://chemicalx.readthedocs.io/en/latest/modules/root.html#chemicalx.models.epgcnds.EPGCNDS)** from [Structure-Based Drug-Drug Interaction Detection via Expressive Graph Convolutional Networks and Deep Sets](https://ojs.aaai.org/index.php/AAAI/article/view/7236) (AAAI)

* **[DeepDrug](https://chemicalx.readthedocs.io/en/latest/modules/root.html#chemicalx.models.deepdrug.DeepDrug)** from [DeepDrug: A General Graph-Based Deep Learning Framework for Drug Relation Prediction](https://europepmc.org/article/ppr/ppr236757) (PMC)

* **[GCN-BMP](https://chemicalx.readthedocs.io/en/latest/modules/root.html#chemicalx.models.gcnbmp.GCNBMP)** from [GCN-BMP: Investigating graph representation learning for DDI prediction task](https://www.sciencedirect.com/science/article/pii/S1046202320300608) (Methods)

**2021**

* **[DeepDDS](https://chemicalx.readthedocs.io/en/latest/modules/root.html#chemicalx.models.deepdds.DeepDDS)** from [DeepDDS: Deep Graph Neural Network with Attention Mechanism to Predict Synergistic Drug Combinations](https://arxiv.org/abs/2107.02467) (Briefings in Bioinformatics)

* **[MatchMaker](https://chemicalx.readthedocs.io/en/latest/modules/root.html#chemicalx.models.matchmaker.MatchMaker)** from [MatchMaker: A Deep Learning Framework for Drug Synergy Prediction](https://pubmed.ncbi.nlm.nih.gov/34086576/) (ACM TCBB)

--------------------------------------------------------------------------------

Head over to our [documentation](https://chemicalx.readthedocs.io) to find out more about installation, creation of datasets and a full list of implemented methods and available datasets.
For a quick start, check out the [examples](https://chemicalx.readthedocs.io) in the `examples/` directory.

If you notice anything unexpected, please open an [issue](github.com/AstraZeneca/chemicalx/issues). If you are missing a specific method, feel free to open a [feature request](https://github.com/AstraZeneca/chemicalx/issues).


--------------------------------------------------------------------------------

**Installation**

**PyTorch 1.10.0**

To install for PyTorch 1.10.0, simply run

```sh
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.10.0+${CUDA}.html
pip install torchdrug
pip install chemicalx
```

where `${CUDA}` should be replaced by either `cpu`, `cu102`, or `cu111` depending on your PyTorch installation.

|             | `cpu` | `cu102` | `cu111` |
|-------------|-------|---------|---------|
| **Linux**   | ✅    | ✅      | ✅      |
| **Windows** | ✅    | ✅      | ✅      |
| **macOS**   | ✅    |         |         |


--------------------------------------------------------------------------------

**Running tests**

```
$ tox -e py
```
--------------------------------------------------------------------------------

**License**

- [Apache 2.0 License](https://github.com/AstraZeneca/chemicalx/blob/main/LICENSE)
