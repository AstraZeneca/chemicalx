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

**[Documentation](https://chemicalx.readthedocs.io)** | **[External Resources](https://chemicalx.readthedocs.io/en/latest/notes/resources.html)** | **[Datasets](https://chemicalx.readthedocs.io/en/latest/notes/introduction.html#datasets)** | **[Examples](https://github.com/AstraZeneca/chemicalx/tree/main/examples)**

*ChemicalX* is a deep learning library for drug-drug interaction, polypharmacy side effect and synergy prediction. The library consists of data loaders and integrated benchmark datasets. It also includes state-of-the-art deep neural network architectures that solve the [drug pair scoring task](https://arxiv.org/pdf/2111.02916v4.pdf). Implemented methods cover traditional SMILES string based techniques and neural message passing based models.

--------------------------------------------------------------------------------

**Drug Pair Scoring Explained**

Our framework solves the so called [drug pair scoring task](https://arxiv.org/abs/2111.02916) of computational chemistry. In this task a machine learning model has to predict the outcome of administering two drugs together in a biological or chemical context. Deep learning models which solve this task have an architecture with two distinctive parts:

1. A drug encoder layer which takes a pair of drugs as an input (blue and red drugs below).
2. A head layer which outputs scores in the administration context - polypharmacy in our explanatory figure.

<p align="center">
  <img width="90%" src="https://github.com/AstraZeneca/chemicalx/blob/main/images/pair_scoring.jpg?sanitize=true" />
</p>


--------------------------------------------------------------------------------

**Case Study Tutorials**

We provide in-depth case study tutorials in the [Documentation](https://chemicalx.readthedocs.io/en/latest/), each covers an aspect of ChemicalX’s functionality.

--------------------------------------------------------------------------------

**Citing**


If you find *ChemicalX* and the new datasets useful in your research, please consider adding the following citation:

```bibtex
@inproceedings{chemicalx,
               author = {Benedek Rozemberczki and Charles Tapley Hoyt and Benjamin Gyori},
               title = {{ChemicalX: A Deep Learning Library fo Drug Pair Scoring}},
               year = {2022},
}
```

--------------------------------------------------------------------------------

**A simple example**

```python

```
--------------------------------------------------------------------------------

**Methods Included**

In detail, the following temporal graph neural networks were implemented.

**2017**

* **[DeepCCI](docs)** from [DeepCCI: End-to-end Deep Learning for Chemical-Chemical Interaction Prediction](https://arxiv.org/abs/1704.08432) (ACM BCB)

**2018**

* **[DeepDDI](docs)** from [Deep Learning Improves Prediction of Drug–Drug and Drug–Food Interactions](https://www.pnas.org/content/115/18/E4304) (PNAS)

* **[DeepSynergy](docs)** from [DeepSynergy: Predicting Anti-Cancer Drug Synergy with Deep Learning](https://academic.oup.com/bioinformatics/article/34/9/1538/4747884) (Bioinformatics)

**2019**

* **[MR-GNN](docs)** from [MR-GNN: Multi-Resolution and Dual Graph Neural Network for Predicting Structured Entity Interactions](https://arxiv.org/abs/1905.09558) (IJCAI)

* **[MHCADDI](docs)** from [Drug-Drug Adverse Effect Prediction with Graph Co-Attention](https://arxiv.org/pdf/1905.00534v1.pdf) (ICML)

**2020**

* **[CASTER](docs)** from [CASTER: Predicting Drug Interactions with Chemical Substructure Representation](https://arxiv.org/abs/1911.06446) (AAAI)

* **[SSI-DDI](docs)** from [SSI–DDI: Substructure–Substructure Interactions for Drug–Drug Interaction Prediction](https://academic.oup.com/bib/article-abstract/22/6/bbab133/6265181) (Briefings in Bioinformatics)

* **[EPGCN-DS](docs)** from [Structure-Based Drug-Drug Interaction Detection via Expressive Graph Convolutional Networks and Deep Sets](https://ojs.aaai.org/index.php/AAAI/article/view/7236) (AAAI)

* **[AuDNNSynergy](docs)** from [Synergistic Drug Combination Prediction by Integrating Multiomics Data in Deep Learning Models](https://pubmed.ncbi.nlm.nih.gov/32926369/) (Methods in Molecular Biology)

* **[DeepDrug](docs)** from [DeepDrug: A General Graph-Based Deep Learning Framework for Drug Relation Prediction](https://europepmc.org/article/ppr/ppr236757) (PMC)

* **[GCN-BMP](docs)** from [GCN-BMP: Investigating graph representation learning for DDI prediction task](https://www.sciencedirect.com/science/article/pii/S1046202320300608) (Methods)

**2021**

* **[DPDDI](docs)** from [DPDDI: a Deep Predictor for Drug-Drug Interactions](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-020-03724-x) (BMC Bioinformatics)

* **[DeepDDS](docs)** from [DeepDDS: Deep Graph Neural Network with Attention Mechanism to Predict Synergistic Drug Combinations](https://arxiv.org/abs/2107.02467) (Briefings in Bioinformatics)

* **[MatchMaker](docs)** from [MatchMaker: A Deep Learning Framework for Drug Synergy Prediction](https://pubmed.ncbi.nlm.nih.gov/34086576/) (ACM TCBB)

--------------------------------------------------------------------------------

**Auxiliary Layers**



--------------------------------------------------------------------------------


Head over to our [documentation](https://chemicalx.readthedocs.io) to find out more about installation, creation of datasets and a full list of implemented methods and available datasets.
For a quick start, check out the [examples](https://chemicalx.readthedocs.io) in the `examples/` directory.

If you notice anything unexpected, please open an [issue](github.com/AstraZeneca/chemicalx/issues). If you are missing a specific method, feel free to open a [feature request](https://github.com/AstraZeneca/chemicalx/issues).


--------------------------------------------------------------------------------

**Installation**

Binaries are provided for Python version <= 3.9.

**PyTorch 1.9.0**

To install the binaries for PyTorch 1.9.0, simply run

```sh
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.9.0+${CUDA}.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.9.0+${CUDA}.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.9.0+${CUDA}.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.9.0+${CUDA}.html
pip install torch-geometric
pip install chemicalx
```

where `${CUDA}` should be replaced by either `cpu`, `cu102`, or `cu111` depending on your PyTorch installation.

|             | `cpu` | `cu102` | `cu111` |
|-------------|-------|---------|---------|
| **Linux**   | ✅    | ✅      | ✅      |
| **Windows** | ✅    | ✅      | ✅      |
| **macOS**   | ✅    |         |         |

<details>
<summary><b>Expand to see installation guides for older PyTorch versions...</b></summary>


**PyTorch 1.8.0**

To install the binaries for PyTorch 1.8.0, simply run

```sh
$ pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.8.0+${CUDA}.html
$ pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.8.0+${CUDA}.html
$ pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.8.0+${CUDA}.html
$ pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.8.0+${CUDA}.html
$ pip install torch-geometric
$ pip install torch-geometric-temporal
```

where `${CUDA}` should be replaced by either `cpu`, `cu101`, `cu102`, or `cu111` depending on your PyTorch installation.

|             | `cpu` | `cu101` | `cu102` | `cu111` |
|-------------|-------|---------|---------|---------|
| **Linux**   | ✅    | ✅      | ✅      | ✅      |
| **Windows** | ✅    | ✅      | ✅      | ✅      |
| **macOS**   | ✅    |         |         |         |

**PyTorch 1.7.0**

To install the binaries for PyTorch 1.7.0, simply run

```sh
$ pip install torch-scatter==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.7.0.html
$ pip install torch-sparse==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.7.0.html
$ pip install torch-cluster==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.7.0.html
$ pip install torch-spline-conv==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.7.0.html
$ pip install torch-geometric
$ pip install torch-geometric-temporal
```

where `${CUDA}` should be replaced by either `cpu`, `cu92`, `cu101`, `cu102` or `cu110` depending on your PyTorch installation.

|             | `cpu` | `cu92` | `cu101` | `cu102` | `cu110` |
|-------------|-------|--------|---------|---------|---------|
| **Linux**   | ✅    | ✅     | ✅     | ✅      | ✅     |
| **Windows** | ✅    | ❌     | ✅     | ✅      | ✅     |
| **macOS**   | ✅    |        |         |         |         |

--------------------------------------------------------------------------------

**PyTorch 1.6.0**

To install the binaries for PyTorch 1.6.0, simply run

```sh
$ pip install torch-scatter==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.6.0.html
$ pip install torch-sparse==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.6.0.html
$ pip install torch-cluster==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.6.0.html
$ pip install torch-spline-conv==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.6.0.html
$ pip install torch-geometric
$ pip install torch-geometric-temporal
```

where `${CUDA}` should be replaced by either `cpu`, `cu92`, `cu101` or `cu102` depending on your PyTorch installation.

|             | `cpu` | `cu92` | `cu101` | `cu102` |
|-------------|-------|--------|---------|---------|
| **Linux**   | ✅    | ✅    | ✅     | ✅      |
| **Windows** | ✅    | ❌    | ✅     | ✅      |
| **macOS**   | ✅    |        |         |         |


</details>

--------------------------------------------------------------------------------

**Running tests**

```
$ python setup.py test
```
--------------------------------------------------------------------------------

**License**

- [Apache 2.0 License](https://github.com/AstraZeneca/chemicalx/blob/main/LICENSE)
