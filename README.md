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

We are going to overview a short example of training a machine learning model on DrugCombDB. In the first part of this illustrative example
we import the base PyTorch library, data loaders and the DeepSynergy model from ChemicalX. We load the feature sets, triples and create a generator for the training split that we create. We will use this to train the DeepSynergy model.

```python
import torch
from chemicalx.model import DeepSynergy
from chemicalx.data import DatasetLoader, BatchGenerator

loader = DatasetLoader("drugcombdb")

drug_feature_set = loader.get_drug_features()
context_feature_set = loader.get_context_features()
labeled_triples = loader.get_labeled_triples()

train_triples, test_triples = labeled_triples.train_test_split()

generator = BatchGenerator(batch_size=5120,
                           context_features=True,
                           drug_features=True,
                           drug_molecules=False,
                           labels=True)

generator.set_data(context_feature_set, drug_feature_set, train_triples)
```

We define the DeepSynergy model - DrugCombDB has 288 context and 256 drug features. Other hyperparameters of the model are left as defaults. We define
an Adap optimizer instance, set the model to be in training model. We generate batches from the training data generator and train the model by
minimizing binary cross entropy. 

```python

model = DeepSynergy(context_channels=112, drug_channels=256)

optimizer = torch.optim.Adam(model.parameters())

model.train()

loss = torch.nn.BCELoss()

for batch in generator:
    optimizer.zero_grad()

    prediction = model(batch.context_features,
                       batch.drug_features_left,
                       batch.drug_features_right)

    loss_value = loss(prediction, batch.labels)
    loss_value.backward()
    optimizer.step()
```
--------------------------------------------------------------------------------

**Methods Included**

In detail, the following temporal graph neural networks were implemented.

**2017**

* **[DeepCCI](https://chemicalx.readthedocs.io/en/latest/modules/root.html#chemicalx.models.deepcci.DeepCCI)** from [DeepCCI: End-to-end Deep Learning for Chemical-Chemical Interaction Prediction](https://arxiv.org/abs/1704.08432) (ACM BCB)

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

* **[AuDNNSynergy](https://chemicalx.readthedocs.io/en/latest/modules/root.html#chemicalx.models.audnnsynergy.AUDNNSynergy)** from [Synergistic Drug Combination Prediction by Integrating Multiomics Data in Deep Learning Models](https://pubmed.ncbi.nlm.nih.gov/32926369/) (Methods in Molecular Biology)

* **[DeepDrug](https://chemicalx.readthedocs.io/en/latest/modules/root.html#chemicalx.models.deepdrug.DeepDrug)** from [DeepDrug: A General Graph-Based Deep Learning Framework for Drug Relation Prediction](https://europepmc.org/article/ppr/ppr236757) (PMC)

* **[GCN-BMP](https://chemicalx.readthedocs.io/en/latest/modules/root.html#chemicalx.models.gcnbmp.GCNBMP)** from [GCN-BMP: Investigating graph representation learning for DDI prediction task](https://www.sciencedirect.com/science/article/pii/S1046202320300608) (Methods)

**2021**

* **[DPDDI](https://chemicalx.readthedocs.io/en/latest/modules/root.html#chemicalx.models.dpddi.DPDDI)** from [DPDDI: a Deep Predictor for Drug-Drug Interactions](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-020-03724-x) (BMC Bioinformatics)

* **[DeepDDS](https://chemicalx.readthedocs.io/en/latest/modules/root.html#chemicalx.models.deepdds.DeepDDS)** from [DeepDDS: Deep Graph Neural Network with Attention Mechanism to Predict Synergistic Drug Combinations](https://arxiv.org/abs/2107.02467) (Briefings in Bioinformatics)

* **[MatchMaker](https://chemicalx.readthedocs.io/en/latest/modules/root.html#chemicalx.models.matchmaker.MatchMaker)** from [MatchMaker: A Deep Learning Framework for Drug Synergy Prediction](https://pubmed.ncbi.nlm.nih.gov/34086576/) (ACM TCBB)

--------------------------------------------------------------------------------

**Auxiliary Layers**



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
$ python setup.py test
```
--------------------------------------------------------------------------------

**License**

- [Apache 2.0 License](https://github.com/AstraZeneca/chemicalx/blob/main/LICENSE)
