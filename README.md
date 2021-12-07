[pypi-image]: https://badge.fury.io/py/chemicalx.svg
[pypi-url]: https://pypi.python.org/pypi/chemicalx
[size-image]: https://img.shields.io/github/repo-size/AstraZeneca/chemicalx.svg
[size-url]: https://github.com/AstraZeneca/chemicalx/archive/master.zip
[build-image]: https://github.com/AstraZeneca/chemicalx/workflows/CI/badge.svg
[build-url]: https://github.com/AstraZeneca/chemicalx/actions?query=workflow%3ACI
[docs-image]: https://readthedocs.org/projects/chemicalx/badge/?version=latest
[docs-url]: https://chemicalx.readthedocs.io/en/latest/?badge=latest
[coverage-image]: https://codecov.io/gh/AstraZeneca/chemicalx/branch/master/graph/badge.svg
[coverage-url]: https://codecov.io/github/AstraZeneca/chemicalx?branch=master



<p align="center">
  <img width="90%" src="https://github.com/AstraZeneca/chemicalx/blob/master/images/chemicalx_logo.jpg?sanitize=true" />
</p>

--------------------------------------------------------------------------------

[![PyPI Version][pypi-image]][pypi-url]
[![Docs Status][docs-image]][docs-url]
[![Code Coverage][coverage-image]][coverage-url]
[![Build Status][build-image]][build-url]

**[Documentation](https://chemicalx.readthedocs.io)** | **[External Resources](https://chemicalx.readthedocs.io/en/latest/notes/resources.html)** | **[Datasets](https://chemicalx.readthedocs.io/en/latest/notes/introduction.html#datasets)**

*ChemicalX* is a deep learning library for drug-drug interaction, polypharmacy side effect and synergy prediction.

<p align="justify">The library consists of ...</p>

The package interfaces well with [Pytorch Lightning](https://pytorch-lightning.readthedocs.io) which allows training on CPUs, single and multiple GPUs out-of-the-box. Take a look at this [introductory example](https://github.com/AstraZeneca/chemicalx/blob/master/examples/polypharmacy_example.py) of using ChemicalX with Pytorch Lighning.

We also provide [detailed examples]() for each of the models.

--------------------------------------------------------------------------------

**Drug Pair Scoring Explained**


<p align="center">
  <img width="90%" src="https://github.com/AstraZeneca/chemicalx/blob/master/images/pair_scoring.jpg?sanitize=true" />
</p>


--------------------------------------------------------------------------------

**Case Study Tutorials**

We provide in-depth case study tutorials in the [Documentation](https://chemicalx.readthedocs.io/en/latest/), each covers an aspect of ChemicalX’s functionality.

--------------------------------------------------------------------------------

**Citing**


If you find *ChemicalX* and the new datasets useful in your research, please consider adding the following citation:

```bibtex
@inproceedings{chemicalx,
               author = {Name Name and Name Name},
               title = {{ChemicalX: A Deep Learning Library fo Drug Pair Scoring}},
               year = {2022},
}
```

--------------------------------------------------------------------------------

**A simple example**

ChemicalX

```python

```
--------------------------------------------------------------------------------

**Methods Included**

In detail, the following temporal graph neural networks were implemented.

* **[DeepCCI](docs)** from Authors *et al.*: [Time](paper_link) (Venue 2222)

* **[DeepDDI](docs)** from Authors *et al.*: [Time](paper_link) (Venue 2222)

* **[DeepSynergy](docs)** from Authors *et al.*: [Time](paper_link) (Venue 2222)

* **[MRH-GNN](docs)** from Authors *et al.*: [Time](paper_link) (Venue 2222)

* **[MHCADDI](docs)** from Authors *et al.*: [Time](paper_link) (Venue 2222)

* **[CASTER](docs)** from Authors *et al.*: [Time](paper_link) (Venue 2222)

* **[SSI-DDI](docs)** from Authors *et al.*: [Time](paper_link) (Venue 2222)

* **[EPGCN-DS](docs)** from Authors *et al.*: [Time](paper_link) (Venue 2222)

* **[AuDNNSynergy](docs)** from Authors *et al.*: [Time](paper_link) (Venue 2222)

* **[DeepDrug](docs)** from Authors *et al.*: [Time](paper_link) (Venue 2222)

* **[GCN-BMP](docs)** from Authors *et al.*: [Time](paper_link) (Venue 2222)

* **[DPDDI](docs)** from Authors *et al.*: [Time](paper_link) (Venue 2222)

* **[SmileGNN](docs)** from Authors *et al.*: [Time](paper_link) (Venue 2222)

* **[DeepDDS](docs)** from Authors *et al.*: [Time](paper_link) (Venue 2222)

* **[MatchMaker](docs)** from Authors *et al.*: [Time](paper_link) (Venue 2222)

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

- [Apache 2.0 License](https://github.com/AstraZeneca/chemicalx/blob/master/LICENSE)
