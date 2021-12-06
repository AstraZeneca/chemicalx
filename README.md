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
  <img width="90%" src="https://raw.githubusercontent.com/AstraZeneca/chemicalx/master/docs/source/_static/img/text_logo.jpg?sanitize=true" />
</p>

--------------------------------------------------------------------------------

[![PyPI Version][pypi-image]][pypi-url]
[![Docs Status][docs-image]][docs-url]
[![Code Coverage][coverage-image]][coverage-url]
[![Build Status][build-image]][build-url]

**[Documentation](https://pytorch-geometric-temporal.readthedocs.io)** | **[External Resources](https://pytorch-geometric-temporal.readthedocs.io/en/latest/notes/resources.html)** | **[Datasets](https://pytorch-geometric-temporal.readthedocs.io/en/latest/notes/introduction.html#discrete-time-datasets)**

*PyTorch Geometric Temporal* is a temporal (dynamic) extension library for [PyTorch Geometric](https://github.com/rusty1s/pytorch_geometric).

<p align="justify">The library consists of various dynamic and temporal geometric deep learning, embedding, and spatio-temporal regression methods from a variety of published research papers. Moreover, it comes with an easy-to-use dataset loader, train-test splitter and temporal snaphot iterator for dynamic and temporal graphs. The framework naturally provides GPU support. It also comes with a number of benchmark datasets from the epidemological forecasting, sharing economy, energy production and web traffic management domains. Finally, you can also create your own datasets.</p>

The package interfaces well with [Pytorch Lightning](https://pytorch-lightning.readthedocs.io) which allows training on CPUs, single and multiple GPUs out-of-the-box. Take a look at this [introductory example](https://github.com/benedekrozemberczki/pytorch_geometric_temporal/blob/master/examples/recurrent/lightning_example.py) of using PyTorch Geometric Temporal with Pytorch Lighning.

We also provide [detailed examples]() for each of the models.

--------------------------------------------------------------------------------

**Drug Pair Scoring Explained**

--------------------------------------------------------------------------------

**Case Study Tutorials**

We provide in-depth case study tutorials in the [Documentation](https://pytorch-geometric-temporal.readthedocs.io/en/latest/), each covers an aspect of PyTorch Geometric Temporal’s functionality.

**Incremental Training**: [Epidemiological Forecasting Case Study](https://pytorch-geometric-temporal.readthedocs.io/en/latest/notes/introduction.html#epidemiological-forecasting)

**Cumulative Training**: [Web Traffic Management Case Study](https://pytorch-geometric-temporal.readthedocs.io/en/latest/notes/introduction.html#web-traffic-prediction)

--------------------------------------------------------------------------------

**Citing**


If you find *PyTorch Geometric Temporal* and the new datasets useful in your research, please consider adding the following citation:

```bibtex
@inproceedings{rozemberczki2021pytorch,
               author = {Benedek Rozemberczki and Paul Scherer and Yixuan He and George Panagopoulos and Alexander Riedel and Maria Astefanoaei and Oliver Kiss and Ferenc Beres and Guzman Lopez and Nicolas Collignon and Rik Sarkar},
               title = {{PyTorch Geometric Temporal: Spatiotemporal Signal Processing with Neural Machine Learning Models}},
               year = {2021},
               booktitle={Proceedings of the 30th ACM International Conference on Information and Knowledge Management},
               pages = {4564–4573},
}
```

--------------------------------------------------------------------------------

**A simple example**

PyTorch Geometric Temporal makes implementing Dynamic and Temporal Graph Neural Networks quite easy - see the accompanying [tutorial](https://pytorch-geometric-temporal.readthedocs.io/en/latest/notes/introduction.html#applications). For example, this is all it takes to implement a recurrent graph convolutional network with two consecutive [graph convolutional GRU](https://arxiv.org/abs/1612.07659) cells and a linear layer:

```python

```
--------------------------------------------------------------------------------

**Methods Included**

In detail, the following temporal graph neural networks were implemented.


**Recurrent Graph Convolutions**

* **[Paper](docs)** from Authors *et al.*: [Time](paper_link) (Venue 2222)

* **[Paper](docs)** from Authors *et al.*: [Time](paper_link) (Venue 2222)

* **[Paper](docs)** from Authors *et al.*: [Time](paper_link) (Venue 2222)
  
**Attention Aggregated Temporal Graph Convolutions**

* **[Paper](docs)** from Authors *et al.*: [Time](paper_link) (Venue 2222)

* **[Paper](docs)** from Authors *et al.*: [Time](paper_link) (Venue 2222)

* **[Paper](docs)** from Authors *et al.*: [Time](paper_link) (Venue 2222)

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

--------------------------------------------------------------------------------

**Running tests**

```
$ python setup.py test
```
--------------------------------------------------------------------------------

**License**

- [Apache 2.0 License](https://github.com/AstraZeneca/chemicalx/blob/master/LICENSE)
