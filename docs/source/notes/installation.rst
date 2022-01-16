Installation
============
The installation of ChemicalX requires the presence of certain prerequisites.
These are described in great detail in the installation description of
PyTorch Geometric. Please follow the instructions laid out
`here <https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html>`_.
You might also take a look at the `readme file <https://github.com/AstraZeneca/chemicalx>`_
of the ChemicalX repository. The torch-scatter binaries are provided for
Python version <= 3.9.

**PyTorch 1.10.0**

To install the binaries for PyTorch 1.10.0, simply run

.. code-block:: shell

    $ pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.10.0+${CUDA}.html
    $ pip install torchdrug
    $ pip install chemicalx

where `${CUDA}` should be replaced by either `cpu`, `cu102`, or `cu111`
depending on your PyTorch installation.


**Updating the Library**

The package itself can be installed via pip:

.. code-block:: shell

    $ pip install chemicalx

Upgrade your outdated ChemicalX version by using:

.. code-block:: shell

    $ pip install chemicalx --upgrade

To check your current package version just simply run:

.. code-block:: shell

    $ pip freeze | grep chemicalx
