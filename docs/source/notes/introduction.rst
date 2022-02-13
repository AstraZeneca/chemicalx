Introduction
=======================

ChemicalX is a deep learning library for drug-drug interaction, polypharmacy
side effect, and synergy prediction. The library consists of data loaders
and integrated benchmark datasets. It also includes state-of-the-art deep
neural network architectures that solve the drug pair scoring task.
Implemented methods cover traditional SMILES string based techniques
and neural message passing based models.

.. code-block:: latex

     >@article{chemicalx,
               arxivId = {2202.05240},
               author = {Rozemberczki, Benedek and Hoyt, Charles Tapley and Gogleva, Anna and Grabowski, Piotr and Karis, Klas and Lamov, Andrej and Nikolov, Andriy and Nilsson, Sebastian and Ughetto, Michael and Wang, Yu and Derr, Tyler and Gyori, Benjamin M},
               month = {feb},
               title = {{ChemicalX: A Deep Learning Library for Drug Pair Scoring}},
               url = {http://arxiv.org/abs/2202.05240},
               year = {2022}
     }


Overview
========
We shortly overview the fundamental concepts and features of **ChemicalX**
through simple examples. These are the following:

.. contents::
    :local:

Design Philosophy
-----------------

When ``ChemicalX`` was created we wanted to reuse the high level
architectureal elements of ``torch`` and ``torchdrug``. We also wanted to
conceptualize the ideas outlined in `A Unified View of Relational Deep
Learning for Drug Pair Scoring`.

Drug Feature Set
-----------------

.. code-block:: python

    from chemicalx.data import DatasetLoader, BatchGenerator

Context Feature Set
-------------------

.. code-block:: python

    from chemicalx.data import DatasetLoader, BatchGenerator

Labeled Triples
---------------

.. code-block:: python

    from chemicalx.data import DatasetLoader, BatchGenerator

Data Loaders
------------

Data Generators
---------------

Model Layers
------------

Drug pair scoring models in ChemicalX inherit from

Pipelines
---------

Pipelines provide high level abstractions for the end-to-end
training and evaluation of ChemicalX models. Given a dataset
and model a pipeline can easily train and score the model on
the dataset.

.. code-block:: python

    from chemicalx import pipeline
    from chemicalx.models import DeepSynergy
    from chemicalx.data import DrugCombDB

    model = DeepSynergy(context_channels=112,
                        drug_channels=256)

    dataset = DrugCombDB()

    results = pipeline(dataset=dataset,
                       model=model,
                       batch_size=1024,
                       context_features=True,
                       drug_features=True,
                       drug_molecules=False,
                       labels=True,
                       epochs=100)

    results.summarize()

    results.save("~/test_results/")
