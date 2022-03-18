Data Cleaning
=================

ChemicalX comes with benchmark datasets that we pre-processed.
In this section of the documentation we discuss how we obtained the raw data.
We also discuss what pre-processing steps have been taken.
We do this for each of the datasets in the framework.

.. contents::
    :local:

Drugbank DDI
-------------------

* We used the cleaned dataset from the Therapeutic Data Commons.
* Drug identifiers are represented by the DrugBank identifier.
* Contexts are represented by drug-drug interaction identifiers from DrugBank.
* Using RDKit 2021.09.03. we generated 256-dimensional Morgan fingerprints.
* Labels represent the presence of a specific drug-drug interaction.
* Context features are one-hot encoded binary vectors.
* We generated an equal number of negative samples as positives.
* Negative samples do not contain collisions.

TwoSides
-------------------

* This datasets is a *subsample* of TwoSides.
* We only included the 100 most common side effects.
* We used the cleaned dataset from the Therapeutic Data Commons.
* Drug identifiers are represented by the DrugBank identifier.
* Contexts are represented by the top 10 most common side effects in TwoSides.
* Using RDKit 2021.09.03. we generated 256-dimensional Morgan fingerprints.
* Labels represent the presence of a specific drug-drug interaction.
* Context features are one-hot encoded binary vectors.
* We generated an equal number of negative samples as positives.
* Negative samples do not contain collisions.


