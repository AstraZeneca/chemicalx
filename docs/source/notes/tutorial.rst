Tutorial
========

In this lightweight tutorial, we will overview an oncology use case.
The same use case is discussed in detail in the ChemicalX design
paper that accompanies the library. We recommend reading the design
paper with the tutorial -- it can deepen the understanding of the users
and connect the concepts discussed in this tutorial and the introduction.

.. contents:: Contents
    :local:

Data Loading and Generator Definition
-------------------------------------

We import the ``DatasetLoader`` and ``BatchGenerator`` from the ``data``
namespace of ChemicalX. In the first step a DrugComDB DatasetLoader is
instantiated -- it is an oncology-related synergy scoring dataset.
The task related to the dataset is to predict the synergistic nature of
drug pair combinations. Using the ``get_context_features()``,
``get_drug_features()`` and ``get_labeled_triples()`` class methods we
load the context features, drug features, and the triples used for training.
Using the triples we generate training and test sets by using 50% of
the triples for training. Finally, we create a ``BatchGenerator`` instance
this will generate drug pair batches of size 1024, while it will return
the drug and context features for each labeled triple.


.. code-block:: python

    from chemicalx.data import DrugCombDB, BatchGenerator

    loader = DrugCombDB()

    context_set = loader.get_context_features()
    drug_set = loader.get_drug_features()
    triples = loader.get_labeled_triples()

    train, test = triples.train_test_split(train_size=0.5)

    generator = BatchGenerator(batch_size=1024,
                               context_features=True,
                               drug_features=True,
                               drug_molecules=False,
                               context_feature_set=context_set,
                               drug_feature_set=drug_set,
                               labeled_triples=train)

Model Training
--------------

We already have a generator to create batches of data. Now we
will need a model, optimizer, and appropriate loss function.
We import the ``torch`` library and ``DeepSynergy`` from the
``models`` namespace of the library. We create a ``DeepSynergy``
model instance and set the number of input channels to be compatible
with the ``DrugCombdDB`` dataset. We define an ``Adam`` optimizer
and a binary cross-entropy instance to accumulate the loss values.
Using these the model is trained for a single epoch. In each step,
we reset the gradients to be zero, make predictions, calculate
the loss value, backpropagate and make a step with the optimizer.

.. code-block:: python

    import torch
    from chemicalx.models import DeepSynergy

    model = DeepSynergy(context_channels=112,
                        drug_channels=256)

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

Model Scoring
-------------

We will store the predictions in the ``pandas`` data frames and because
of this, we import the ``pandas`` library. We set the model to be in
evaluation mode and we assign the test set triples to the generator.
We accumulate the predictions in a ``list`` and iterate over the
batches in the generator. In each step we make predictions for the
drug pairs in the batch, these are detached from the computation
graph and added the batch identifiers ``DataFrame``. This is
appended to the predictions in each step. Finally, the predictions
are turned into a ``DataFrame``.

.. code-block:: python

    import pandas as pd

    model.eval()
    generator.labeled_triples = test

    predictions = []
    for batch in generator:
        prediction = model(batch.context_features,
                           batch.drug_features_left,
                           batch.drug_features_right)
        prediction = prediction.detach().cpu().numpy()
        identifiers = batch.identifiers
        identifiers["prediction"] = prediction
        predictions.append(identifiers)

    predictions = pd.concat(predictions)
