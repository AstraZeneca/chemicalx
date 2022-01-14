"""A collection of full training and evaluation pipelines."""

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import pandas as pd
import torch
from class_resolver import HintOrType
from sklearn.metrics import roc_auc_score
from torch.nn.modules.loss import _Loss
from tqdm import trange

from chemicalx.data import BatchGenerator, DatasetLoader, DrugPairBatch, dataset_resolver
from chemicalx.models import Model, model_resolver

__all__ = [
    "Result",
    "pipeline",
]


@dataclass
class Result:
    """A result package."""

    model: Model
    roc_auc: float
    predictions: pd.DataFrame


def context_forward(model: Model, batch: DrugPairBatch):
    return model(batch.context_features, batch.drug_features_left, batch.drug_features_right)


def contextless_forward(model: Model, batch: DrugPairBatch):
    return model(batch.drug_features_left, batch.drug_features_right)


def pipeline(
    *,
    dataset: HintOrType[DatasetLoader],
    model: HintOrType[Model],
    model_kwargs: Optional[Mapping[str, Any]] = None,
    optimizer_kwargs: Optional[Mapping[str, Any]] = None,
    batch_size: int = 5120,
    epochs: int,
    loss: Optional[_Loss] = None,
    context_features: bool,
    drug_features: bool,
    drug_molecules: bool,
) -> Result:
    """Run the training and evaluation pipeline.

    :param dataset:
        The dataset can be specified in one of three ways:

        1. The name of the dataset
        2. A subclass of :class:`chemicalx.DatasetLoader`
        3. An instance of a :class:`chemicalx.DatasetLoader`
    :param model:
        The model can be specified in one of three ways:

        1. The name of the model
        2. A subclass of :class:`chemicalx.Model`
        3. An instance of a :class:`chemicalx.Model`
    :param model_kwargs:
        Keyword arguments to pass through to the model constructor. Relevant if passing model by string or class.
    :param batch_size:
        The batch size
    :param epochs:
        The number of epochs to train
    :param loss:
        The loss to use. If none given, uses :class:`torch.nn.BCELoss`.
    :returns:
        The area under the AUC curve
    """
    loader = dataset_resolver.make(dataset)

    drug_feature_set = loader.get_drug_features()
    context_feature_set = loader.get_context_features()
    labeled_triples = loader.get_labeled_triples()

    train_triples, test_triples = labeled_triples.train_test_split()

    generator = BatchGenerator(
        batch_size=batch_size,
        context_features=context_features,
        drug_features=drug_features,
        drug_molecules=drug_molecules,
        labels=True,
    )

    generator.set_data(context_feature_set, drug_feature_set, train_triples)

    model = model_resolver.make(model, model_kwargs)

    optimizer = torch.optim.Adam(model.parameters(), **(optimizer_kwargs or {}))

    model.train()

    # Switch depending on context
    if context_features:
        forwarder = context_forward
    else:
        forwarder = contextless_forward

    if loss is None:
        loss = torch.nn.BCELoss()

    for _epoch in trange(epochs):
        for batch in generator:
            optimizer.zero_grad()
            prediction = forwarder(model, batch)
            loss_value = loss(prediction, batch.labels)
            loss_value.backward()
            optimizer.step()

    model.eval()

    generator.set_labeled_triples(test_triples)

    predictions = []
    for batch in generator:
        prediction = forwarder(model, batch)
        prediction = prediction.detach().cpu().numpy()
        identifiers = batch.identifiers
        identifiers["prediction"] = prediction
        predictions.append(identifiers)

    predictions = pd.concat(predictions)

    return Result(
        model=model, predictions=predictions, roc_auc=roc_auc_score(predictions["label"], predictions["prediction"])
    )
