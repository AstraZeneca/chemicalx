"""A module for the batch generator class."""

import math
from typing import Iterable, Iterator, List, Optional

import numpy as np
import pandas as pd
import torch
from torchdrug.data import PackedGraph

from .contextfeatureset import ContextFeatureSet
from .drugfeatureset import DrugFeatureSet
from .drugpairbatch import DrugPairBatch
from .labeledtriples import LabeledTriples

__all__ = [
    "BatchGenerator",
]


class BatchGenerator(Iterator[DrugPairBatch]):
    """Generator to create batches of drug pairs efficiently."""

    def __init__(
        self,
        batch_size: int,
        context_features: bool,
        drug_features: bool,
        drug_molecules: bool,
        labels: bool,
        context_feature_set: Optional[ContextFeatureSet],
        drug_feature_set: Optional[DrugFeatureSet],
        labeled_triples: Optional[LabeledTriples],
    ):
        """Initialize a batch generator.

        Args:
            batch_size: Number of drug pairs per batch.
            context_features: Indicator whether the batch should include biological context features.
            drug_features: Indicator whether the batch should include drug features.
            drug_molecules: Indicator whether the batch should include drug molecules
            labels: Indicator whether the batch should include drug pair labels.
            context_feature_set: A context feature set for feature generation.
            drug_feature_set: A drug feature set for feature generation.
            labeled_triples: A labeled triples object used to generate batches.
        """
        self.batch_size = batch_size
        self.context_features = context_features
        self.drug_features = drug_features
        self.drug_molecules = drug_molecules
        self.labels = labels
        self.context_feature_set = context_feature_set
        self.drug_feature_set = drug_feature_set
        self.labeled_triples = labeled_triples

    def _get_context_features(self, context_identifiers: Iterable[str]) -> Optional[torch.FloatTensor]:
        """Get the context features as a matrix.

        Args:
            context_identifiers (pd.Series): The context identifiers of interest.
        Returns:
            context_features (torch.FloatTensor): The matrix of biological context features.
        """
        if not self.context_features:
            return None
        return self.context_feature_set.get_feature_matrix(context_identifiers)

    def _get_drug_features(self, drug_identifiers: Iterable[str]) -> Optional[torch.FloatTensor]:
        """Get the global drug features as a matrix.

        Args:
            drug_identifiers: The drug identifiers of interest.
        Returns:
            drug_features: The matrix of drug features.
        """
        if not self.drug_features:
            return None
        return self.drug_feature_set.get_feature_matrix(drug_identifiers)

    def _get_drug_molecules(self, drug_identifiers: Iterable[str]) -> Optional[PackedGraph]:
        """Get the molecular structure of drugs.

        Args:
            drug_identifiers: The drug identifiers of interest.
        Returns:
            molecules: The molecules diagonally batched together for message passing.
        """
        if not self.drug_molecules:
            return None
        return self.drug_feature_set.get_molecules(drug_identifiers)

    def _transform_labels(self, labels: List):
        """Transform the labels from a chunk of the labeled triples frame.

        Args:
            labels (pd.Series): The drug pair binary labels.
        Returns:
            labels (torch.FloatTensor): The label target vector as a column vector.
        """
        if not self.labels:
            return None
        return torch.FloatTensor(np.array(labels).reshape(-1, 1))

    def generate_batch(self, batch_frame: pd.DataFrame) -> DrugPairBatch:
        """
        Generate a batch of drug features, molecules, context features and labels for a set of pairs.

        Args:
            batch_frame (pd.DataFrame): The labeled pairs of interest.
        Returns:
            batch (DrugPairBatch): A batch of tensors for the pairs.
        """
        drug_features_left = self._get_drug_features(batch_frame["drug_1"])
        drug_molecules_left = self._get_drug_molecules(batch_frame["drug_1"])

        drug_features_right = self._get_drug_features(batch_frame["drug_2"])
        drug_molecules_right = self._get_drug_molecules(batch_frame["drug_2"])

        context_features = self._get_context_features(batch_frame["context"])

        labels = self._transform_labels(batch_frame["label"])

        return DrugPairBatch(
            identifiers=batch_frame,
            drug_features_left=drug_features_left,
            drug_molecules_left=drug_molecules_left,
            drug_features_right=drug_features_right,
            drug_molecules_right=drug_molecules_right,
            context_features=context_features,
            labels=labels,
        )

    def __iter__(self) -> Iterator[DrugPairBatch]:
        """Iterate by first shuffling the triples and resetting the interator index."""
        self.labeled_triples.data = self.labeled_triples.data.sample(frac=1.0)
        self.sample_count = self.labeled_triples.data.shape[0]
        self.lower_frame_index = 0
        return self

    def __len__(self) -> int:
        """Get the maximal index of batches - this helps tools like tqdm."""
        return math.ceil(len(self.labeled_triples) / self.batch_size)

    def __next__(self) -> DrugPairBatch:
        """Get the next batch from the generator."""
        if self.lower_frame_index < self.sample_count:
            self.upper_frame_index = self.lower_frame_index + self.batch_size
            sub_frame = self.labeled_triples.data.iloc[
                self.lower_frame_index : min(self.upper_frame_index, self.sample_count)
            ]
            batch = self.generate_batch(sub_frame)
            self.lower_frame_index = self.upper_frame_index
            return batch
        else:
            self.frame_index = 0
            raise StopIteration
