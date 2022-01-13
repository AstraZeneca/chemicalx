import math
import torch
import pandas as pd
import numpy as np
from typing import List
from torchdrug.data import PackedGraph
from chemicalx.data import LabeledTriples, DrugFeatureSet, ContextFeatureSet, DrugPairBatch


class BatchGenerator:
    """
    Generator to create batches of drug pairs efficiently.

    Args:
        batch_size (int): Number of drug pairs per batch.
        context_features (bool): Indicator whether the batch should include biological context features.
        drug_features (bool): Indicator whether the batch should include drug features.
        drug_molecules (bool): Indicator whether the batch should include drug molecules
        labels (bool): Indicator whether the batch should include drug pair labels.
    """

    def __init__(
        self,
        batch_size: int,
        context_features: bool,
        drug_features: bool,
        drug_molecules: bool,
        labels: bool,
    ):
        self.batch_size = batch_size
        self.context_features = context_features
        self.drug_features = drug_features
        self.drug_molecules = drug_molecules
        self.labels = labels

    def set_context_feature_set(self, context_feature_set: None):
        """
        Method to set the context feature set.

        Args:
            context_feature_set (ContextFeatureSet): A context feature set for feature generation.
        """
        self.context_feature_set = context_feature_set

    def set_drug_feature_set(self, drug_feature_set: None):
        """
        Method to set the drug feature set.

        Args:
            drug_feature_set (DrugFeatureSet): A drug feature set for feature generation.
        """
        self.drug_feature_set = drug_feature_set

    def set_labeled_triples(self, labeled_triples: None):
        """
        Method to set the labeled triples.

        Args:
            labeled_triples (LabeledTriples): A labeled triples object used to generate batches.
        """
        self.labeled_triples = labeled_triples

    def set_data(self, context_feature_set: None, drug_feature_set: None, labeled_triples: None):
        """
        Method to set the feature sets and the labeled triples in one pass.

        Args:
            context_feature_set (ContextFeatureSet): A context feature set for feature generation.
            drug_feature_set (DrugFeatureSet): A drug feature set for feature generation.
            labeled_triples (LabeledTriples): A labeled triples object used to generate batches.
        """
        self.set_context_feature_set(context_feature_set)
        self.set_drug_feature_set(drug_feature_set)
        self.set_labeled_triples(labeled_triples)

    def _get_context_features(self, context_identifiers: List[str]):
        """
        Getting the context features as a matrix.

        Args:
            context_identifiers (pd.Series): The context identifiers of interest.
        Returns:
            context_features (torch.FloatTensor): The matrix of biological context features.
        """
        context_features = None
        if self.context_features:
            context_features = self.context_feature_set.get_feature_matrix(context_identifiers)
        return context_features

    def _get_drug_features(self, drug_identifiers: List[str]):
        """
        Getting the global drug features as a matrix.

        Args:
            drug_identifiers (pd.Series): The drug identifiers of interest.
        Returns:
            drug_features (torch.FloatTensor): The matrix of drug features.
        """
        drug_features = None
        if self.drug_features:
            drug_features = self.drug_feature_set.get_feature_matrix(drug_identifiers)
        return drug_features

    def _get_drug_molecules(self, drug_identifiers: pd.Series) -> PackedGraph:
        """
        Getting the molecular structure of drugs.

        Args:
            drug_identifiers (pd.Series): The drug identifiers of interest.
        Returns:
            molecules (torch.PackedGraph): The molecules diagonnaly batched together for message passing.
        """
        molecules = None
        if self.drug_molecules:
            molecules = self.drug_feature_set.get_molecules(drug_identifiers)
        return molecules

    def _transform_labels(self, labels: List):
        """
        Transforming the labels from a chunk of the labeled triples frame.

        Args:
            labels (pd.Series): The drug pair binary labels.
        Returns:
            labels (torch.FloatTensor): The label target vector as a column vector.
        """
        if self.labels:
            labels = torch.FloatTensor(np.array(labels).reshape(-1, 1))
        else:
            labels = None
        return labels

    def generate_batch(self, batch_frame: pd.DataFrame):
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

        batch = DrugPairBatch(
            batch_frame,
            drug_features_left,
            drug_molecules_left,
            drug_features_right,
            drug_molecules_right,
            context_features,
            labels,
        )

        return batch

    def __iter__(self):
        """
        Start of iteration method - shuffles the triples and resets the interator index.
        """
        self.labeled_triples.data = self.labeled_triples.data.sample(frac=1.0)
        self.sample_count = self.labeled_triples.data.shape[0]
        self.lower_frame_index = 0
        return self

    def __len__(self):
        """
        Method to set the maximal index of batches - helps tools like tqdm.
        """
        return math.ceil(self.labeled_triples.data.shape[0] / self.batch_size)

    def __next__(self):
        """
        Getting the next batch from the generator.
        """
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
