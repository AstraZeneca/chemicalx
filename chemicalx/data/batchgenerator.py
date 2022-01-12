import math
import torch
from typing import List
from torchdrug.data import Graph
from chemicalx.data import LabeledTriples, DrugFeatureSet, ContextFeatureSet, DrugPairBatch


class BatchGenerator:
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

    def set_data(self, contex_feature_set: None, drug_feature_set: None, labeled_triples: None):

        self.context_feature_set = context_feature_set
        self.drug_feature_set = drug_feature_set
        self.labeled_triples = labeled_triples

    def _get_context_features(self, context_identifiers: List):
        context_features = None
        if self.context_features:
            context_features = self.context_feature_set.get_feature_matrix(context_identifiers)
            context_features = torch.FloatTensor(context_features)
        return context_features

    def _get_drug_features(self, drug_identifiers: List):
        drug_features = None
        if self.drug_features:
            drug_features = self.drug_feature_set.get_feature_matrix(drug_identifiers)
            drug_features = torch.FloatTensor(drug_features)
        return drug_features

    def _get_drug_molecules(self, drug_identifiers: List):
        molecules = None
        if self.drug_molecules:
            molecules = self.drug_feature_set.get_molecules(drug_identifiers)
            molecules = Graph.pack(molecules)
        return molecules

    def _get_labels(self, labels: List):
        if self.labels:
            labels = torch.FloatTensor(np.array(labels))
        else:
            labels = None
        return labels

    def generate_batch(self, batch_frame: pd.DataFrame):

        drug_features_left = self._get_drug_features(batch_frame["drug_1"])
        drug_molecules_left = self._get_drug_molecules(batch_frame["drug_1"])

        drug_features_right = self._get_drug_features(batch_frame["drug_2"])
        drug_molecules_right = self._get_drug_molecules(batch_frame["drug_2"])

        context_features = self._get_context_features(batch_frame["context"])

        labels = self._get_labels(batch_frame["label"])

        batch = DrugPairBatch(
            drug_features_left, drug_molecules_left, drug_features_right, drug_molecules_right, context_features, labels
        )

        return batch

    def __iter__(self):
        self.labeled_triples.data = self.labeled_triples.data.sample(frac=1.0)
        self.sample_count = self.labeled_triples.data.shape[0]
        self.lower_frame_index = 0
        return self

    def __len__(self):
        return math.ceil(self.labeled_triples.data.shape[0] / self.batch_size)

    def __next__(self):
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
