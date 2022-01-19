"""A module for the drug pair batch class."""

from typing import Optional

import pandas as pd
import torch
from torchdrug.data import PackedGraph

__all__ = [
    "DrugPairBatch",
]


class DrugPairBatch:
    """A data class to store a labeled drug pair batch."""

    def __init__(
        self,
        identifiers: Optional[pd.DataFrame] = None,
        drug_features_left: Optional[torch.FloatTensor] = None,
        drug_molecules_left: Optional[PackedGraph] = None,
        drug_features_right: Optional[torch.FloatTensor] = None,
        drug_molecules_right: Optional[PackedGraph] = None,
        context_features: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.FloatTensor] = None,
    ):
        """Initialize the drug pair batch.

        Args:
            identifiers: A dataframe with drug pair, context and label columns.
            drug_features_left: A matrix of molecular features for the left hand drugs.
            drug_molecules_left: Packed molecules for the left hand drugs.
            drug_features_right: A matrix of molecular features for the right hand drugs.
            drug_molecules_right: Packed molecules for the right hand drugs.
            context_features: A matrix of biological/chemical context features.
            labels: A vector of drug pair labels.
        """
        self.identifiers = identifiers
        self.drug_features_left = drug_features_left
        self.drug_molecules_left = drug_molecules_left
        self.drug_features_right = drug_features_right
        self.drug_molecules_right = drug_molecules_right
        self.context_features = context_features
        self.labels = labels
