"""A module for the drug pair batch class."""

from dataclasses import dataclass
from typing import Optional

import pandas as pd
import torch
from torchdrug.data import PackedGraph

__all__ = [
    "DrugPairBatch",
]


@dataclass
class DrugPairBatch:
    """A data class to store a labeled drug pair batch."""

    #: A dataframe with drug pair, context and label columns.
    identifiers: Optional[pd.DataFrame]
    #: A matrix of molecular features for the left-hand drugs.
    drug_features_left: Optional[torch.FloatTensor]
    #: Packed molecules for the left-hand drugs.
    drug_molecules_left: Optional[PackedGraph]
    #: A matrix of molecular features for the right-hand drugs.
    drug_features_right: Optional[torch.FloatTensor]
    #: Packed molecules for the right-hand drugs.
    drug_molecules_right: Optional[PackedGraph]
    #: A matrix of biological/chemical context features.
    context_features: Optional[torch.FloatTensor]
    #: A vector of drug pair labels.
    labels: Optional[torch.FloatTensor]
