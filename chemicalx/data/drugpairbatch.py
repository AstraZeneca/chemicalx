"""A module for the drug pair batch class."""

from dataclasses import dataclass
from typing import Optional

import pandas as pd
import torch
from torch.types import Device

from chemicalx.compat import PackedGraph

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

    def to(self, device: Device) -> "DrugPairBatch":
        """Move this batch to the given device (out of place)."""
        return DrugPairBatch(
            identifiers=self.identifiers,
            drug_features_left=_move_tensor(self.drug_features_left, device),
            drug_molecules_left=_move_graph(self.drug_molecules_left, device),
            drug_features_right=_move_tensor(self.drug_features_right, device),
            drug_molecules_right=_move_graph(self.drug_molecules_right, device),
            context_features=_move_tensor(self.context_features, device),
            labels=_move_tensor(self.labels, device),
        )


def _move_tensor(x: Optional[torch.FloatTensor], device: Device) -> Optional[torch.FloatTensor]:
    if x is None:
        return None
    return x.to(device)


def _move_graph(x: Optional[PackedGraph], device: Device) -> Optional[PackedGraph]:
    if x is None:
        return None
    return x.to(device)
