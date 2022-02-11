"""A module for the drug feature set."""

from collections import UserDict
from typing import Dict, Iterable, Mapping, Union

import torch
from torchdrug.data import Molecule

from chemicalx.compat import Graph, PackedGraph

__all__ = [
    "DrugFeatureSet",
]


class DrugFeatureSet(UserDict, Mapping[str, Mapping[str, Union[torch.FloatTensor, Molecule]]]):
    """Drug feature set for compounds."""

    @classmethod
    def from_dict(cls, data: Dict[str, Dict]) -> "DrugFeatureSet":
        """Generate a drug feature set from a data dictionary."""
        return cls(
            {
                key: {
                    "features": torch.FloatTensor(features["features"]).view(1, -1),
                    "molecule": Molecule.from_smiles(features["smiles"]),
                }
                for key, features in data.items()
            }
        )

    def get_feature_matrix(self, drugs: Iterable[str]) -> torch.FloatTensor:
        """Get the drug feature matrix for a list of drugs.

        :param drugs: A list of drug identifiers.
        :returns: A matrix of drug features.
        """
        return torch.cat([self.data[drug]["features"] for drug in drugs])

    def get_molecules(self, drugs: Iterable[str]) -> PackedGraph:
        """Get the molecular structures.

        :param drugs: A list of drug identifiers.
        :returns: The molecules batched together for message passing.
        """
        return Graph.pack([self.data[drug]["molecule"] for drug in drugs])
