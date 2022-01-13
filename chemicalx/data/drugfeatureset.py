import torch
import torchdrug
import numpy as np
from typing import List, Dict, Union
from torchdrug.data import Molecule, Graph, PackedGraph


class DrugFeatureSet(dict):
    """
    Drug feature set for compunds.
    """

    def __setitem__(self, drug: str, features: Dict[str, Union[str, torch.FloatTensor]]):
        """Setting the features for a compound key

        Args:
            drug (str): Drug identifier.
            features (dict): Dictionary of smiles string and molecular features.
        """
        self.__dict__[drug] = {}
        self.__dict__[drug]["features"] = torch.FloatTensor(features["features"])
        self.__dict__[drug]["molecule"] = Molecule.from_smiles(features["smiles"])

    def __getitem__(self, drug: str):
        """Getting the features for a drug key.

        Args:
            drug (str): Drug identifier.
        Returns:
            dict: The drug features corresponding to the key.
        """
        return self.__dict__[drug]

    def __len__(self):
        """Getting the number of drugs.

        Returns:
            int: The number of drugs.
        """
        return len(self.__dict__)

    def __delitem__(self, drug: str):
        """Deleting the features for a drug key.

        Args:
            drug (str): Drug identifier.
        """
        del self.__dict__[drug]

    def clear(self):
        """Deleting all of the drugs from the drug feature set.

        Returns:
            DrugFeatureSet: An empty drug feature set.
        """
        return self.__dict__.clear()

    def has_drug(self, drug: str):
        """Checking whether a drug feature set contains a drug.

        Args:
            drug (str): Drug identifier.
        Returns:
            bool: Boolean describing whether the drug is in the drug set.
        """
        return drug in self.__dict__

    def update(self, data: Dict[str, Dict]):
        """Adding a dictionary of drug keys - feature dictionaries to a drug set.

        Args:
            data (dict): A dictionary of drug keys with feature dictionaries.
        Returns:
            DrugFeatureSet: The updated drug feature set.
        """
        return self.__dict__.update(
            {
                drug: {
                    "features": torch.FloatTensor(features["features"]),
                    "molecule": Molecule.from_smiles(features["smiles"]),
                }
                for drug, features in data.items()
            }
        )

    def keys(self):
        """Retrieving the drugs in a feature set.

        Returns:
            list: An iterator of drug identifiers.
        """
        return self.__dict__.keys()

    def values(self):
        """Retrieving the iterator of drug features.

        Returns:
            list: Feature iterator.
        """
        return self.__dict__.values()

    def items(self):
        """Retrieving the iterator of tuples containing drug identifier - feature pairs.

        Returns:
            list: An iterator of (drug - feature dictionary) tuples.
        """
        return self.__dict__.items()

    def __contains__(self, drug: str):
        """A data class method which allows the use of the 'in' operator.

        Args:
            drug (str): A drug identifier.
        Returns:
            bool: An indicator whether the drug is in the drug feature set.
        """
        return drug in self.__dict__

    def __iter__(self):
        """A data class method which allows iteration over the drug feature set.

        Returns:
            iterable: An iterable of the drug feature set.
        """
        return iter(self.__dict__)

    def get_drug_count(self) -> int:
        """Getting the number of drugs.

        Returns:
            int: The number of drugs.
        """
        return len(self.__dict__)

    def get_feature_matrix(self, drugs: List[str]) -> torch.FloatTensor:
        """Getting the drug feature matrix for a list of drugs.

        Args:
            drugs (list): A list of drug identifiers.
        Return:
            features (torch.FloatTensor): A matrix of drug features.
        """
        features = torch.cat([self.__dict__[drug]["features"] for drug in drugs])
        return features

    def get_molecules(self, drugs: List[str]) -> PackedGraph:
        """Getting the molecular structures.

        Args:
            drugs (list): A list of drug identifiers.
        Return:
            molecules (torch.PackedGraph): The molecules batched together for message passing.
        """
        molecules = Graph.pack([self.__dict__[drug]["molecule"] for drug in drugs])
        return molecules
