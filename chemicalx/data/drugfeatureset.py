import numpy as np
from typing import List, Dict, Union
from torchdrug.data import Molecule


class DrugFeatureSet(dict):
    """
    Drug feature set for compunds.
    """

    def __setitem__(self, drug: str, features: Dict[str, Union[str, np.ndarray]]):
        """Setting the features for a compound key

        Args:
            drug (str): Drug identifier.
            features (dict): Dictionary of smiles string and molecular features.
        """
        self.__dict__[drug] = {}
        self.__dict__[drug]["smiles"] = features["smiles"]
        self.__dict__[drug]["features"] = features["features"].reshape(1, -1)
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

    def update(self, data: Dict[str, Dict[str, Union[str, np.ndarray]]]):
        """Adding a dictionary of drug keys - feature dictionaries to a drug set.

        Args:
            data (dict): A dictionary of drug keys with feature dictionaries.
        Returns:
            DrugFeatureSet: The updated drug feature set.
        """
        return self.__dict__.update(
            {
                drug: {
                    "smiles": features["smiles"],
                    "features": features["features"].reshape(1, -1),
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

    def get_drug_feature_count(self) -> int:
        """Getting the number of drug feature dimensions.

        Returns:
            feature_count (int): The number of drug feature dimensions.
        """
        feature_count = 0
        if len(self.__dict__) > 0:
            drugs = list(self.keys())
            first_drug = drugs[0]
            feature_vector = self.__dict__[drugs[0]]["features"]
            feature_count = feature_vector.shape[1]
        return feature_count

    def get_feature_matrix(self, drugs: List[str]) -> np.ndarray:
        """Getting the drug feature matrix for a list of drugs.

        Args:
            drugs (list): A list of drug identifiers.
        Return:
            features (np.ndarray): A matrix of drug features.
        """
        features = [self.__dict__[drug]["features"] for drug in drugs]
        features = np.concatenate(features, axis=0)
        return features

    def get_smiles_strings(self, drugs: List[str]) -> List[str]:
        """Getting the list of drugs.

        Args:
            drugs (list): A list of drug identifiers.
        Return:
            smiles_strings (list): A list of smiles strings for the drugs.
        """
        smiles_strings = [self.__dict__[drug]["smiles"] for drug in drugs]
        return smiles_strings

    def get_molecules(self, drugs):
        """Getting the molecular structures.

        Args:
            drugs (list): A list of drug identifiers.
        Return:
            molecules (list): A list of drug molecules as torchdrug.Molecule objects.
        """
        molecules = [self.__dict__[drug]["molecule"] for drug in drugs]
        return molecules

    def get_feature_density_rate(self) -> float:
        """Getting the ratio of non-zero drug feature values in the drug feature matrix.

        Returns:
            float: The ratio of non-zero entries in the whole drug feature matrix.
        """
        feature_matrix_density = None
        if len(self.__dict__) > 0:
            all_drugs = list(self.keys())
            feature_matrix = self.get_feature_matrix(all_drugs)
            non_zero_count = np.sum(feature_matrix != 0)
            drug_count, feature_count = feature_matrix.shape
            feature_matrix_density = non_zero_count / (feature_count * drug_count)
        return feature_matrix_density
