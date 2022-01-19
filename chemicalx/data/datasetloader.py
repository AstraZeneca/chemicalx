"""A module for dataset loaders."""

import io
import json
import urllib.request
from typing import Dict

import numpy as np
import pandas as pd

from .contextfeatureset import ContextFeatureSet
from .drugfeatureset import DrugFeatureSet
from .labeledtriples import LabeledTriples

__all__ = [
    "DatasetLoader",
    "DrugCombDB",
    "DrugComb",
    "TwoSides",
    "DrugbankDDI",
]


class DatasetLoader:
    """General dataset loader for the integrated drug pair scoring datasets."""

    def __init__(self, dataset_name: str):
        """Instantiate the dataset loader.

        Args:
            dataset_name (str): The name of the dataset.
        """
        self.base_url = "https://raw.githubusercontent.com/AstraZeneca/chemicalx/main/dataset"
        self.dataset_name = dataset_name
        assert dataset_name in ["drugcombdb", "drugcomb", "twosides", "drugbankddi"]

    def generate_path(self, file_name: str) -> str:
        """
        Generate a complete url for a dataset file.

        Args:
            file_name (str): Name of the data file.
        Returns:
            data_path (str): The complete url to the dataset.
        """
        data_path = "/".join([self.base_url, self.dataset_name, file_name])
        return data_path

    def load_raw_json_data(self, path: str) -> Dict:
        """
        Load a raw JSON dataset at the given path.

        Args:
            path (str): The path to the JSON file.
        Returns:
            raw_data (dict): A dictionary with the data.
        """
        with urllib.request.urlopen(path) as url:
            raw_data = json.loads(url.read().decode())
        return raw_data

    def load_raw_csv_data(self, path: str) -> pd.DataFrame:
        """
        Load a CSV dataset at the given path.

        Args:
            path (str): The path to the triples CSV file.
        Returns:
            raw_data (pd.DataFrame): A pandas DataFrame with the data.
        """
        data_bytes = urllib.request.urlopen(path).read()
        types = {"drug_1": str, "drug_2": str, "context": str, "label": float}
        raw_data = pd.read_csv(io.BytesIO(data_bytes), encoding="utf8", sep=",", dtype=types)
        return raw_data

    def get_context_features(self):
        """
        Get the context feature set.

        Returns:
            context_feature_set (ContextFeatureSet): The ContextFeatureSet of the dataset of interest.
        """
        path = self.generate_path("context_set.json")
        raw_data = self.load_raw_json_data(path)
        raw_data = {k: np.array(v).reshape(1, -1) for k, v in raw_data.items()}
        context_feature_set = ContextFeatureSet()
        context_feature_set.update(raw_data)
        return context_feature_set

    def get_drug_features(self):
        """
        Get the drug feature set.

        Returns:
            drug_feature_set (DrugFeatureSet): The DrugFeatureSet of the dataset of interest.
        """
        path = self.generate_path("drug_set.json")
        raw_data = self.load_raw_json_data(path)
        raw_data = {
            k: {"smiles": v["smiles"], "features": np.array(v["features"]).reshape(1, -1)} for k, v in raw_data.items()
        }
        drug_feature_set = DrugFeatureSet()
        drug_feature_set.update(raw_data)
        return drug_feature_set

    def get_labeled_triples(self):
        """
        Get the labeled triples file from the storage.

        Returns:
            labeled_triples (LabeledTriples): The labeled triples in the dataset.
        """
        path = self.generate_path("labeled_triples.csv")
        raw_data = self.load_raw_csv_data(path)
        labeled_triples = LabeledTriples()
        labeled_triples.update_from_pandas(raw_data)
        return labeled_triples


class DrugCombDB(DatasetLoader):
    """A dataset loader for `DrugCombDB <http://drugcombdb.denglab.org>`_."""

    def __init__(self):
        """Instantiate the DrugCombDB dataset loader."""
        super().__init__("drugcombdb")


class DrugComb(DatasetLoader):
    """A dataset loader for `DrugComb <https://drugcomb.fimm.fi/>`_."""

    def __init__(self):
        """Instantiate the DrugComb dataset loader."""
        super().__init__("drugcomb")


class TwoSides(DatasetLoader):
    """A dataset loader for a sample of `TWOSIDES <http://tatonettilab.org/offsides/>`_."""

    def __init__(self):
        """Instantiate the TWOSIDES dataset loader."""
        super().__init__("twosides")


class DrugbankDDI(DatasetLoader):
    """A dataset loader for `Drugbank DDI <https://www.pnas.org/content/115/18/E4304>`_."""

    def __init__(self):
        """Instantiate the Drugbank DDI dataset loader."""
        super().__init__("drugbankddi")
