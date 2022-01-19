"""A module for dataset loaders."""

import io
import json
import urllib.request
from functools import lru_cache
from textwrap import dedent
from typing import Dict, Optional, Tuple, cast

import numpy as np
import pandas as pd
import torch

from .batchgenerator import BatchGenerator
from .contextfeatureset import ContextFeatureSet
from .drugfeatureset import DrugFeatureSet
from .labeledtriples import LabeledTriples

__all__ = [
    "DatasetLoader",
    # Actual datasets
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

    def get_generators(
        self,
        batch_size: int,
        context_features: bool,
        drug_features: bool,
        drug_molecules: bool,
        train_size: Optional[float] = None,
        random_state: Optional[int] = None,
    ) -> Tuple[BatchGenerator, BatchGenerator]:
        """Generate a pre-stratified pair of batch generators."""
        return cast(
            Tuple[BatchGenerator, BatchGenerator],
            tuple(
                self.get_generator(
                    batch_size=batch_size,
                    context_features=context_features,
                    drug_features=drug_features,
                    drug_molecules=drug_molecules,
                    labeled_triples=labeled_triples,
                )
                for labeled_triples in self.get_labeled_triples().train_test_split(
                    train_size=train_size,
                    random_state=random_state,
                )
            ),
        )

    def get_generator(
        self,
        batch_size: int,
        context_features: bool,
        drug_features: bool,
        drug_molecules: bool,
        labeled_triples: Optional[LabeledTriples] = None,
    ) -> BatchGenerator:
        """Initialize a batch generator.

        Args:
            batch_size: Number of drug pairs per batch.
            context_features: Indicator whether the batch should include biological context features.
            drug_features: Indicator whether the batch should include drug features.
            drug_molecules: Indicator whether the batch should include drug molecules
            labels: Indicator whether the batch should include drug pair labels.
            labeled_triples: A labeled triples object used to generate batches. If none is given, will use
                all triples from the dataset.
        """
        return BatchGenerator(
            batch_size=batch_size,
            context_features=context_features,
            drug_features=drug_features,
            drug_molecules=drug_molecules,
            context_feature_set=self.get_context_features() if context_features else None,
            drug_feature_set=self.get_drug_features() if drug_features else None,
            labeled_triples=self.get_labeled_triples() if labeled_triples is None else labeled_triples,
        )

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

    @lru_cache(maxsize=1)
    def get_context_features(self):
        """
        Get the context feature set.

        Returns:
            context_feature_set (ContextFeatureSet): The ContextFeatureSet of the dataset of interest.
        """
        path = self.generate_path("context_set.json")
        raw_data = self.load_raw_json_data(path)
        raw_data = {k: torch.FloatTensor(np.array(v).reshape(1, -1)) for k, v in raw_data.items()}
        return ContextFeatureSet(raw_data)

    @property
    def num_contexts(self) -> int:
        """Get the number of contexts."""
        return len(self.get_context_features())

    @property
    def context_channels(self) -> int:
        """Get the number of features for each context."""
        return next(iter(self.get_context_features().values())).shape[1]

    @lru_cache(maxsize=1)
    def get_drug_features(self):
        """
        Get the drug feature set.

        Returns:
            drug_feature_set (DrugFeatureSet): The DrugFeatureSet of the dataset of interest.
        """
        path = self.generate_path("drug_set.json")
        raw_data = self.load_raw_json_data(path)
        raw_data = {
            key: {"smiles": value["smiles"], "features": np.array(value["features"]).reshape(1, -1)}
            for key, value in raw_data.items()
        }
        return DrugFeatureSet.from_dict(raw_data)

    @property
    def num_drugs(self) -> int:
        """Get the number of drugs."""
        return len(self.get_drug_features())

    @property
    def drug_channels(self) -> int:
        """Get the number of features for each drug."""
        return next(iter(self.get_drug_features().values()))["features"].shape[1]

    @lru_cache(maxsize=1)
    def get_labeled_triples(self):
        """
        Get the labeled triples file from the storage.

        Returns:
            labeled_triples (LabeledTriples): The labeled triples in the dataset.
        """
        path = self.generate_path("labeled_triples.csv")
        df = self.load_raw_csv_data(path)
        return LabeledTriples(df)

    @property
    def num_labeled_triples(self) -> int:
        """Get the number of labeled triples."""
        return len(self.get_labeled_triples())

    def summarize(self) -> None:
        """Summarize the dataset."""
        print(
            dedent(
                f"""\
            Name: {self.dataset_name}
            Contexts: {self.num_contexts}
            Context Feature Size: {self.context_channels}
            Drugs: {self.num_drugs}
            Drug Feature Size: {self.drug_channels}
            Triples: {self.num_labeled_triples}
        """
            )
        )


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
