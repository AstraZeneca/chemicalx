"""A module for dataset loaders."""

import csv
import io
import json
import urllib.request
from abc import ABC, abstractmethod
from functools import lru_cache
from itertools import chain
from pathlib import Path
from textwrap import dedent
from typing import ClassVar, Dict, Mapping, Optional, Sequence, Tuple, cast

import numpy as np
import pandas as pd
import pystow as pystow
import torch

from .batchgenerator import BatchGenerator
from .contextfeatureset import ContextFeatureSet
from .drugfeatureset import DrugFeatureSet
from .labeledtriples import LabeledTriples
from .utils import get_features, get_tdc_synergy

__all__ = [
    "DatasetLoader",
    "RemoteDatasetLoader",
    "LocalDatasetLoader",
    # Actual datasets
    "DrugCombDB",
    "DrugComb",
    "TwoSides",
    "DrugbankDDI",
    "OncoPolyPharmacology",
]


class DatasetLoader(ABC):
    """A generic dataset."""

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

        :param batch_size: Number of drug pairs per batch.
        :param context_features: Indicator whether the batch should include biological context features.
        :param drug_features: Indicator whether the batch should include drug features.
        :param drug_molecules: Indicator whether the batch should include drug molecules
        :param labeled_triples:
            A labeled triples object used to generate batches. If none is given, will use
            all triples from the dataset.
        :returns: A batch generator
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

    @abstractmethod
    def get_context_features(self) -> ContextFeatureSet:
        """Get the context feature set."""

    @property
    def num_contexts(self) -> int:
        """Get the number of contexts."""
        return len(self.get_context_features())

    @property
    def context_channels(self) -> int:
        """Get the number of features for each context."""
        return next(iter(self.get_context_features().values())).shape[1]

    @abstractmethod
    def get_drug_features(self):
        """Get the drug feature set."""

    @property
    def num_drugs(self) -> int:
        """Get the number of drugs."""
        return len(self.get_drug_features())

    @property
    def drug_channels(self) -> int:
        """Get the number of features for each drug."""
        return next(iter(self.get_drug_features().values()))["features"].shape[1]

    @abstractmethod
    def get_labeled_triples(self) -> LabeledTriples:
        """Get the labeled triples file from the storage."""

    @property
    def num_labeled_triples(self) -> int:
        """Get the number of labeled triples."""
        return len(self.get_labeled_triples())

    def summarize(self) -> None:
        """Summarize the dataset."""
        print(
            dedent(
                f"""\
            Name: {self.__class__.__name__}
            Contexts: {self.num_contexts}
            Context Feature Size: {self.context_channels}
            Drugs: {self.num_drugs}
            Drug Feature Size: {self.drug_channels}
            Triples: {self.num_labeled_triples}
        """
            )
        )


class RemoteDatasetLoader(DatasetLoader):
    """A dataset loader for remote data."""

    def __init__(self, dataset_name: str):
        """Instantiate the dataset loader.

        :param dataset_name: The name of the dataset.
        """
        self.base_url = "https://raw.githubusercontent.com/AstraZeneca/chemicalx/main/dataset"
        self.dataset_name = dataset_name
        assert dataset_name in ["drugcombdb", "drugcomb", "twosides", "drugbankddi"]

    def generate_path(self, file_name: str) -> str:
        """Generate a complete url for a dataset file.

        :param file_name: Name of the data file.
        :returns: The complete url to the dataset.
        """
        data_path = "/".join([self.base_url, self.dataset_name, file_name])
        return data_path

    def load_raw_json_data(self, path: str) -> Dict:
        """Load a raw JSON dataset at the given path.

        :param path: The path to the JSON file.
        :returns: A dictionary with the data.
        """
        with urllib.request.urlopen(path) as url:
            raw_data = json.loads(url.read().decode())
        return raw_data

    def load_raw_csv_data(self, path: str) -> pd.DataFrame:
        """Load a CSV dataset at the given path.

        :param path: The path to the triples CSV file.
        :returns: A pandas DataFrame with the data.
        """
        data_bytes = urllib.request.urlopen(path).read()
        types = {"drug_1": str, "drug_2": str, "context": str, "label": float}
        raw_data = pd.read_csv(io.BytesIO(data_bytes), encoding="utf8", sep=",", dtype=types)
        return raw_data

    @lru_cache(maxsize=1)  # noqa: B019
    def get_context_features(self) -> ContextFeatureSet:
        """Get the context feature set."""
        path = self.generate_path("context_set.json")
        raw_data = self.load_raw_json_data(path)
        raw_data = {k: torch.FloatTensor(np.array(v).reshape(1, -1)) for k, v in raw_data.items()}
        return ContextFeatureSet(raw_data)

    @lru_cache(maxsize=1)  # noqa: B019
    def get_drug_features(self) -> DrugFeatureSet:
        """Get the drug feature set."""
        path = self.generate_path("drug_set.json")
        raw_data = self.load_raw_json_data(path)
        raw_data = {
            key: {"smiles": value["smiles"], "features": np.array(value["features"]).reshape(1, -1)}
            for key, value in raw_data.items()
        }
        return DrugFeatureSet.from_dict(raw_data)

    @lru_cache(maxsize=1)  # noqa: B019
    def get_labeled_triples(self) -> LabeledTriples:
        """Get the labeled triples file from the storage."""
        path = self.generate_path("labeled_triples.csv")
        df = self.load_raw_csv_data(path)
        return LabeledTriples(df)


class DrugCombDB(RemoteDatasetLoader):
    """A dataset loader for `DrugCombDB <http://drugcombdb.denglab.org>`_."""

    def __init__(self):
        """Instantiate the DrugCombDB dataset loader."""
        super().__init__("drugcombdb")


class DrugComb(RemoteDatasetLoader):
    """A dataset loader for `DrugComb <https://drugcomb.fimm.fi/>`_."""

    def __init__(self):
        """Instantiate the DrugComb dataset loader."""
        super().__init__("drugcomb")


class TwoSides(RemoteDatasetLoader):
    """A dataset loader for a sample of `TWOSIDES <http://tatonettilab.org/offsides/>`_."""

    def __init__(self):
        """Instantiate the TWOSIDES dataset loader."""
        super().__init__("twosides")


class DrugbankDDI(RemoteDatasetLoader):
    """A dataset loader for `Drugbank DDI <https://www.pnas.org/content/115/18/E4304>`_."""

    def __init__(self):
        """Instantiate the Drugbank DDI dataset loader."""
        super().__init__("drugbankddi")


class LocalDatasetLoader(DatasetLoader, ABC):
    """A dataset loader that processes and caches data locally."""

    structures_name: ClassVar[str] = "structures.tsv"
    features_name: ClassVar[str] = "features.tsv"
    contexts_name: ClassVar[str] = "contexts.tsv"
    labels_name: ClassVar[str] = "labels.tsv"

    def __init__(self, directory: Optional[Path] = None):
        """Instantiate the local dataset loader."""
        self.directory = directory or pystow.join("chemicalx", self.__class__.__name__.lower())
        self.drug_structures_path = self.directory.joinpath(self.structures_name)
        self.drug_features_path = self.directory.joinpath(self.features_name)
        self.contexts_path = self.directory.joinpath(self.contexts_name)
        self.labels_path = self.directory.joinpath(self.labels_name)

        if any(
            not path.exists()
            for path in (self.drug_features_path, self.drug_structures_path, self.contexts_path, self.labels_path)
        ):
            self.preprocess()

    @abstractmethod
    def preprocess(self):
        """Download and preprocess the dataset.

        The implementation of this function should write to all three of ``self.drugs_path``,
        ``self.contexts_path``, and ``self.labels_path`` using respectively :func:`write_drugs`,
        :func:`write_contexts`, and :func:`write_labels`.
        """

    @lru_cache(maxsize=1)  # noqa: B019
    def get_drug_features(self) -> DrugFeatureSet:
        """Get the drug feature set."""
        with self.drug_structures_path.open() as struct_file, self.drug_features_path.open() as feat_file:
            struct_reader = csv.reader(struct_file, delimiter="\t")
            feat_reader = csv.reader(feat_file, delimiter="\t")
            return DrugFeatureSet.from_dict(
                {
                    drug: {"smiles": smiles, "features": [float(f) for f in features]}
                    for (drug, smiles), (_, *features) in zip(struct_reader, feat_reader)
                }
            )

    def write_drugs(self, drugs: Mapping[str, str]) -> None:
        """Write the drug data."""
        with self.drug_structures_path.open("w") as struct_file, self.drug_features_path.open("w") as feat_file:
            for drug, smiles in sorted(drugs.items()):
                print(drug, smiles, sep="\t", file=struct_file)
                print(drug, *get_features(smiles), sep="\t", file=feat_file)

    @lru_cache(maxsize=1)  # noqa: B019
    def get_context_features(self) -> ContextFeatureSet:
        """Get the context feature set."""
        with self.contexts_path.open() as file:
            return ContextFeatureSet.from_dict(
                {key: [float(v) for v in values] for key, *values in csv.reader(file, delimiter="\t")}
            )

    def write_contexts(self, contexts: Mapping[str, Sequence[float]]):
        """Write the context feature set."""
        with self.contexts_path.open("w") as file:
            for key, values in contexts.items():
                print(key, *values, sep="\t", file=file)

    @lru_cache(maxsize=1)  # noqa: B019
    def get_labeled_triples(self) -> LabeledTriples:
        """Get the labeled triples dataframe."""
        return LabeledTriples(pd.read_csv(self.labels_path, sep="\t"))

    def write_labels(self, df: pd.DataFrame):
        """Write the labeled triples dataframe."""
        df.to_csv(self.labels_path, index=False, sep="\t")


class OncoPolyPharmacology(LocalDatasetLoader):
    """A large-scale oncology screen of drug-drug synergy from [oneil2016]_.

    .. [oneil2016] O’Neil, J., *et al.* (2016). `An Unbiased Oncology Compound Screen to Identify Novel
       Combination Strategies <https://doi.org/10.1158/1535-7163.MCT-15-0843>`_. *Molecular Cancer
       Therapeutics*, 15(6), 1155–1162.
    """

    def preprocess(self) -> None:
        """Download and process the OncoPolyPharmacology dataset."""
        tdc_directory = get_tdc_synergy("OncoPolyPharmacology")
        df = pd.read_pickle(tdc_directory.joinpath("oncopolypharmacology.pkl"))

        drugs = dict(
            chain(
                df[["Drug1_ID", "Drug1"]].values,
                df[["Drug2_ID", "Drug2"]].values,
            )
        )
        self.write_drugs(drugs)

        contexts = {key: values.round(4).tolist() for key, values in df[["Cell_Line_ID", "Cell_Line"]].values}
        self.write_contexts(contexts)

        labels_df = df[["Drug1_ID", "Drug2_ID", "Cell_Line_ID", "Y"]].rename(
            columns={
                "Drug1_ID": "drug_1",
                "Drug2_ID": "drug_2",
                "Cell_Line_ID": "context",
                "Y": "label",
            }
        )
        self.write_labels(labels_df)
