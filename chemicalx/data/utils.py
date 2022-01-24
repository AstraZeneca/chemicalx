"""Dataset processing utilities."""

import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import pystow
import rdkit
from rdkit.Chem import AllChem, DataStructs
from tdc.multi_pred import DrugSyn

__all__ = [
    "get_tdc_synergy",
    "get_features",
    "write_drugs_json",
    "write_triples",
    "write_contexts_json",
]

DRUG_FILE_NAME = "drug_set.json"
CONTEXT_FILE_NAME = "context_set.json"
LABELS_FILE_NAME = "labeled_triples.tsv"


def get_tdc_synergy(name: str) -> Path:
    """Get the input and output directories for a given drug synergy dataset from Therapeutic Data Commons."""
    directory = pystow.join("tdc", name)
    DrugSyn(name=name, path=directory)
    return directory


def get_features(smiles: str):
    """Get a morgan fingerprint vector for the given molecule."""
    molecule = rdkit.Chem.MolFromSmiles(smiles)
    features = AllChem.GetHashedMorganFingerprint(molecule, 2, nBits=256)
    array = np.zeros((0,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(features, array)
    return array.tolist()


def write_drugs_json(drugs_raw: Mapping[str, str], output_directory: Path) -> Path:
    """Write drugs dictionary."""
    drug_set = {drug: {"smiles": smiles, "features": get_features(smiles)} for drug, smiles in drugs_raw.items()}
    path = output_directory.joinpath(DRUG_FILE_NAME)
    with path.open("w") as file:
        json.dump(drug_set, file)
    return path


def write_contexts_json(context_set: Mapping[str, Sequence[float]], output_directory: Path) -> Path:
    """Write contexts dictionary."""
    path = output_directory.joinpath(CONTEXT_FILE_NAME)
    with path.open("w") as file:
        json.dump(context_set, file)
    return path


def write_triples(df: pd.DataFrame, output_directory: Path, sep: str = "\t") -> Path:
    """Write labeled triples."""
    path = output_directory.joinpath(LABELS_FILE_NAME)
    df.to_csv(path, index=False, sep=sep)
    return path
