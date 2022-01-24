"""Dataset processing utilities."""

import json
from pathlib import Path
from typing import List, Mapping

import numpy as np
import pandas as pd
import pystow
import rdkit
from rdkit.Chem import AllChem, DataStructs
from tdc.multi_pred import DrugSyn

__all__ = [
    "get_tdc_synergy",
    "get_features",
    "write_drugs",
    "write_triples",
    "write_contexts",
]


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


def write_drugs(drugs_raw: Mapping[str, str], output_directory: Path) -> None:
    """Write drugs dictionary."""
    drug_set = {drug: {"smiles": smiles, "features": get_features(smiles)} for drug, smiles in drugs_raw.items()}
    with output_directory.joinpath("drug_set.json").open("w") as file:
        json.dump(drug_set, file)


def write_contexts(context_set: Mapping[str, List[str]], output_directory: Path) -> None:
    """Write contexts dictionary."""
    with output_directory.joinpath("context_set.json").open("w") as file:
        json.dump(context_set, file)


def write_triples(df: pd.DataFrame, output_directory: Path) -> None:
    """Write labeled triples."""
    df.to_csv(output_directory.joinpath("labeled_triples.csv"), index=False)
