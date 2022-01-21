"""Utilities for dataset pre-processing.

Requires ``pip install PyTDC pystow tabulate``.
"""

import json
from pathlib import Path
from random import Random
from typing import Collection, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import pystow
import rdkit
from rdkit.Chem import AllChem, DataStructs
from tdc.multi_pred import DDI
from tqdm import trange

__all__ = [
    "OUTPUT",
    "get_tdc",
    "get_features",
    "map_context",
    "get_samples",
    "get_index",
]

HERE = Path(__file__).parent.resolve()
OUTPUT = HERE.parent.joinpath("dataset")

COLUMNS = ["drug_1", "drug_2", "context", "label"]


def get_tdc(name: str, out_name: str) -> Tuple[Path, Path]:
    """Get the input and output directories for a given dataset from Therapeutic Data Commons."""
    directory = pystow.join("tdc", name)
    DDI(name=name, path=directory)
    od = OUTPUT.joinpath(out_name)
    od.mkdir(exist_ok=True, parents=True)
    return directory, od


def get_features(smiles: str):
    """Get a morgan fingerprint vector for the given molecule."""
    molecule = rdkit.Chem.MolFromSmiles(smiles)
    features = AllChem.GetHashedMorganFingerprint(molecule, 2, nBits=256)
    array = np.zeros((0,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(features, array)
    return array.tolist()


def map_context(index: int, context_count: int) -> List[int]:
    """Get a one-hot encoding for the given context."""
    context_vector = [0 for _ in range(context_count)]
    context_vector[index] = 1
    return context_vector


def get_samples(
    *, rng: Random, n: int, drugs: Sequence[str], contexts: Sequence[str], big_map: Collection[Tuple[str, str, str]]
):
    """Get a negative samples dataframe."""
    negative_samples = []
    for _ in trange(n, desc="Generating negative samples", unit_scale=True):
        drug_1, drug_2 = rng.sample(drugs, 2)
        context = rng.choice(contexts)
        while (drug_1, drug_2, context) in big_map:
            drug_1, drug_2 = rng.sample(drugs, 2)
            context = rng.choice(contexts)
        negative_samples.append((drug_1, drug_2, context, 0.0))
    return pd.DataFrame(negative_samples, columns=COLUMNS)


def get_index(df: pd.DataFrame) -> Tuple[Dict[str, str], Dict[Tuple[str, str, str], int]]:
    """Index drugs' SMILES and drug-drug-context triples."""
    drugs_raw = {}
    big_map = {}
    for left_id, right_id, context, left_smiles, right_smiles in df.values:
        drugs_raw[left_id] = left_smiles
        drugs_raw[right_id] = right_smiles
        big_map[left_id, right_id, context] = 1
        big_map[right_id, left_id, context] = 1
    return drugs_raw, big_map


def write_artifacts(output_directory: Path, drugs_raw: Mapping[str, str], contexts: Sequence[str]) -> None:
    """Write drugs and one-hot contexts files."""
    # Generate drugs file
    drug_set = {drug: {"smiles": smiles, "features": get_features(smiles)} for drug, smiles in drugs_raw.items()}
    with output_directory.joinpath("drug_set.json").open("w") as file:
        json.dump(drug_set, file)
    # with output_directory.joinpath("structures.tsv").open("w") as structures_file, output_directory.joinpath(
    #     "features.tsv"
    # ).open("w") as features_file:
    #     for drug, d in sorted(drug_set.items()):
    #         print(drug, d["smiles"], sep="\t", file=structures_file)
    #         print(drug, *d["features"], sep="\t", file=features_file)

    # Generate contexts file
    context_count = len(contexts)
    context_set = {context: map_context(i, context_count) for i, context in enumerate(contexts)}
    with output_directory.joinpath("context_set.json").open("w") as file:
        json.dump(context_set, file)
    # with output_directory.joinpath("contexts.tsv").open("w") as file:
    #     for context, v in sorted(context_set.items()):
    #         print(context, *v, sep="\t", file=file)
