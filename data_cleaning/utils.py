"""Utilities for dataset pre-processing.

Requires ``pip install PyTDC pystow tabulate``.
"""

from pathlib import Path
from random import Random
from typing import List, Tuple

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


def get_samples(*, rng: Random, n: int, drugs, contexts, big_map):
    """Get a negative samples dataframe."""
    negative_samples = []
    for _ in trange(n, desc="Generating negative samples"):
        drug_1, drug_2 = rng.sample(drugs, 2)
        context = rng.choice(contexts)
        while (drug_1, drug_2, context) in big_map:
            drug_1, drug_2 = rng.sample(drugs, 2)
            context = rng.choice(contexts)
        negative_samples.append((drug_1, drug_2, context, 0.0))
    return pd.DataFrame(negative_samples, columns=COLUMNS)


def get_index(df: pd.DataFrame):
    """Index drugs' SMILES and drug-drug-context triples."""
    drugs_raw = {}
    big_map = {}
    for left_id, right_id, context, left_smiles, right_smiles in df.values:
        drugs_raw[left_id] = left_smiles
        drugs_raw[right_id] = right_smiles
        big_map[left_id, right_id, context] = 1
        big_map[right_id, left_id, context] = 1
    return drugs_raw, big_map
