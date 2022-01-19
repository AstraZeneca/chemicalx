"""Constants for ChemicalX."""

from torchdrug.data import Molecule
from torchdrug.data.feature import atom_default

__all__ = [
    "TORCHDRUG_NODE_FEATURES",
]

#: The default number of node features on a molecule in torchdrug
TORCHDRUG_NODE_FEATURES = len(atom_default(Molecule.dummy_atom))
