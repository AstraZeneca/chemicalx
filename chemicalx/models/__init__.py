"""Models for ChemicalX."""

from class_resolver import Resolver

from .base import Model, UnimplementedModel
from .caster import CASTER
from .deepddi import DeepDDI
from .deepdds import DeepDDS
from .deepdrug import DeepDrug
from .deepsynergy import DeepSynergy
from .epgcnds import EPGCNDS
from .gcnbmp import GCNBMP
from .matchmaker import MatchMaker
from .mhcaddi import MHCADDI
from .mrgnn import MRGNN
from .ssiddi import SSIDDI

__all__ = [
    "model_resolver",
    # Base models
    "Model",
    "UnimplementedModel",
    # Implementations
    "CASTER",
    "DeepDDI",
    "DeepDDS",
    "DeepDrug",
    "DeepSynergy",
    "EPGCNDS",
    "GCNBMP",
    "MatchMaker",
    "MHCADDI",
    "MRGNN",
    "SSIDDI",
]

model_resolver = Resolver.from_subclasses(base=Model)
