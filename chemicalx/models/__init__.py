"""Models for ChemicalX."""

from class_resolver import Resolver

from .base import UnimplementedModel
from .caster import CASTER
from .deepcci import DeepCCI
from .deepddi import DeepDDI
from .deepdds import DeepDDS
from .deepdrug import DeepDrug
from .deepsynergy import DeepSynergy
from .dpddi import DPDDI
from .epgcnds import EPGCNDS
from .gcnbmp import GCNBMP
from .matchmaker import MatchMaker
from .mhcaddi import MHCADDI
from .mrgnn import MRGNN
from .ssiddi import SSIDDI

__all__ = [
    "model_resolver",
    # Base models
    "UnimplementedModel",
    # Implementations
    "AUDNNSynergy",
    "CASTER",
    "DeepCCI",
    "DeepDDI",
    "DeepDDS",
    "DeepDrug",
    "DeepSynergy",
    "DPDDI",
    "EPGCNDS",
    "GCNBMP",
    "MatchMaker",
    "MHCADDI",
    "MRGNN",
    "SSIDDI",
]

model_resolver = Resolver.from_subclasses(base=UnimplementedModel)
