"""Models for ChemicalX."""

from class_resolver import Resolver  # noqa:F401,F403

from .audnnsynergy import *  # noqa:F401,F403
from .base import (  # noqa:F401,F403; noqa:F401,F403
    ContextlessModel,
    ContextModel,
    Model,
)
from .caster import *  # noqa:F401,F403
from .deepcci import *  # noqa:F401,F403
from .deepddi import *  # noqa:F401,F403
from .deepdds import *  # noqa:F401,F403
from .deepdrug import *  # noqa:F401,F403
from .deepsynergy import *  # noqa:F401,F403
from .dpddi import *  # noqa:F401,F403
from .epgcnds import *  # noqa:F401,F403
from .gcnbmp import *  # noqa:F401,F403
from .matchmaker import *  # noqa:F401,F403
from .mhcaddi import *  # noqa:F401,F403
from .mrgnn import *  # noqa:F401,F403
from .ssiddi import *  # noqa:F401,F403

model_resolver = Resolver.from_subclasses(
    base=Model,
    # Skip base classes
    skip={ContextModel, ContextlessModel},
)
