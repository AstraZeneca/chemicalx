"""Datasets and utilities."""

from class_resolver import Resolver

from .batchgenerator import *  # noqa:F401,F403
from .batchgenerator import BatchGenerator
from .contextfeatureset import *  # noqa:F401,F403
from .contextfeatureset import ContextFeatureSet
from .datasetloader import DrugCombDB  # noqa:F401,F403
from .datasetloader import DatasetLoader, DrugComb
from .drugfeatureset import *  # noqa:F401,F403
from .drugfeatureset import DrugFeatureSet
from .drugpairbatch import *  # noqa:F401,F403
from .drugpairbatch import DrugPairBatch
from .labeledtriples import *  # noqa:F401,F403
from .labeledtriples import LabeledTriples

__all__ = [
    "BatchGenerator",
    "ContextFeatureSet",
    "DrugFeatureSet",
    "DrugPairBatch",
    "LabeledTriples",
    # Datasets
    "dataset_resolver",
    "DatasetLoader",
    "DrugCombDatasetLoader",
    "DrugCombDbDatasetLoader",
]

dataset_resolver = Resolver.from_subclasses(base=DatasetLoader)
