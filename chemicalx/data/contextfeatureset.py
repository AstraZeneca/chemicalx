import torch
import numpy as np
from typing import Dict


class ContextFeatureSet(dict):
    def __setitem__(self, context: str, features: np.ndarray):
        self.__dict__[context] = torch.FloatTensor(features)

    def __getitem__(self, context: str):
        return self.__dict__[context]

    def __repr__(self):
        return repr(self.__dict__)

    def __len__(self):
        return len(self.__dict__)

    def __delitem__(self, context: str):
        del self.__dict__[context]

    def clear(self):
        return self.__dict__.clear()

    def copy(self):
        return self.__dict__.copy()

    def has_context(self, context: str):
        return context in self.__dict__

    def update(self, data: Dict[str, np.ndarray]):
        return self.__dict__.update({context: torch.FloatTensor(features) for context, features in data.items()})

    def contexts(self):
        return self.__dict__.keys()

    def features(self):
        return self.__dict__.values()

    def context_features(self):
        return self.__dict__.items()

    def pop(self, *args):
        return self.__dict__.pop(*args)

    def __cmp__(self, dict_):
        return self.__cmp__(self.__dict__, dict_)

    def __contains__(self, context: str):
        return context in self.__dict__

    def __iter__(self):
        return iter(self.__dict__)

    def __unicode__(self):
        return unicode(repr(self.__dict__))

    def get_context_count(self) -> int:
        return len(self.__dict__)

    def get_context_feature_count(self) -> int:
        pass

    def get_feature_density_rate(self) -> float:
        pass
