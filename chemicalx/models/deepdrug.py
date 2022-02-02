"""An implementation of the DeepDrug model."""

from .base import UnimplementedModel

__all__ = [
    "DeepDrug",
]


class DeepDrug(UnimplementedModel):
    """An implementation of the DeepDrug model from [cao2020]_.

    .. seealso:: This model was suggested in https://github.com/AstraZeneca/chemicalx/issues/14

    .. [cao2020] Cao, X., *et al.* (2020). `DeepDrug: A general graph-based deep learning framework
       for drug relation prediction <https://doi.org/10.1101/2020.11.09.375626>`_.
       *bioRxiv*, 2020.11.09.375626.
    """
