"""An implementation of the MHCADDI model."""

from .base import UnimplementedModel

__all__ = [
    "MHCADDI",
]


class MHCADDI(UnimplementedModel):
    """An implementation of the MHCADDI model from [deac2019]_.

    .. seealso:: This model was suggested in https://github.com/AstraZeneca/chemicalx/issues/13

    .. [deac2019] Deac, A., *et al.* (2019). `Drug-Drug Adverse Effect Prediction with
       Graph Co-Attention <http://arxiv.org/abs/1905.00534>`_. *arXiv*, 1905.00534.
    """
