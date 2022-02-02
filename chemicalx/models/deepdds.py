"""An implementation of the DeepDDS model."""

from .base import UnimplementedModel

__all__ = [
    "DeepDDS",
]


class DeepDDS(UnimplementedModel):
    """An implementation of the DeepDDS model from [wang2021]_.

    .. seealso:: https://github.com/AstraZeneca/chemicalx/issues/19

    .. [wang2021] Wang, J., *et al.* (2021). `DeepDDS: deep graph neural network with attention
       mechanism to predict synergistic drug combinations <http://arxiv.org/abs/2107.02467>`_.
       *arXiv*, 2107.02467.
    """
