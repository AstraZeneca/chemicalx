"""An implementation of the MR-GNN model."""

from .base import UnimplementedModel

__all__ = [
    "MRGNN",
]


class MRGNN(UnimplementedModel):
    """An implementation of the MR-GNN model from [xu2019]_.

    .. seealso:: This model was suggested in https://github.com/AstraZeneca/chemicalx/issues/12

    .. [xu2019] Xu, N., *et al.* (2019). `MR-GNN: Multi-resolution and dual graph neural network for
       predicting structured entity interactions <https://doi.org/10.24963/ijcai.2019/551>`_.
       *IJCAI International Joint Conference on Artificial Intelligence*, 2019, 3968–3974.
    """
