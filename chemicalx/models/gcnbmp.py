"""An implementation of the GCNBMP model."""

from .base import UnimplementedModel

__all__ = [
    "GCNBMP",
]


class GCNBMP(UnimplementedModel):
    """An implementation of the GCN-BMP model from [chen2020]_.

    .. seealso:: This model was suggested in https://github.com/AstraZeneca/chemicalx/issues/21

    .. [chen2020] Chen, X., *et al.* (2020). `GCN-BMP: Investigating graph representation learning
       for DDI prediction task <https://doi.org/10.1016/j.ymeth.2020.05.014>`_. *Methods*, 179, 47–54.
    """
