"""An implementation of the CASTER model."""

from .base import UnimplementedModel

__all__ = [
    "CASTER",
]


class CASTER(UnimplementedModel):
    """An implementation of the CASTER model from [huang2020]_.

    .. seealso:: https://github.com/AstraZeneca/chemicalx/issues/15

    .. [huang2020] Huang, K., *et al.* (2020). `CASTER: Predicting drug interactions
       with chemical substructure representation <https://doi.org/10.1609/aaai.v34i01.5412>`_.
       *AAAI 2020 - 34th AAAI Conference on Artificial Intelligence*, 702–709.
    """
