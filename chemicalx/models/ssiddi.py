"""An implementation of the SSI-DDI model."""

from .base import UnimplementedModel

__all__ = [
    "SSIDDI",
]


class SSIDDI(UnimplementedModel):
    """An implementation of the SSI-DDI model from [nyamabo2021]_.

    .. seealso:: This model was suggested in https://github.com/AstraZeneca/chemicalx/issues/11

    .. [nyamabo2021] Nyamabo, A. K., *et al.* (2021). `SSI–DDI: substructure–substructure interactions
       for drug–drug interaction prediction <https://doi.org/10.1093/bib/bbab133>`_.
       *Briefings in Bioinformatics*, 22(6).
    """
