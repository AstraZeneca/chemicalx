"""An implementation of the DeepDDI model."""

import torch
from torch import nn

from chemicalx.data import DrugPairBatch
from chemicalx.models import Model

__all__ = [
    "DeepDDI",
]


class DeepDDI(Model):
    """An implementation of the DeepDDI model from [ryu2018]_.

    .. seealso:: This model was suggested in https://github.com/AstraZeneca/chemicalx/issues/2

    .. [ryu2018] Ryu, J. Y., *et al.* (2018). `Deep learning improves prediction
       of drug–drug and drug–food interactions <https://doi.org/10.1073/pnas.1803294115>`_.
       *Proceedings of the National Academy of Sciences*, 115(18), E4304–E4311.
    """

    def __init__(
        self,
        *,
        drug_channels: int,
        hidden_channels: int = 2048,
        hidden_layers_num: int = 9,
        out_channels: int = 1,
    ):
        """Instantiate the DeepDDI model.

        :param drug_channels: The number of drug features.
        :param hidden_channels: The number of hidden layer neurons.
        :param hidden_layers_num: The number of hidden layers.
        :param out_channels: The number of output channels.
        """
        super().__init__()
        assert hidden_layers_num > 1
        layers = [
            nn.Linear(drug_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=hidden_channels, affine=True, momentum=None),
            nn.ReLU(),
        ]
        for _ in range(hidden_layers_num - 1):
            layers.extend(
                [
                    nn.Linear(hidden_channels, hidden_channels),
                    nn.ReLU(),
                    nn.BatchNorm1d(num_features=hidden_channels, affine=True, momentum=None),
                    nn.ReLU(),
                ]
            )
        layers.extend([nn.Linear(hidden_channels, out_channels), nn.Sigmoid()])
        self.final = nn.Sequential(*layers)

    def unpack(self, batch: DrugPairBatch):
        """Return the context features, left drug features and right drug features."""
        return (
            batch.drug_features_left,
            batch.drug_features_right,
        )

    def _combine_sides(self, left: torch.FloatTensor, right: torch.FloatTensor) -> torch.FloatTensor:
        return torch.cat([left, right], dim=1)

    def forward(
        self,
        drug_features_left: torch.FloatTensor,
        drug_features_right: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """Run a forward pass of the DeepDDI model.

        :param drug_features_left: A matrix of head drug features.
        :param drug_features_right: A matrix of tail drug features.
        :returns: A column vector of predicted interaction scores.
        """
        hidden = self._combine_sides(drug_features_left, drug_features_right)
        return self.final(hidden)
