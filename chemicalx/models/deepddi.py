"""An implementation of the DeepDDI model."""

import torch

from chemicalx.data import DrugPairBatch
from chemicalx.models import Model

__all__ = [
    "DeepDDI",
]


class DeepDDI(Model):
    """An implementation of the DeepDDI model.

    .. [DeepDDI] `Deep learning improves prediction of drug–drug and drug–food interactions
       <https://www.pnas.org/content/115/18/E4304>`_
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
        super(DeepDDI, self).__init__()
        assert hidden_layers_num > 1
        dnn = [
            torch.nn.Linear(drug_channels * 2, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(num_features=hidden_channels, affine=True, momentum=None),
            torch.nn.ReLU(),
        ]
        for _ in range(hidden_layers_num - 1):
            dnn.extend(
                [
                    torch.nn.Linear(hidden_channels, hidden_channels),
                    torch.nn.ReLU(),
                    torch.nn.BatchNorm1d(num_features=hidden_channels, affine=True, momentum=None),
                    torch.nn.ReLU(),
                ]
            )
        dnn.extend([torch.nn.Linear(hidden_channels, out_channels), torch.nn.Sigmoid()])
        self.dnn = torch.nn.Sequential(*dnn)

    def unpack(self, batch: DrugPairBatch):
        """Return the context features, left drug features and right drug features."""
        return (
            batch.drug_features_left,
            batch.drug_features_right,
        )

    def forward(
        self,
        drug_features_left: torch.FloatTensor,
        drug_features_right: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """
        Run a forward pass of the DeepDDI model.

        Args:
            drug_features_left (torch.FloatTensor): A matrix of head drug features.
            drug_features_right (torch.FloatTensor): A matrix of tail drug features.
        Returns:
            hidden (torch.FloatTensor): A column vector of predicted interaction scores.
        """
        input_feature = torch.cat([drug_features_left, drug_features_right], 1)
        hidden = self.dnn(input_feature)
        return hidden
