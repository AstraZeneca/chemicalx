"""An implementation of the MatchMaker model."""

import torch
from torch import nn

from chemicalx.data import DrugPairBatch
from chemicalx.models import Model

__all__ = [
    "MatchMaker",
]


class MatchMaker(Model):
    """An implementation of the MatchMaker model from [kuru2021]_.

    .. seealso:: This model was suggested in https://github.com/AstraZeneca/chemicalx/issues/23

    .. [kuru2021] Kuru, H. I., *et al.* (2021). `MatchMaker: A Deep Learning Framework
       for Drug Synergy Prediction <https://doi.org/10.1109/TCBB.2021.3086702>`_.
       *IEEE/ACM Transactions on Computational Biology and Bioinformatics*, 1–1.
    """

    def __init__(
        self,
        *,
        context_channels: int,
        drug_channels: int,
        input_hidden_channels: int = 32,
        middle_hidden_channels: int = 32,
        final_hidden_channels: int = 32,
        out_channels: int = 1,
        dropout_rate: float = 0.5,
    ):
        """Instantiate the MatchMaker model.

        :param context_channels: The number of context features.
        :param drug_channels: The number of drug features.
        :param input_hidden_channels: The number of hidden layer neurons in the input layer.
        :param middle_hidden_channels: The number of hidden layer neurons in the middle layer.
        :param final_hidden_channels: The number of hidden layer neurons in the final layer.
        :param out_channels: The number of output channels.
        :param dropout_rate: The rate of dropout before the scoring head is used.
        """
        super().__init__()
        #: Applied to the left+context and right+context separately
        self.drug_context_layer = nn.Sequential(
            nn.Linear(drug_channels + context_channels, input_hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(input_hidden_channels, middle_hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(middle_hidden_channels, middle_hidden_channels),
        )
        # Applied to the concatenated left/right tensors
        self.final = nn.Sequential(
            nn.Linear(2 * middle_hidden_channels, final_hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(final_hidden_channels, out_channels),
            nn.Sigmoid(),
        )

    def unpack(self, batch: DrugPairBatch):
        """Return the context features, left drug features, and right drug features."""
        return (
            batch.context_features,
            batch.drug_features_left,
            batch.drug_features_right,
        )

    def _combine_sides(self, left: torch.FloatTensor, right: torch.FloatTensor) -> torch.FloatTensor:
        return torch.cat([left, right], dim=1)

    def forward(
        self,
        context_features: torch.FloatTensor,
        drug_features_left: torch.FloatTensor,
        drug_features_right: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """Run a forward pass of the MatchMaker model.

        :param context_features: A matrix of biological context features.
        :param drug_features_left: A matrix of head drug features.
        :param drug_features_right: A matrix of tail drug features.
        :returns: A column vector of predicted synergy scores.
        """
        # The left drug
        hidden_left = torch.cat([context_features, drug_features_left], dim=1)
        hidden_left = self.drug_context_layer(hidden_left)

        # The right drug
        hidden_right = torch.cat([context_features, drug_features_right], dim=1)
        hidden_right = self.drug_context_layer(hidden_right)

        hidden = self._combine_sides(hidden_left, hidden_right)
        return self.final(hidden)
