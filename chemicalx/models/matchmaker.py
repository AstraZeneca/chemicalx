"""An implementation of the MatchMaker model."""

from .base import UnimplementedModel

from chemicalx.models import Model
from chemicalx.data import DrugPairBatch
import torch
import torch.nn.functional as F  # noqa:N812

__all__ = [
    "MatchMaker",
]


class MatchMaker(Model):
    """An implementation of the MatchMaker model.

    .. [matchmaker] `MatchMaker: A Deep Learning Framework for Drug Synergy Prediction
       <https://www.biorxiv.org/content/10.1101/2020.05.24.113241v3.full>`_
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
        self.encoder = torch.nn.Linear(drug_channels + context_channels, input_hidden_channels)
        self.hidden_first = torch.nn.Linear(input_hidden_channels, middle_hidden_channels)
        self.hidden_second = torch.nn.Linear(middle_hidden_channels, middle_hidden_channels)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.scoring_head_first = torch.nn.Linear(2 * middle_hidden_channels, final_hidden_channels)
        self.scoring_head_second = torch.nn.Linear(final_hidden_channels, out_channels)

    def unpack(self, batch: DrugPairBatch):
        """Return the context features, left drug features, and right drug features."""
        return (
            batch.context_features,
            batch.drug_features_left,
            batch.drug_features_right,
        )

    def _forward_hidden(self, tensor: torch.FloatTensor) -> torch.FloatTensor:
        hidden = self.encoder(tensor)
        hidden = F.relu(hidden)
        hidden = self.dropout(hidden)
        hidden = self.hidden_first(hidden)
        hidden = F.relu(hidden)
        hidden = self.dropout(hidden)
        hidden = self.hidden_second(hidden)
        return hidden

    def _forward_hidden_merged(self, tensor: torch.FloatTensor) -> torch.FloatTensor:
        hidden = self.scoring_head_first(tensor)
        hidden = F.relu(hidden)
        hidden = self.dropout(hidden)
        hidden = self.scoring_head_second(hidden)
        hidden = torch.sigmoid(hidden)
        return hidden

    def forward(
        self,
        context_features: torch.FloatTensor,
        drug_features_left: torch.FloatTensor,
        drug_features_right: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """
        Run a forward pass of the MatchMaker model model.

        Args:
            context_features (torch.FloatTensor): A matrix of biological context features.
            drug_features_left (torch.FloatTensor): A matrix of head drug features.
            drug_features_right (torch.FloatTensor): A matrix of tail drug features.
        Returns:
            hidden (torch.FloatTensor): A column vector of predicted synergy scores.
        """

        # The left drug
        hidden_left = torch.cat([context_features, drug_features_left], dim=1)
        hidden_left = self._forward_hidden(hidden_left)

        # The right drug
        hidden_right = torch.cat([context_features, drug_features_right], dim=1)
        hidden_right = self._forward_hidden(hidden_right)

        # Merged
        hidden_merged = torch.cat([hidden_left, hidden_right], dim=1)
        hidden_merged = self._forward_hidden_merged(hidden_merged)

        return hidden_merged
