r"""An implementation of the DeepSynergy model."""

import torch
import torch.nn.functional as F  # noqa:N812

from chemicalx.data import DrugPairBatch
from chemicalx.models import Model

__all__ = [
    "DeepSynergy",
]


class DeepSynergy(Model):
    r"""The DeepSynergy model from [deepsynergy]_.

    .. [deepsynergy] `DeepSynergy: Predicting Anti-Cancer Drug Synergy with Deep Learning
       <https://academic.oup.com/bioinformatics/article/34/9/1538/4747884>`_
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
        """Instantiate the DeepSynergy model.

        :param context_channels: The number of context features.
        :param drug_channels: The number of drug features.
        :param input_hidden_channels: The number of hidden layer neurons in the input layer.
        :param middle_hidden_channels: The number of hidden layer neurons in the middle layer.
        :param final_hidden_channels: The number of hidden layer neurons in the final layer.
        :param out_channels: The number of output channels.
        :param dropout_rate: The rate of dropout before the scoring head is used.
        """
        super(DeepSynergy, self).__init__()
        self.encoder = torch.nn.Linear(drug_channels + drug_channels + context_channels, input_hidden_channels)
        self.hidden_first = torch.nn.Linear(input_hidden_channels, middle_hidden_channels)
        self.hidden_second = torch.nn.Linear(middle_hidden_channels, final_hidden_channels)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.scoring_head = torch.nn.Linear(final_hidden_channels, out_channels)

    def unpack(self, batch: DrugPairBatch):
        """Return the context features, left drug features, and right drug features."""
        return (
            batch.context_features,
            batch.drug_features_left,
            batch.drug_features_right,
        )

    def forward(
        self,
        context_features: torch.FloatTensor,
        drug_features_left: torch.FloatTensor,
        drug_features_right: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """
        Run a forward pass of the DeepSynergy model.

        Args:
            context_features (torch.FloatTensor): A matrix of biological context features.
            drug_features_left (torch.FloatTensor): A matrix of head drug features.
            drug_features_right (torch.FloatTensor): A matrix of tail drug features.
        Returns:
            hidden (torch.FloatTensor): A column vector of predicted synergy scores.
        """
        hidden = torch.cat([context_features, drug_features_left, drug_features_right], dim=1)
        hidden = self.encoder(hidden)
        hidden = F.relu(hidden)
        hidden = self.hidden_first(hidden)
        hidden = F.relu(hidden)
        hidden = self.hidden_second(hidden)
        hidden = F.relu(hidden)
        hidden = self.dropout(hidden)
        hidden = self.scoring_head(hidden)
        hidden = torch.sigmoid(hidden)
        return hidden
