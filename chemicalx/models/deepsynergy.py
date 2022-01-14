import torch
import torch.nn.functional as F


class DeepSynergy(torch.nn.Module):
    r"""The DeepSynergy model from the `"DeepSynergy: Predicting
    Anti-Cancer Drug Synergy with Deep Learning"
    <https://academic.oup.com/bioinformatics/article/34/9/1538/4747884>`_ paper.

    Args:
        context_channels (int): The number of context features.
        drug_channels (int): The number of drug features.
        input_hidden_channels (int): The number of hidden layer neurons in the input layer.
        middle_hidden_channels (int): The number of hidden layer neurons in the middle layer.
        final_hidden_channels (int): The number of hidden layer neurons in the final layer.
        dropout_rate (float): The rate of dropout before the scoring head is used.
    """

    def __init__(
        self,
        context_channels: int,
        drug_channels: int,
        input_hidden_channels: int = 32,
        middle_hidden_channels: int = 32,
        final_hidden_channels: int = 32,
        dropout_rate: float = 0.5,
    ):
        super(DeepSynergy, self).__init__()
        self.encoder = torch.nn.Linear(drug_channels + drug_channels + context_channels, input_hidden_channels)
        self.hidden_first = torch.nn.Linear(input_hidden_channels, middle_hidden_channels)
        self.hidden_second = torch.nn.Linear(middle_hidden_channels, final_hidden_channels)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.scoring_head = torch.nn.Linear(final_hidden_channels, 1)

    def forward(
        self,
        context_features: torch.FloatTensor,
        drug_features_left: torch.FloatTensor,
        drug_features_right: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """
        A forward pass of the DeepSynergy model.

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
