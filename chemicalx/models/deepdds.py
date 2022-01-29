"""An implementation of the DeepDDS model.

DeepDDS: deep graph neural network with attention mechanism to predict
synergistic drug combinations.

arXiv:2107.02467 [cs.LG]
https://arxiv.org/abs/2107.02467

SMILES strings transformed into a graph representation are used as input to
both the GAT and the GCN version of the model.

MLP is used to extract the feature embedding of gene expression profiles of
cancer cell line.

The embedding vector from both inputs are concatenated and fed into the
fully connected layers for binary classification of the drug combination as
synergistic or antagonistic.
"""
import torch
from torch.nn.functional import normalize, softmax
from torchdrug.data import PackedGraph
from torchdrug.layers import MLP, MaxReadout
from torchdrug.models import GraphAttentionNetwork, GraphConvolutionalNetwork

from chemicalx.constants import TORCHDRUG_NODE_FEATURES
from chemicalx.data import DrugPairBatch
from chemicalx.models import Model

__all__ = [
    "DeepDDS",
]


class DeepDDS(Model):
    """An implementation of the DeepDDS model from [wang2021]_.

    .. seealso:: This model was suggested in https://github.com/AstraZeneca/chemicalx/issues/19

    .. [wang2021] Wang, J., *et al.* (2021). `DeepDDS: deep graph neural network with attention
       mechanism to predict synergistic drug combinations <http://arxiv.org/abs/2107.02467>`_.
       *arXiv*, 2107.02467.
    """

    # todo: implement the GAT version of the model as well

    def __init__(
        self,
        *,
        context_feature_size: int,
        context_output_size: int,
        in_channels: int = TORCHDRUG_NODE_FEATURES,
        dropout: float = 0.2,  # Rate used in paper
    ):
        """Instantiate the DeepDDS model.

        :param context_feature_size:
        :param context_output_size:
        :param in_channels:
        :param dropout:
        """
        super(DeepDDS, self).__init__()
        self.cell_mlp = MLP(input_dim=context_feature_size, hidden_dims=[2048, 512, context_output_size])
        self.conv_left = GraphConvolutionalNetwork(
            input_dim=in_channels, hidden_dims=[in_channels, in_channels * 2, in_channels * 4]
        )
        self.conv_right = GraphConvolutionalNetwork(
            input_dim=in_channels, hidden_dims=[in_channels, in_channels * 2, in_channels * 4]
        )
        self.mlp_left = MLP(
            input_dim=in_channels * 4,
            hidden_dims=[in_channels * 2, context_output_size],
            dropout=dropout,
        )
        self.mlp_right = MLP(
            input_dim=in_channels * 4,
            hidden_dims=[in_channels * 2, context_output_size],
            dropout=dropout,
        )
        self.mlp_final = MLP(input_dim=context_output_size * 3, hidden_dims=[1024, 512, 128, 1], dropout=dropout)
        self.max_readout = MaxReadout()

    def unpack(self, batch: DrugPairBatch):
        """Return the context features, left drug features and right drug features."""
        return batch.context_features, batch.drug_molecules_left, batch.drug_molecules_right

    def forward(
        self, context_features: torch.FloatTensor, molecules_left: PackedGraph, molecules_right: PackedGraph
    ) -> torch.FloatTensor:
        """Run a forward pass of the DeeDDS model.

        Args:
            context_features (torch.FloatTensor): A matrix of cell line features
            molecules_left (torch.FloatTensor): A matrix of left drug features
            molecules_right (torch.FloatTensor): A matrix of right drug features
        Returns:
            (torch.FloatTensor): A column vector of predicted synergy scores
        """
        # Run the MLP forward for the cell line features
        mlp_out = self.cell_mlp(normalize(context_features, p=2, dim=1))

        # Todo: source code does *not* run a final activation before
        #  concatenating; torch does this automatically
        features_left = self.conv_left(molecules_left, molecules_left.data_dict["node_feature"])["node_feature"]
        features_left = self.mlp_left(features_left)
        features_left = self.max_readout.forward(input=features_left, graph=molecules_left)
        features_right = self.conv_right(molecules_right, molecules_right.data_dict["node_feature"])["node_feature"]
        features_right = self.mlp_right(features_right)
        features_right = self.max_readout.forward(input=features_right, graph=molecules_right)

        # Concatenate the output of the MLP and the GNN
        concat_in = torch.cat([mlp_out, features_left, features_right], dim=1)

        # Run the fully connected layers forward
        out_mlp = self.mlp_final(concat_in)

        # Apply the softmax function to the output
        return softmax(out_mlp, dim=1)
