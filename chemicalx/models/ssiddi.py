"""An implementation of the SSI-DDI model."""

import torch
import torch.nn.functional
from torch import nn
from torch.nn import LayerNorm
from torch.nn.modules.container import ModuleList
from torchdrug.layers import GraphAttentionConv, MeanReadout

from chemicalx.compat import PackedGraph
from chemicalx.constants import TORCHDRUG_NODE_FEATURES
from chemicalx.data import DrugPairBatch
from chemicalx.models import Model

__all__ = [
    "SSIDDI",
]


class EmbeddingLayer(torch.nn.Module):
    """Attention layer."""

    def __init__(self, feature_number: int):
        """Initialize the relational embedding layer.

        :param feature_number: Number of features.
        """
        super().__init__()
        self.weights = torch.nn.Parameter(torch.zeros(feature_number, feature_number))
        torch.nn.init.xavier_uniform_(self.weights)

    def forward(
        self,
        left_representations: torch.FloatTensor,
        right_representations: torch.FloatTensor,
        alpha_scores: torch.FloatTensor,
    ):
        """
        Make a forward pass with the drug representations.

        :param left_representations: Left side drug representations.
        :param right_representations: Right side drug representations.
        :param alpha_scores: Attention scores.
        :returns: Positive label scores vector.
        """
        attention = torch.nn.functional.normalize(self.weights, dim=-1)
        left_representations = torch.nn.functional.normalize(left_representations, dim=-1)
        right_representations = torch.nn.functional.normalize(right_representations, dim=-1)
        attention = attention.view(-1, self.weights.shape[0], self.weights.shape[1])
        scores = alpha_scores * (left_representations @ attention @ right_representations.transpose(-2, -1))
        scores = scores.sum(dim=(-2, -1)).view(-1, 1)
        return scores


class DrugDrugAttentionLayer(torch.nn.Module):
    """Co-attention layer for drug pairs."""

    def __init__(self, feature_number: int):
        """Initialize the co-attention layer.

        :param feature_number: Number of input features.
        """
        super().__init__()
        self.weight_query = torch.nn.Parameter(torch.zeros(feature_number, feature_number // 2))
        self.weight_key = torch.nn.Parameter(torch.zeros(feature_number, feature_number // 2))
        self.bias = torch.nn.Parameter(torch.zeros(feature_number // 2))
        self.attention = torch.nn.Parameter(torch.zeros(feature_number // 2))
        self.tanh = torch.nn.Tanh()

        torch.nn.init.xavier_uniform_(self.weight_query)
        torch.nn.init.xavier_uniform_(self.weight_key)
        torch.nn.init.xavier_uniform_(self.bias.view(*self.bias.shape, -1))
        torch.nn.init.xavier_uniform_(self.attention.view(*self.attention.shape, -1))

    def forward(self, left_representations: torch.Tensor, right_representations: torch.Tensor):
        """Make a forward pass with the co-attention calculation.

        :param left_representations: Matrix of left hand side representations.
        :param right_representations: Matrix of right hand side representations.
        :returns: Attention scores.
        """
        keys = left_representations @ self.weight_key
        queries = right_representations @ self.weight_query
        e_activations = queries.unsqueeze(-3) + keys.unsqueeze(-2) + self.bias
        attentions = self.tanh(e_activations) @ self.attention
        return attentions


class SSIDDIBlock(torch.nn.Module):
    """SSIDDI Block with convolution and pooling."""

    def __init__(self, head_number: int, in_channels: int, out_channels: int):
        """Initialize an SSI-DDI Block.

        :param head_number: Number of attention heads.
        :param in_channels: Number of input channels.
        :param out_channels: Number of convolutional filters.
        """
        super().__init__()
        self.conv = GraphAttentionConv(input_dim=in_channels, output_dim=out_channels, num_head=head_number)
        self.readout = MeanReadout()

    def forward(self, molecules: PackedGraph):
        """Make a forward pass.

        :param molecules: A batch of graphs.
        :returns: The molecules with updated atom states and the pooled representations.
        """
        molecules.node_feature = self.conv(molecules, molecules.node_feature)
        h_graphs = self.readout(molecules, molecules.node_feature)
        return molecules, h_graphs


class SSIDDI(Model):
    """An implementation of the SSI-DDI model from [nyamabo2021]_.

    .. seealso:: This model was suggested in https://github.com/AstraZeneca/chemicalx/issues/11

    .. [nyamabo2021] Nyamabo, A. K., *et al.* (2021). `SSI–DDI: substructure–substructure interactions
       for drug–drug interaction prediction <https://doi.org/10.1093/bib/bbab133>`_.
       *Briefings in Bioinformatics*, 22(6).
    """

    def __init__(
        self,
        *,
        molecule_channels: int = TORCHDRUG_NODE_FEATURES,
        hidden_channels=(32, 32),
        head_number=(2, 2),
    ):
        """Instantiate the SSI-DDI model.

        :param molecule_channels: The number of molecular features.
        :param hidden_channels: The list of neurons for each hidden layer block.
        :param head_number: The number of attention heads in each block.
        """
        super().__init__()
        self.initial_norm = LayerNorm(molecule_channels)
        self.blocks = ModuleList()
        self.net_norms = ModuleList()

        channels = molecule_channels
        for hidden_channel, head_number in zip(hidden_channels, head_number):  # noqa: B020
            self.blocks.append(SSIDDIBlock(head_number, channels, hidden_channel))
            self.net_norms.append(LayerNorm(hidden_channel))
            channels = hidden_channel

        self.co_attention = DrugDrugAttentionLayer(channels)
        self.relational_embedding = EmbeddingLayer(channels)
        self.final = nn.Sigmoid()

    def unpack(self, batch: DrugPairBatch):
        """Return the left molecular graph and right molecular graph."""
        return (
            batch.drug_molecules_left,
            batch.drug_molecules_right,
        )

    def _forward_molecules(self, molecules: PackedGraph):
        molecules.node_feature = self.initial_norm(molecules.node_feature)
        representation = []
        for block, net_norm in zip(self.blocks, self.net_norms):
            molecules, pooled_hidden_left = block(molecules)
            representation.append(pooled_hidden_left)
            molecules.node_feature = torch.nn.functional.elu(net_norm(molecules.node_feature))
        return torch.stack(representation, dim=-2)

    def _combine_sides(self, features_left, features_right) -> torch.FloatTensor:
        attentions = self.co_attention(features_left, features_right)
        return self.relational_embedding(features_left, features_right, attentions)

    def forward(self, molecules_left: PackedGraph, molecules_right: PackedGraph) -> torch.FloatTensor:
        """Run a forward pass of the SSI-DDI model.

        :param molecules_left: Batched molecules for the left side drugs.
        :param molecules_right: Batched molecules for the right side drugs.
        :returns: A column vector of predicted synergy scores.
        """
        features_left = self._forward_molecules(molecules_left)
        features_right = self._forward_molecules(molecules_right)
        hidden = self._combine_sides(features_left, features_right)
        return self.final(hidden)
