"""An implementation of the GCNBMP model."""
from collections.abc import Sequence
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa:N812
import torchdrug
from more_itertools import chunked, pairwise
from torch.fft import fft, ifft
from torch_scatter import scatter_add
from torchdrug import core, layers

from chemicalx.constants import TORCHDRUG_NODE_FEATURES
from chemicalx.data import DrugPairBatch
from chemicalx.models import Model

__all__ = [
    "GCNBMP",
]


def circular_correlation(left_x: torch.FloatTensor, right_x: torch.FloatTensor) -> torch.FloatTensor:
    """
    Compute the circular correlation of two vectors a and b via their fast fourier transforms.

    :param left_x: Molecular representations on the left.
    :param right_x: Molecular representations on the right.
    :returns: Joint representation by circular correlation.
    """
    left_x_cfft = torch.conj(fft(left_x))
    right_x_fft = fft(right_x)
    circ_corr = ifft(torch.mul(left_x_cfft, right_x_fft))
    return circ_corr.real


class Highway(nn.Module):
    """
    The Highway update layer from [highway].

    .. [highway] `Highway Networks
       <https://arxiv.org/abs/1505.00387>`_
    """

    def __init__(self, input_size: int, prev_input_size: int):
        """
        Instantiate the Highway update layer.

        :param input_size: Current representation size.
        :param prev_input_size: Size of the representation obtained by the previous convolutional layer.
        """
        super(Highway, self).__init__()
        total_size = input_size + prev_input_size
        self.proj = nn.Linear(total_size, input_size)
        self.transform = nn.Linear(total_size, input_size)
        self.transform.bias.data.fill_(-2.0)

    def forward(self, input: torch.Tensor, prev_input: torch.Tensor) -> torch.Tensor:
        """
        Compute the gated update.

        :param input: Current node representations.
        :param prev_input: Previous layer node representations.
        :returns: The highway-updated inputs.
        """
        concat_inputs = torch.cat((input, prev_input), 1)
        proj_result = F.relu(self.proj(concat_inputs))
        proj_gate = F.sigmoid(self.transform(concat_inputs))
        gated = (proj_gate * proj_result) + ((1 - proj_gate) * input)
        return gated


class AttentionPooling(nn.Module):
    """
    The Attention pooling layer from [gcnbmp].

    .. [gcnbmp] `GCN-BMP: Investigating graph representation learning for DDI prediction task
       <https://www.sciencedirect.com/science/article/pii/S1046202320300608>`_
    """

    def __init__(self, molecule_channels: int, hidden_channels: int):
        """
        Instantiate the Attention pooling layer.

        :param molecule_channels: Input node features.
        :param hidden_channels: Final node representation.
        """
        super(AttentionPooling, self).__init__()
        total_features_channels = molecule_channels + hidden_channels
        self.lin = nn.Linear(
            total_features_channels, hidden_channels
        )  # weights here must be shared across all nodes according to the paper
        self.last_rep = nn.Linear(hidden_channels, hidden_channels)

    def forward(self, input_rep: torch.Tensor, final_rep: torch.Tensor, graph_index: torch.Tensor) -> torch.Tensor:
        """
        Compute an attention-based readout using the input and output layers of the RGCN encoder for one molecule.

        :param input_rep: Input nodes representations.
        :param final_rep: Final nodes representations.
        :param graph_index: Node to graph readout index.
        :returns: Graph-level representation.
        """
        att = torch.sigmoid(self.lin(torch.cat((input_rep, final_rep), 1)))
        g = att.mul(self.last_rep(final_rep))
        g = scatter_add(g, graph_index, dim=0)
        return g


class GCNBMPEncoder(nn.Module, core.Configurable):
    """
    The drug encoding backbone from [gcnbmp].

    .. [gcnbmp] `GCN-BMP: Investigating graph representation learning for DDI prediction task
       <https://www.sciencedirect.com/science/article/pii/S1046202320300608>`_

    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        num_relation: int,
        edge_input_dim: Optional[int] = None,
        batch_norm: Optional[bool] = False,
        activation: Optional[str] = "sigmoid",
    ):
        """
        Instantiate the GCN-BMP encoder.

        :param input_dim: Input dimensions.
        :param hidden_dims: Hidden dimensions.
        :param num_relation: Number of relations.
        :param edge_input_dim: Dimension of edge features.
        :param batch_norm: Apply batch normalization on nodes or not.
        :param activation: Activation function.
        """
        super(GCNBMPEncoder, self).__init__()

        if not isinstance(hidden_dims, Sequence):
            hidden_dims = [hidden_dims]
        self.input_dim = input_dim
        self.dims = [input_dim] + list(hidden_dims)
        self.num_relation = num_relation

        self.layers = nn.ModuleList()
        for left_dim, right_dim in pairwise(self.dims):
            self.layers.append(
                layers.RelationalGraphConv(left_dim, right_dim, num_relation, edge_input_dim, batch_norm, activation)
            )
            self.layers.append(Highway(right_dim, left_dim))

    def forward(self, graph: torchdrug.data.graph.PackedGraph, input_node_features: torch.Tensor) -> dict:
        """
        Compute the node representations and the graph representation(s).

        :param graph: Batch of molecular graphs.
        :param input_node_features: Input node representations
        :returns: Node representation matrix.
        """
        hiddens = []
        layer_input = input_node_features
        prev_gcn = input_node_features

        for conv, highway in chunked(self.layers, 2):
            hidden = conv(graph, layer_input)
            hiddens.append(hidden)
            hidden = highway(hidden, prev_gcn)
            hiddens.append(hidden)
            layer_input = hidden
            prev_gcn = hiddens[-2]
        node_feature = hiddens[-1]
        return {"node_feature": node_feature}


class GCNBMP(Model):
    """An implementation of the GCN-BMP model from [chen2020]_.
    .. seealso:: This model was suggested in https://github.com/AstraZeneca/chemicalx/issues/21
    .. [chen2020] Chen, X., *et al.* (2020). `GCN-BMP: Investigating graph representation learning
       for DDI prediction task <https://doi.org/10.1016/j.ymeth.2020.05.014>`_. *Methods*, 179, 47–54.
    """

    def __init__(
        self,
        *,
        molecule_channels: int = TORCHDRUG_NODE_FEATURES,
        num_relations: int = 4,  # TODO: This default value should be set by a dataset-specific constant
        hidden_channels: int = 16,
        hidden_conv_layers: int = 1,
        out_channels: int = 1,
    ):
        """
        Instantiate the GCN-BMP model.

        :param molecule_channels: The number of node-level features.
        :param num_relations: Number of edge types.
        :param hidden_channels: The number of hidden layer neurons in the input layer.
        :param hidden_conv_layers: The number of hidden layers in the encoder.
        :param out_channels: The number of output channels.
        """
        super(GCNBMP, self).__init__()
        self.graph_convolutions = GCNBMPEncoder(
            molecule_channels, [hidden_channels for _ in range(hidden_conv_layers)], num_relations
        )

        self.attention_readout = AttentionPooling(molecule_channels, hidden_channels)

        self.final = torch.nn.Linear(hidden_channels, out_channels)

    def unpack(self, batch: DrugPairBatch) -> Tuple[torchdrug.data.graph.PackedGraph, torchdrug.data.graph.PackedGraph]:
        """Return the left and right drugs PackedGraphs."""
        return (
            batch.drug_molecules_left,
            batch.drug_molecules_right,
        )

    def encoder_pass(self, molecules: torchdrug.data.graph.PackedGraph) -> torch.FloatTensor:
        """
        Run a forward pass of the encoder layers.

        :param molecules: The bacthed molecular graphs.
        :returns: A matrix of molecular features
        """
        features = self.graph_convolutions(molecules, molecules.data_dict["node_feature"])["node_feature"]
        features = self.attention_readout(molecules.data_dict["node_feature"], features, molecules.node2graph)
        return features

    def forward(
        self,
        molecules_left: torchdrug.data.graph.PackedGraph,
        molecules_right: torchdrug.data.graph.PackedGraph,
    ) -> torch.FloatTensor:
        """
        Run a forward pass of the GCN-BMP model.

        :param molecules_left: The graph of left drug and node features.
        :param molecules_right: The graph of right drug and node features.
        :returns: A column vector of predicted synergy scores.
        """
        features_left = self.encoder_pass(molecules_left)
        features_right = self.encoder_pass(molecules_right)
        joint_rep = circular_correlation(features_left, features_right)
        interaction_estimators = torch.sigmoid(self.final(joint_rep))
        return interaction_estimators
