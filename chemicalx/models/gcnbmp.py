"""An implementation of the GCNBMP model."""
from collections.abc import Sequence
from typing import Tuple, Optional

from more_itertools import chunked, pairwise

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fft import fft, ifft

import torchdrug
from torchdrug import layers, core

from torch_scatter import scatter_add

from chemicalx.constants import TORCHDRUG_NODE_FEATURES
from chemicalx.data import DrugPairBatch
from chemicalx.models import Model


__all__ = [
    "GCNBMP",
]


def circular_correlation(left_x: torch.FloatTensor, right_x: torch.FloatTensor) -> torch.FloatTensor:
    """
    Computes the circular correlation of two vectors a and b via their fast fourier transforms
    In python code, ifft(np.conj(fft(a)) * fft(b)).real
    :param left_x: representation of the left molecule.
    :param right_x: representation of the right molecule.
    :return circ_corr.real: joint representation by circular correlation.
    """
    left_x_cfft = torch.conj(fft(left_x))
    right_x_fft = fft(right_x)
    circ_corr = ifft(torch.mul(left_x_cfft, right_x_fft))

    return circ_corr.real


class Highway(nn.Module):
    def __init__(self, input_size: int, prev_input_size: int):
        """Instantiate the Highway update layer based on https://arxiv.org/abs/1505.00387
        :param input_size: current representation size.
        :param prev_input_size: size of the representation obtained by the previous convolutional layer.
        """
        super(Highway, self).__init__()
        total_size = input_size + prev_input_size
        self.proj = nn.Linear(total_size, input_size)
        self.transform = nn.Linear(total_size, input_size)
        # Not in GCN-BMP paper but present in the original implementation
        self.transform.bias.data.fill_(-2.0)

    def forward(self, input: torch.Tensor, prev_input: torch.Tensor) -> torch.Tensor:
        """
        Compute the gated update

        Parameters:
            input: Current node representations.
            prev_input: Previous layer node representation.

        Returns:
            gated: the highway-updated inputs
        """
        concat_inputs = torch.cat((input, prev_input), 1)
        proj_result = F.relu(self.proj(concat_inputs))
        proj_gate = F.sigmoid(self.transform(concat_inputs))
        gated = (proj_gate * proj_result) + ((1 - proj_gate) * input)
        return gated


class AttentionPooling(nn.Module):
    def __init__(self, molecule_channels: int, hidden_channels: int):
        """Instantiate the Attention pooling layer
        :param molecules_channels: input node features (layer 0 of the backbone).
        :param hidden_channels: final node representation (layer L of the backbone).
        """
        super(AttentionPooling, self).__init__()
        total_features_channels = molecule_channels + hidden_channels
        self.lin = nn.Linear(
            total_features_channels, hidden_channels
        )  # weights here must be shared across all nodes according to the paper
        self.last_rep = nn.Linear(hidden_channels, hidden_channels)

    def forward(self, input_rep: torch.Tensor, final_rep: torch.Tensor, graph_index: torch.Tensor) -> torch.Tensor:
        """
        Compute an attention-based readout using the input and output layers of the 
        RGCN encoder for one molecule.

        Parameters:
            input_rep: input nodes representations
            final_rep: final nodes representations
            graph_index: node to graph readout index

        Returns:
            g: graph-level representation
        """
        att = torch.sigmoid(self.lin(torch.cat((input_rep, final_rep), 1)))
        g = att.mul(self.last_rep(final_rep))
        g = scatter_add(g, graph_index, dim=0)
        return g


class GCNBMPEncoder(nn.Module, core.Configurable):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: int,
        num_relation: int,
        edge_input_dim: Optional[int] = None,
        batch_norm: Optional[bool] = False,
        activation: Optional[str] = "sigmoid",
    ):
        """Instantiate the GCN-BMP encoder.
        :param input_dim: input dimension.
        :param hidden_dims: hidden dimensions.
        :param num_relation: number of relations.
        :param edge_input_dim: dimension of edge features.
        :param batch_norm: apply batch normalization on nodes or not.
        :param activation: activation function.
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
                layers.RelationalGraphConv(
                    left_dim, right_dim, num_relation, edge_input_dim, batch_norm, activation
                )
            )
            self.layers.append(Highway(right_dim, left_dim))

    def forward(self, graph: torchdrug.data.graph.PackedGraph, input: torch.Tensor) -> dict:
        """
        Compute the node representations and the graph representation(s).

        Require the graph(s) to have the same number of relations as this module.

        Parameters:
            graph: :math:`n` graph(s)
            input: input node representations

        Returns:
            dict with ``node_feature`` field:
                node representations of shape :math:`(|V|, d)`
        """
        hiddens = []
        layer_input = input
        prev_gcn = input

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
    r"""The GCNBMP model from [gcnbmp].

    .. [gcnbmp] `GCN-BMP: Investigating graph representation learning for DDI prediction task
       <https://www.sciencedirect.com/science/article/pii/S1046202320300608>`_
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
        """Instantiate the GCN-BMP model.
        :param molecule_channels: The number of node-level features.
        :param hidden_channels: The number of hidden layer neurons in the input layer.
        :param hidden_conv_layers: The number of hidden layers in the encoder.
        :param out_channels: The number of output channels.
        """
        super(GCNBMP, self).__init__()

        self.graph_convolutions = GCNBMPEncoder(
            molecule_channels, [hidden_channels for i in range(hidden_conv_layers)], num_relations
        )

        self.attention_readout = AttentionPooling(molecule_channels, hidden_channels)

        self.final = torch.nn.Linear(hidden_channels, out_channels)

    def unpack(self, batch: DrugPairBatch) -> Tuple[torchdrug.data.graph.PackedGraph, torchdrug.data.graph.PackedGraph]:
        """Return the left and right drugs PackedGraphs."""
        return (
            batch.drug_molecules_left,
            batch.drug_molecules_right,
        )

    def forward(
        self, molecules_left: torchdrug.data.graph.PackedGraph, molecules_right: torchdrug.data.graph.PackedGraph,
    ) -> torch.FloatTensor:
        """
        Run a forward pass of the GCN-BMP model.
        Args:
            molecules_left: The graph of left drug and node features.
            molecules_right: The graph of right drug and node features.
        Returns:
            hidden: A column vector of predicted synergy scores.
        """
        features_left = self.graph_convolutions(molecules_left, molecules_left.data_dict["node_feature"])[
            "node_feature"
        ]

        features_right = self.graph_convolutions(molecules_right, molecules_right.data_dict["node_feature"])[
            "node_feature"
        ]

        features_left = self.attention_readout(
            molecules_left.data_dict["node_feature"], features_left, molecules_left.node2graph
        )
        features_right = self.attention_readout(
            molecules_right.data_dict["node_feature"], features_right, molecules_right.node2graph
        )

        joint_rep = circular_correlation(features_left, features_right)

        interaction_estimators = torch.sigmoid(self.final(joint_rep))

        return interaction_estimators
