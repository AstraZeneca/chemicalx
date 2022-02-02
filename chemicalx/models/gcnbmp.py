"""An implementation of the GCNBMP model."""
import sys
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fft import fft, ifft

import torchdrug
from torchdrug import layers, core
from torchdrug.models import RelationalGraphConvolutionalNetwork

from torch_scatter import scatter_add

from chemicalx.constants import TORCHDRUG_NODE_FEATURES
from chemicalx.data import DrugPairBatch
from chemicalx.models import Model


__all__ = [
    "GCNBMP",
]


def circular_correlation(left_x, right_x):
    """
    Computes the circular correlation of two vectors a and b via their fast fourier transforms
    In python code, ifft(np.conj(fft(a)) * fft(b)).real
    :param left_x (Tensor): representation of the left molecule.
    :param right_x (Tensor): representation of the right molecule.
    :return circ_corr.real (Tensor): joint representation by circular correlation.
    """
    left_x_cfft = torch.conj(fft(left_x))
    right_x_fft = fft(right_x)
    circ_corr = ifft(torch.mul(left_x_cfft, right_x_fft))

    return circ_corr.real


class Highway(nn.Module):
    def __init__(self, input_size, prev_input_size):
        """Instantiate the Highway update layer based on https://arxiv.org/abs/1505.00387
        :param input_size (int): current representation.
        :param prev_input_size (int): representation obtained by the previous convolutional layer.
        """
        super(Highway, self).__init__()
        total_size = input_size + prev_input_size
        self.proj = nn.Linear(total_size, input_size)
        self.transform = nn.Linear(total_size, input_size)
        # Not in GCN-BMP paper but present in the original implementation
        self.transform.bias.data.fill_(-2.0)

    def forward(self, input, prev_input):
        """
        Compute the gated update

        Parameters:
            input: Current node representations.
            prev_input: Previous layer node representation.

        Returns:
            gated: the highway-updated 
        """
        concat_inputs = torch.cat((input, prev_input), 1)
        proj_result = F.relu(self.proj(concat_inputs))
        proj_gate = F.sigmoid(self.transform(concat_inputs))
        gated = (proj_gate * proj_result) + ((1 - proj_gate) * input)
        return gated


class AttentionPooling(nn.Module):
    def __init__(self, molecule_channels, hidden_channels):
        """Instantiate the Attention pooling layer
        :param molecules_channels (int): input node features (layer 0 of the backbone).
        :param hidden_channels (int): final node representation (layer L of the backbone).
        """
        super(AttentionPooling, self).__init__()
        total_features_channels = molecule_channels + hidden_channels
        self.lin = nn.Linear(
            total_features_channels, hidden_channels
        )  # weights here must be shared across all nodes according to the paper
        self.last_rep = nn.Linear(hidden_channels, hidden_channels)

    def forward(self, input_rep, final_rep, graph_index):
        """
        Compute an attention-based readout using the input and output layers of the 
        RGCN encoder for one molecule.

        Parameters:
            input_rep (Tensor): input nodes representations
            final_rep (Tensor): final nodes representations
            graph_index (Tensor): node to graph readout index

        Returns:
            g (Tensor): graph-level representation
        """
        att = torch.sigmoid(self.lin(torch.cat((input_rep, final_rep), 1)))
        g = att.mul(self.last_rep(final_rep))
        g = scatter_add(g, graph_index, dim=0)
        return g


class GCNBMPEncoder(nn.Module, core.Configurable):
    def __init__(
        self,
        input_dim,
        hidden_dims,
        num_relation,
        edge_input_dim=None,
        short_cut=False,
        batch_norm=False,
        activation="sigmoid",
        concat_hidden=False,
    ):
        """Instantiate the GCN-BMP encoder.
        :param input_dim (int): input dimension.
        :param hidden_dims (list of int): hidden dimensions.
        :param output_dim (int): output dimension.
        :param num_relation (int): number of relations.
        :param edge_input_dim (int, optional): dimension of edge features.
        :param batch_norm (bool, optional): apply batch normalization on nodes or not.
        :param activation (str or function, optional): activation function.
        """
        super(GCNBMPEncoder, self).__init__()

        if not isinstance(hidden_dims, Sequence):
            hidden_dims = [hidden_dims]
        self.input_dim = input_dim
        self.output_dim = hidden_dims[-1] * (len(hidden_dims) if concat_hidden else 1)
        self.dims = [input_dim] + list(hidden_dims)
        self.num_relation = num_relation
        self.short_cut = short_cut
        self.concat_hidden = concat_hidden

        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(
                layers.RelationalGraphConv(
                    self.dims[i], self.dims[i + 1], num_relation, edge_input_dim, batch_norm, activation
                )
            )
            self.layers.append(Highway(self.dims[i + 1], self.dims[i]))

    def forward(self, graph, input):
        """
        Compute the node representations and the graph representation(s).

        Require the graph(s) to have the same number of relations as this module.

        Parameters:
            graph (Graph): :math:`n` graph(s)
            input (Tensor): input node representations

        Returns:
            dict with ``node_feature`` field:
                node representations of shape :math:`(|V|, d)`
        """
        hiddens = []
        layer_input = input
        prev_gcn = input

        for i, layer in enumerate(self.layers):
            if isinstance(
                layer, layers.RelationalGraphConv
            ):  # Achievable with 0==i%2, maybe better perf but less readable
                hidden = layer(graph, layer_input)
                if self.short_cut and hidden.shape == layer_input.shape:
                    hidden = hidden + layer_input
                hiddens.append(hidden)
                layer_input = hidden
            else:
                hidden = layer(hidden, prev_gcn)
                if self.short_cut and hidden.shape == layer_input.shape:
                    hidden = hidden + layer_input
                hiddens.append(hidden)
                layer_input = hidden
                prev_gcn = hiddens[-2]

        if self.concat_hidden:
            node_feature = torch.cat(hiddens, dim=-1)
        else:
            node_feature = hiddens[-1]

        return {"node_feature": node_feature}


class GCNBMP(Model):
    r"""An implementation of the GCNBMP model.

    .. seealso:: https://github.com/AstraZeneca/chemicalx/issues/21
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

        self.attention_readout_left = AttentionPooling(molecule_channels, hidden_channels)
        self.attention_readout_right = AttentionPooling(molecule_channels, hidden_channels)

        self.final = torch.nn.Linear(hidden_channels, out_channels)

    def unpack(self, batch: DrugPairBatch):
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
            molecules_left (torchdrug.data.graph.PackedGraph): The graph of left drug and node features.
            molecules_right (torchdrug.data.graph.PackedGraphtorch.FloatTensor): The graph of right drug and node features.
        Returns:
            hidden (torch.FloatTensor): A column vector of predicted synergy scores.
        """
        features_left = self.graph_convolutions(molecules_left, molecules_left.data_dict["node_feature"])[
            "node_feature"
        ]

        features_right = self.graph_convolutions(molecules_right, molecules_right.data_dict["node_feature"])[
            "node_feature"
        ]

        features_left = self.attention_readout_left(
            molecules_left.data_dict["node_feature"], features_left, molecules_left.node2graph
        )
        features_right = self.attention_readout_right(
            molecules_right.data_dict["node_feature"], features_right, molecules_right.node2graph
        )

        joint_rep = circular_correlation(features_left, features_right)

        interaction_estimators = torch.sigmoid(self.final(joint_rep))

        return interaction_estimators
