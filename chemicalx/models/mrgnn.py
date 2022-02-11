"""An implementation of the MR-GNN model."""

from typing import Any

import torch
from torchdrug.layers import MeanReadout
from torchdrug.models import GraphConvolutionalNetwork

from chemicalx.compat import PackedGraph
from chemicalx.constants import TORCHDRUG_NODE_FEATURES
from chemicalx.data import DrugPairBatch
from chemicalx.models import Model

__all__ = [
    "MRGNN",
]


class MRGNN(Model):
    """An implementation of the MR-GNN model from [xu2019]_.

    .. seealso:: This model was suggested in https://github.com/AstraZeneca/chemicalx/issues/12

    .. [xu2019] Xu, N., *et al.* (2019). `MR-GNN: Multi-resolution and dual graph neural network for
       predicting structured entity interactions <https://doi.org/10.24963/ijcai.2019/551>`_.
       *IJCAI International Joint Conference on Artificial Intelligence*, 2019, 3968–3974.
    """

    def __init__(
        self,
        *,
        molecule_channels: int = TORCHDRUG_NODE_FEATURES,
        hidden_channels: int = 32,
        middle_channels: int = 16,
        layer_count: int = 4,
        out_channels: int = 1,
    ):
        """Instantiate the MRGNN model.

        :param molecule_channels: The number of molecular features.
        :param hidden_channels: The number of graph convolutional filters.
        :param middle_channels: The number of hidden layer neurons in the last layer.
        :param layer_count: The number of graph convolutional and recurrent blocks.
        :param out_channels: The number of output channels.
        """
        super().__init__()
        self.graph_convolutions = torch.nn.ModuleList()
        self.graph_convolutions.append(GraphConvolutionalNetwork(molecule_channels, hidden_channels))
        for _ in range(1, layer_count):
            self.graph_convolutions.append(GraphConvolutionalNetwork(hidden_channels, hidden_channels))
        self.border_rnn = torch.nn.LSTM(hidden_channels, hidden_channels, 1)
        self.middle_rnn = torch.nn.LSTM(2 * hidden_channels, 2 * hidden_channels, 1)
        self.readout = MeanReadout()
        self.final = torch.nn.Sequential(
            # First two are the "bottleneck"
            torch.nn.Linear(6 * hidden_channels, middle_channels),
            torch.nn.ReLU(),
            # Second to are the "final"
            torch.nn.Linear(middle_channels, out_channels),
            torch.nn.Sigmoid(),
        )

    def unpack(self, batch: DrugPairBatch):
        """Return the left molecular graph and right molecular graph."""
        return (
            batch.drug_molecules_left,
            batch.drug_molecules_right,
        )

    def _forward_molecules(
        self,
        conv: GraphConvolutionalNetwork,
        molecules: PackedGraph,
        gcn_hidden: torch.FloatTensor,
        states: Any,
    ) -> torch.FloatTensor:
        """Do a forward pass with a pack of molecules and a GCN layer.

        :param conv: The graph convolutational layer.
        :param molecules: The molecules used for message passing and information propagation.
        :param gcn_hidden: The states of the previous graph convolutional layers.
        :param states: The hidden and cell states of the previous recurrent block.
        :returns: New graph convolutional and recurrent output, states and graph level features.
        """
        gcn_hidden = conv(molecules, gcn_hidden)["node_feature"]
        rnn_out, (hidden_state, cell_state) = self.border_rnn(gcn_hidden[None, :, :], states)
        rnn_out = rnn_out.squeeze()
        graph_level = self.readout(molecules, gcn_hidden)
        return gcn_hidden, rnn_out, (hidden_state, cell_state), graph_level

    def forward(self, molecules_left: PackedGraph, molecules_right: PackedGraph) -> torch.FloatTensor:
        """Run a forward pass of the MR-GNN model.

        :param molecules_left: Batched molecules for the left side drugs.
        :param molecules_right: Batched molecules for the right side drugs.
        :returns: A column vector of predicted synergy scores.
        """
        gcn_hidden_left = molecules_left.data_dict["node_feature"]
        gcn_hidden_right = molecules_right.data_dict["node_feature"]
        left_states, right_states, shared_states = None, None, None
        for conv in self.graph_convolutions:
            gcn_hidden_left, rnn_out_left, left_states, graph_level_left = self._forward_molecules(
                conv, molecules_left, gcn_hidden_left, left_states
            )
            gcn_hidden_right, rnn_out_right, right_states, graph_level_right = self._forward_molecules(
                conv, molecules_right, gcn_hidden_right, right_states
            )

            shared_graph_level = torch.cat([graph_level_left, graph_level_right], dim=1)
            shared_out, shared_states = self.middle_rnn(shared_graph_level[None, :, :], shared_states)

        rnn_pooled_left = self.readout(molecules_left, rnn_out_left)
        rnn_pooled_right = self.readout(molecules_right, rnn_out_right)
        shared_out = shared_out.squeeze()
        out = torch.cat([shared_graph_level, shared_out, rnn_pooled_left, rnn_pooled_right], dim=1)
        out = self.final(out)
        return out
