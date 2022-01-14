"""An implementation of the EPGCN-DS model."""

import torch
from torchdrug.data import PackedGraph
from torchdrug.layers import MeanReadout
from torchdrug.models import GraphConvolutionalNetwork

from chemicalx.models import ContextlessModel

__all__ = [
    "EPGCNDS",
]


class EPGCNDS(ContextlessModel):
    r"""The EPGCN-DS model from [epgcnds]_.

    .. [epgcnds] `"Structure-Based Drug-Drug Interaction Detection
    via Expressive Graph Convolutional Networks and Deep Sets"
    <https://ojs.aaai.org/index.php/AAAI/article/view/7236>`_ paper.
    """

    def __init__(self, *, in_channels: int, hidden_channels: int = 32, out_channels: int = 16):
        """Instantiate the EPGCN-DS model.

        :param in_channels: The number of molecular features.
        :param hidden_channels: The number of graph convolutional filters.
        :param out_channels: The number of hidden layer neurons in the last layer.
        """
        super(EPGCNDS, self).__init__()
        self.graph_convolution_in = GraphConvolutionalNetwork(in_channels, hidden_channels)
        self.graph_convolution_out = GraphConvolutionalNetwork(hidden_channels, out_channels)
        self.mean_readout = MeanReadout()
        self.final = torch.nn.Linear(out_channels, 1)

    def forward(self, molecules_left: PackedGraph, molecules_right: PackedGraph) -> torch.FloatTensor:
        """
        Run a forward pass of the EPGCN-DS model.

        Args:
            molecules_left (torch.FloatTensor): Batched molecules for the left side drugs.
            molecules_right (torch.FloatTensor): Batched molecules for the right side drugs.
        Returns:
            hidden (torch.FloatTensor): A column vector of predicted synergy scores.
        """
        features_left = self.graph_convolution_in(molecules_left, molecules_left.data_dict["node_feature"])[
            "node_feature"
        ]
        features_right = self.graph_convolution_in(molecules_right, molecules_right.data_dict["node_feature"])[
            "node_feature"
        ]

        features_left = self.graph_convolution_out(molecules_left, features_left)["node_feature"]
        features_right = self.graph_convolution_out(molecules_right, features_right)["node_feature"]

        features_left = self.mean_readout(molecules_left, features_left)
        features_right = self.mean_readout(molecules_right, features_right)
        hidden = features_left + features_right
        hidden = torch.sigmoid(self.final(hidden))
        return hidden
