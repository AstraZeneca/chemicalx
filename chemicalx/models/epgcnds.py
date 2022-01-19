"""An implementation of the EPGCN-DS model."""

import torch
from torchdrug.data import PackedGraph
from torchdrug.layers import MeanReadout
from torchdrug.models import GraphConvolutionalNetwork

from chemicalx.constants import TORCHDRUG_NODE_FEATURES
from chemicalx.data import DrugPairBatch
from chemicalx.models import Model

__all__ = [
    "EPGCNDS",
]


class EPGCNDS(Model):
    r"""The EPGCN-DS model from [epgcnds]_.

    .. [epgcnds] `Structure-Based Drug-Drug Interaction Detection
       via Expressive Graph Convolutional Networks and Deep Sets
       <https://ojs.aaai.org/index.php/AAAI/article/view/7236>`_
    """

    def __init__(
        self,
        *,
        molecule_channels: int = TORCHDRUG_NODE_FEATURES,
        hidden_channels: int = 32,
        middle_channels: int = 16,
        out_channels: int = 1,
    ):
        """Instantiate the EPGCN-DS model.

        :param molecule_channels: The number of molecular features.
        :param hidden_channels: The number of graph convolutional filters.
        :param middle_channels: The number of hidden layer neurons in the last layer.
        :param out_channels: The number of output channels.
        """
        super(EPGCNDS, self).__init__()
        self.graph_convolution_in = GraphConvolutionalNetwork(molecule_channels, hidden_channels)
        self.graph_convolution_out = GraphConvolutionalNetwork(hidden_channels, middle_channels)
        self.mean_readout = MeanReadout()
        self.final = torch.nn.Linear(middle_channels, out_channels)

    def unpack(self, batch: DrugPairBatch):
        """Return the left molecular graph and right molecular graph."""
        return (
            batch.drug_molecules_left,
            batch.drug_molecules_right,
        )

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
