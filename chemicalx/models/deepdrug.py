"""An implementation of the DeepDrug model."""

import torch
from torchdrug.data import PackedGraph
from torchdrug.layers import MaxReadout
from torchdrug.models import GraphConvolutionalNetwork

from chemicalx.constants import TORCHDRUG_NODE_FEATURES
from chemicalx.data import DrugPairBatch
from chemicalx.models import Model

__all__ = [
    "DeepDrug",
]


class DeepDrug(Model):
    """An implementation of the DeepDrug model.

    .. see also:: https://github.com/AstraZeneca/chemicalx/issues/14
    """

    def __init__(
        self,
        *,
        context_channels: int = 112,
        molecule_channels: int = TORCHDRUG_NODE_FEATURES,
        num_gcn_layers: int = 4,
        gcn_layer_hidden_size: int = 64,
        out_channels: int = 1,
    ):
        """Instantiate the DeepDrug model.

        :param context_channels: The number of contexts used for learning.
        :param molecule_channels: The number of molecular features.
        :param num_gcn_layers: Number of GCN layers
        :param gcn_layer_hidden_size: number of hidden units in GCN layers
        :param out_channels: The number of output channels.
        """
        super(DeepDrug, self).__init__()
        self.num_gcn_layers = num_gcn_layers
        self.gcn_layer_hidden_size = gcn_layer_hidden_size
        self.graph_convolution_first = GraphConvolutionalNetwork(molecule_channels, self.gcn_layer_hidden_size)

        # add remaining GCN layers
        self.gcn_layer_list = torch.nn.ModuleList()
        for i in range(num_gcn_layers - 1):
            self.gcn_layer_list.append(
                GraphConvolutionalNetwork(self.gcn_layer_hidden_size, self.gcn_layer_hidden_size)
            )

        self.max_readout = MaxReadout()
        self.middle_channels = 2 * self.gcn_layer_hidden_size + context_channels  # left/right feats + context
        self.final = torch.nn.Linear(self.middle_channels, out_channels)

    def unpack(self, batch: DrugPairBatch):
        """Return the context features, left drug molecules, and right drug molecules."""
        return (
            batch.context_features,
            batch.drug_molecules_left,
            batch.drug_molecules_right,
        )

    def forward(
        self, context_features: torch.FloatTensor, molecules_left: PackedGraph, molecules_right: PackedGraph
    ) -> torch.FloatTensor:

        features_left = self.graph_convolution_first(molecules_left, molecules_left.data_dict["node_feature"])[
            "node_feature"
        ]
        features_right = self.graph_convolution_first(molecules_right, molecules_right.data_dict["node_feature"])[
            "node_feature"
        ]

        # run remaining GCN layers
        for i in range(self.num_gcn_layers - 1):
            layer_i = self.gcn_layer_list[i]
            features_left = layer_i(molecules_left, features_left)["node_feature"]
            features_right = layer_i(molecules_right, features_right)["node_feature"]

        features_left = self.max_readout(molecules_left, features_left)
        features_right = self.max_readout(molecules_right, features_right)

        hidden = torch.cat([features_left, features_right, context_features], dim=1)
        hidden = self.final(hidden)
        hidden = torch.sigmoid(hidden)

        return hidden
