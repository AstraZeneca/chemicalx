"""An implementation of the DeepDrug model."""

from typing import Optional

import torch
from torchdrug.data import PackedGraph
from torchdrug.layers import GraphConv, MaxReadout

from chemicalx.constants import TORCHDRUG_NODE_FEATURES
from chemicalx.data import DrugPairBatch
from chemicalx.models import Model

__all__ = [
    "DeepDrug",
]


class DeepDrug(Model):
    """An implementation of the DeepDrug model.

    .. seealso:: 'DeepDrug: A general graph-based deep learning framework for drug relation prediction' by Cao et al. \
    https://www.biorxiv.org/content/10.1101/2020.11.09.375626v1
    """

    def __init__(
        self,
        *,
        context_channels: int = 112,
        molecule_channels: int = TORCHDRUG_NODE_FEATURES,
        num_gcn_layers: int = 4,
        gcn_layer_hidden_size: int = 64,
        out_channels: int = 1,
        dropout_rate: float = 0.1,
    ):
        """Instantiate the DeepDrug model.

        :param context_channels: The number of contexts used for learning. Set to 0 if not using any context. In that
        case make sure you're not calling forward() with the context feature tensor.
        :param molecule_channels: The number of molecular features.
        :param num_gcn_layers: Number of GCN layers.
        :param gcn_layer_hidden_size: number of hidden units in GCN layers
        :param out_channels: The number of output channels.
        :param dropout_rate: Dropout rate on the final fully-connected layer.
        """
        super(DeepDrug, self).__init__()
        self.num_gcn_layers = num_gcn_layers
        self.gcn_layer_hidden_size = gcn_layer_hidden_size
        self.graph_convolution_first = GraphConv(molecule_channels, self.gcn_layer_hidden_size, batch_norm=True)

        # add remaining GCN layers
        self.gcn_layer_list = torch.nn.ModuleList()
        for _ in range(num_gcn_layers - 1):
            self.gcn_layer_list.append(
                GraphConv(self.gcn_layer_hidden_size, self.gcn_layer_hidden_size, batch_norm=True)
            )

        self.max_readout = MaxReadout()
        self.middle_channels = 2 * self.gcn_layer_hidden_size + context_channels  # left/right feats + context
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.batch_norm = torch.nn.BatchNorm1d(self.middle_channels)
        self.final = torch.nn.Linear(self.middle_channels, out_channels)
        self.context_channels = context_channels

    def unpack(self, batch: DrugPairBatch):
        """Return the context features, left drug molecules, and right drug molecules."""
        if self.context_channels > 0:
            return (
                batch.context_features,
                batch.drug_molecules_left,
                batch.drug_molecules_right,
            )
        else:
            return (
                batch.drug_molecules_left,
                batch.drug_molecules_right,
            )

    def forward(
        self, context_features: Optional[torch.FloatTensor], molecules_left: PackedGraph, molecules_right: PackedGraph
    ) -> torch.FloatTensor:

        features_left = self.graph_convolution_first(molecules_left, molecules_left.data_dict["node_feature"])
        features_right = self.graph_convolution_first(molecules_right, molecules_right.data_dict["node_feature"])

        # run remaining GCN layers
        for layer_i in self.gcn_layer_list:
            features_left = layer_i(molecules_left, features_left)
            features_right = layer_i(molecules_right, features_right)

        features_left = self.max_readout(molecules_left, features_left)
        features_right = self.max_readout(molecules_right, features_right)

        if self.context_channels > 0:
            hidden = torch.cat([features_left, features_right, context_features], dim=1)
        else:
            hidden = torch.cat([features_left, features_right], dim=1)
        hidden = self.batch_norm(hidden)
        hidden = self.dropout(hidden)
        hidden = self.final(hidden)
        hidden = torch.sigmoid(hidden)

        return hidden
