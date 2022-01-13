import torch
import torch.nn.functional as F
from torchdrug.layers import MeanReadout
from torchdrug.models import GraphConvolutionalNetwork


class EPGCNDS(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 32, out_channels: int = 16):
        super(EPGCNDS, self).__init__()
        self.graph_convolution_in = GraphConvolutionalNetwork(in_channels, hidden_channels)
        self.graph_convolution_out = GraphConvolutionalNetwork(hidden_channels, out_channels)
        self.mean_readout = MeanReadout()
        self.final = torch.nn.Linear(out_channels, 1)

    def forward(self, molecules_left, molecules_right):
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
