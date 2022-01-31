"""An implementation of the GCNBMP model."""
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fft import fft, ifft

from torchdrug import layers
from torchdrug.data import PackedGraph
from torchdrug.layers import MeanReadout
from torchdrug.models import GraphConvolutionalNetwork, RelationalGraphConvolutionalNetwork

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
    :param left_x:
    :param right_x:
    (a - j * b) * (c + j * d) = (ac + bd) + j * (ad - bc)
    :return:
    """
    left_x_cfft = torch.conj(fft(left_x))
    right_x_fft = fft(right_x)
    circ_corr = ifft(torch.mul(left_x_cfft, right_x_fft))

    return circ_corr.real


class Highway(nn.Module):  # The traditional one, not exactly the one from the GCN-BMP paper
    def __init__(self, input_size):
        super(Highway, self).__init__()
        self.proj = nn.Linear(input_size, input_size)
        self.transform = nn.Linear(input_size, input_size)
        self.transform.bias.data.fill_(-2.0)

    def forward(self, input):
        proj_result = F.relu(self.proj(input))
        proj_gate = F.sigmoid(self.transform(input))
        gated = (proj_gate * proj_result) + ((1 - proj_gate) * input)
        return gated


class AttentionPooling(nn.Module):
    def __init__(self, molecule_channels, hidden_channels):
        super(AttentionPooling, self).__init__()
        # Here we concatenate layer 0 with the final representation
        total_features_channels = molecule_channels + hidden_channels
        self.lin = nn.Linear(
            total_features_channels, hidden_channels
        )  # weights here must be shared across all nodes according to the paper
        self.last_rep = nn.Linear(hidden_channels, hidden_channels)

    def forward(self, input_rep, final_rep, graph_index):
        #        print("Input and Final representation shapes: ")
        #        print(input_rep.shape, final_rep.shape)
        att = torch.sigmoid(self.lin(torch.cat((input_rep, final_rep), 1)))  # .mul(input)
        #        print("Attention shape", att.shape)
        g = att.mul(self.last_rep(final_rep))

        g = scatter_add(g, graph_index, dim=0)


        return g


class GCNBMP(Model):
    r"""An implementation of the GCNBMP model.
    .. seealso:: https://github.com/AstraZeneca/chemicalx/issues/21
    """

    def __init__(
        self,
        *,
        molecule_channels: int = TORCHDRUG_NODE_FEATURES,
        hidden_channels: int = 16,
        hidden_conv_layers: int = 1,
        out_channels: int = 1,
    ):
        super(GCNBMP, self).__init__()

        self.graph_convolution_in = RelationalGraphConvolutionalNetwork(molecule_channels, hidden_channels, 4)
        self.highway = Highway(hidden_channels)

        self.encoding_backbone = nn.ModuleList()

        for i in range(hidden_conv_layers):
            self.encoding_backbone.append(RelationalGraphConvolutionalNetwork(hidden_channels, hidden_channels, 4))
            self.encoding_backbone.append(Highway(hidden_channels))

        self.attention_readout_left = AttentionPooling(molecule_channels, hidden_channels)
        self.attention_readout_right = AttentionPooling(molecule_channels, hidden_channels)

        self.final = torch.nn.Linear(hidden_channels, out_channels)

    def unpack(self, batch: DrugPairBatch):
        """Return the context features, left drug features, and right drug features."""
        return (
            batch.drug_molecules_left,
            batch.drug_molecules_right,
        )

    def forward(self, molecules_left: torch.FloatTensor, molecules_right: torch.FloatTensor,) -> torch.FloatTensor:

        features_left = self.graph_convolution_in(molecules_left, molecules_left.data_dict["node_feature"])[
            "node_feature"
        ]

        features_right = self.graph_convolution_in(molecules_right, molecules_right.data_dict["node_feature"])[
            "node_feature"
        ]

        # for i,l in enumerate(self.encoding_backbone):
        #    print(i, l)

        features_left = self.attention_readout_left(molecules_left.data_dict["node_feature"], features_left, molecules_left.node2graph)
        features_right = self.attention_readout_right(molecules_right.data_dict["node_feature"], features_right, molecules_right.node2graph)

        joint_rep = circular_correlation(features_left, features_right)



        # features_left = self.graph_convolution_out(molecules_left, features_left)["node_feature"]
        # features_right = self.graph_convolution_out(molecules_right, features_right)["node_feature"]

        # features_left = self.mean_readout(molecules_left, features_left)
        # features_right = self.mean_readout(molecules_right, features_right)
        # hidden = features_left + features_right
        hidden = torch.sigmoid(self.final(joint_rep))
        return hidden