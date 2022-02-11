"""An implementation of the DeepDDS model.

DeepDDS: deep graph neural network with attention mechanism to predict
synergistic drug combinations.

Paper on arXiv:
arXiv:2107.02467 [cs.LG]
https://arxiv.org/abs/2107.02467

Published Code:
https://github.com/Sinwang404/DeepDDs/tree/master

SMILES strings transformed into a graph representation are used as input to
both the GAT and the GCN version of the model.

MLP is used to extract the feature embedding of gene expression profiles of
cancer cell line.

The embedding vector from both inputs are concatenated and fed into the
fully connected layers for binary classification of the drug combination as
synergistic or antagonistic.
"""

from typing import List, Optional

import torch
from torch import nn
from torch.nn.functional import normalize
from torchdrug.layers import MLP, MaxReadout
from torchdrug.models import GraphConvolutionalNetwork

from chemicalx.compat import PackedGraph
from chemicalx.constants import TORCHDRUG_NODE_FEATURES
from chemicalx.data import DrugPairBatch
from chemicalx.models import Model

__all__ = [
    "DeepDDS",
]


class DeepDDS(Model):
    """An implementation of the DeepDDS model from [wang2021]_.

    This implementation follows the code on github where the paper and
    the code diverge.
    https://github.com/Sinwang404/DeepDDs/tree/master

    .. seealso:: This model was suggested in https://github.com/AstraZeneca/chemicalx/issues/19

    .. [wang2021] Wang, J., *et al.* (2021). `DeepDDS: deep graph neural network with attention
       mechanism to predict synergistic drug combinations <http://arxiv.org/abs/2107.02467>`_.
       *arXiv*, 2107.02467.
    """

    def __init__(
        self,
        *,
        context_channels: int,
        context_hidden_dims: Optional[List[int]] = None,
        drug_channels: int = TORCHDRUG_NODE_FEATURES,
        drug_gcn_hidden_dims: Optional[List[int]] = None,
        drug_mlp_hidden_dims: Optional[List[int]] = None,
        context_output_size: int = 32,
        fc_hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.5,  # Different from rate used in paper
    ):
        """Instantiate the DeepDDS model.

        :param context_channels:
            The size of the context feature embedding for cell lines.
        :param context_hidden_dims:
            The hidden dimensions of the MLP used to extract the context
            feature embedding. Default: [32, 32]. Note: the last layer
            will always be of size=context_output_size and appended to the
            provided list.
        :param drug_channels:
            The number of input channels for the GCN. Default:
            ``chemicalx.constants.TORCHDRUG_NODE_FEATURES``.
        :param drug_gcn_hidden_dims:
            The hidden dimensions of the GCN. Default:
            [drug_channels, drug_channels * 2, drug_channels * 4].
        :param drug_mlp_hidden_dims:
            The hidden dimensions of the MLP used to extract the drug features.
            Default: [drug_channels * 2]. Note: The input layer will be set
            automatically to match the last layer of the preceding GCN
            layer. The last layer will always be of size=drug_output_size and
            appended to the provided list.
        :param context_output_size:
            The size of the context output embedding. This is the size of
            the vectors that are concatenated before running the final fully
            connected layers.
        :param fc_hidden_dims:
            The hidden dimensions of the final fully connected layers.
            Default: [32, 32]. Note: the last layer will always be of
            size=1 (the synergy prediction readout) and appended to the
            provided list.
        :param dropout:
            The dropout rate used in the FC layers of the drugs after the
            initial GCN and in the final fully connected layers.
        """
        super().__init__()
        # Check default parameters:
        # Defaults are different from the original implementation.
        if context_hidden_dims is None:
            context_hidden_dims = [32, 32]
        if drug_gcn_hidden_dims is None:
            drug_gcn_hidden_dims = [drug_channels, drug_channels * 2, drug_channels * 4]
        if drug_mlp_hidden_dims is None:
            drug_mlp_hidden_dims = [drug_channels * 2]
        if fc_hidden_dims is None:
            fc_hidden_dims = [32, 32]

        # Cell feature extraction with MLP
        self.cell_mlp = MLP(
            input_dim=context_channels,
            # Paper: [2048, 512, context_output_size]
            # Code: [512, 256, context_output_size]
            # Our code: [32, 32, context_output_size]
            hidden_dims=[*context_hidden_dims, context_output_size],
        )

        # GCN
        # Paper: GCN with three hidden layers + global max pool
        # Code: Same as paper + two FC layers. With different layer sizes.
        self.drug_conv = GraphConvolutionalNetwork(
            # Paper: [1024, 512, 156],
            # Code: [drug_channels, drug_channels * 2, drug_channels * 4]
            input_dim=drug_channels,
            hidden_dims=drug_gcn_hidden_dims,
            activation="relu",
        )
        self.drug_readout = MaxReadout()

        # Paper: no FC layers after GCN layers and global max pooling
        self.drug_mlp = MLP(
            input_dim=drug_gcn_hidden_dims[-1],
            hidden_dims=[*drug_mlp_hidden_dims, context_output_size],
            dropout=dropout,
            activation="relu",
        )

        # Final layers
        self.final = nn.Sequential(
            MLP(
                input_dim=context_output_size * 3,
                hidden_dims=[*fc_hidden_dims, 1],
                dropout=dropout,
            ),
            torch.nn.Sigmoid(),
        )

    def unpack(self, batch: DrugPairBatch):
        """Return the context features, left drug features and right drug features."""
        return batch.context_features, batch.drug_molecules_left, batch.drug_molecules_right

    def _forward_molecules(self, molecules: PackedGraph) -> torch.FloatTensor:
        features = self.drug_conv(molecules, molecules.data_dict["node_feature"])["node_feature"]
        features = self.drug_readout(molecules, features)
        return self.drug_mlp(features)

    def forward(
        self, context_features: torch.FloatTensor, molecules_left: PackedGraph, molecules_right: PackedGraph
    ) -> torch.FloatTensor:
        """Run a forward pass of the DeeDDS model.

        :param context_features: A matrix of cell line features
        :param molecules_left: A matrix of left drug features
        :param molecules_right: A matrix of right drug features
        :returns: A vector of predicted synergy scores
        """
        # Run the MLP forward for the cell line features
        mlp_out = self.cell_mlp(normalize(context_features, p=2, dim=1))

        # Run the GCN forward for the drugs: GCN -> Global Max Pool -> MLP
        features_left = self._forward_molecules(molecules_left)
        features_right = self._forward_molecules(molecules_right)

        # Concatenate the output of the MLP and the GNN
        concat_in = torch.cat([mlp_out, features_left, features_right], dim=1)

        return self.final(concat_in)
