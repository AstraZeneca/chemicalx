r"""An implementation of the MHCADDI model."""

import functools
import operator

import numpy as np
import torch
import torch.nn as nn

from chemicalx.data import DrugPairBatch
from chemicalx.models import Model

__all__ = [
    "MHCADDI",
]


def segment_max(
    logit: torch.FloatTensor,
    number_of_segments: torch.LongTensor,
    segmentation_index: torch.LongTensor,
    index: torch.LongTensor,
):
    """Segmentation maximal index finder.

    :param logit: Logit vector.
    :param number_of_segments: Segment numbers.
    :param segmentation_index: Index of segments.
    :param index: Global index
    :returns: Largest index in each segmentation.
    """
    max_number_of_segments = index.max().item() + 1
    segmentation_max = logit.new_full((number_of_segments, max_number_of_segments), -np.inf)
    segmentation_max = segmentation_max.index_put_((segmentation_index, index), logit).max(dim=1)[0]
    return segmentation_max[segmentation_index]


def segment_sum(logit: torch.FloatTensor, number_of_segments: torch.LongTensor, segmentation_index: torch.LongTensor):
    """Segmentation sum calculation.

    :param logit: Logit vector.
    :param number_of_segments: Segment numbers.
    :param segmentation_index: Index of segments.
    :returns: Sum of logits on segments.
    """
    norm = logit.new_zeros(number_of_segments).index_add(0, segmentation_index, logit)
    return norm[segmentation_index]


def segment_softmax(
    logit: torch.FloatTensor,
    number_of_segments: torch.LongTensor,
    segmentation_index: torch.LongTensor,
    index: torch.LongTensor,
    temperature: torch.FloatTensor,
):
    """Segmentation softmax calculation.

    :param logit: Logit vector.
    :param number_of_segments: Segment numbers.
    :param segmentation_index: Index of segmentation.
    :param index: Global index.
    :param temperature: Normalization values.
    :returns: Probability scores for attention.
    """
    logit_max = segment_max(logit, number_of_segments, segmentation_index, index).detach()
    logit = torch.exp((logit - logit_max) / temperature)
    logit_norm = segment_sum(logit, number_of_segments, segmentation_index)
    prob = logit / (logit_norm + torch.finfo(logit_norm.dtype).eps)
    return prob


class MessagePassing(nn.Module):
    """A network for creating node representations based on internal message passing."""

    def __init__(self, node_channels: int, edge_channels: int, hidden_channels: int, dropout: float = 0.5):
        """Instantiate the MessagePassing network.

        :param node_channels: Dimension of node features
        :param edge_channels: Dimension of edge features
        :param hidden_channels: Dimension of hidden layer
        :param dropout: Dropout probability
        """
        super().__init__()
        self.node_projection = nn.Sequential(
            nn.Linear(node_channels, hidden_channels, bias=False),
            nn.Dropout(dropout),
        )
        self.edge_projection = nn.Sequential(
            nn.Linear(edge_channels, hidden_channels),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        nodes: torch.FloatTensor,
        edges: torch.FloatTensor,
        segmentation_index: torch.LongTensor,
        index: torch.LongTensor,
    ) -> torch.FloatTensor:
        """Calculate forward pass of message passing network.

        :param nodes: Node feature matrix.
        :param edges: Edge feature matrix.
        :param segmentation_index: List of node indices from where edges in the molecular graph start.
        :param index: List of node indices from where edges in the molecular graph end.
        :returns: Messages between nodes.
        """
        edges = self.edge_projection(edges)
        messages = self.node_projection(nodes)
        messages = self.message_composing(messages, edges, index)
        messages = self.message_aggregation(nodes, messages, segmentation_index)
        return messages

    def message_composing(
        self, messages: torch.FloatTensor, edges: torch.FloatTensor, index: torch.LongTensor
    ) -> torch.FloatTensor:
        """Compose message based by elementwise multiplication of edge and node projections.

        :param messages: Message matrix.
        :param edges: Edge feature matrix.
        :param index: Global node indexing.
        :returns: Composed messages.
        """
        messages = messages.index_select(0, index)
        messages = messages * edges
        return messages

    def message_aggregation(
        self, nodes: torch.FloatTensor, messages: torch.FloatTensor, segmentation_index: torch.LongTensor
    ) -> torch.FloatTensor:
        """Aggregate the messages.

        :param nodes: Node feature matrix.
        :param messages: Message feature matrix.
        :param segmentation_index: List of node indices from where edges in the molecular graph start.
        :returns: Messages between nodes.
        """
        messages = torch.zeros_like(nodes).index_add(0, segmentation_index, messages)
        return messages


class CoAttention(nn.Module):
    """The co-attention network for MHCADDI model."""

    def __init__(self, input_channels: int, output_channels: int, dropout: float = 0.1):
        """Instantiate the co-attention network.

        :param input_channels: The number of atom features.
        :param output_channels: The number of output features.
        :param dropout: Dropout probability.
        """
        super().__init__()
        self.temperature = np.sqrt(input_channels)

        self.key_projection = nn.Linear(input_channels, input_channels, bias=False)
        self.value_projection = nn.Linear(input_channels, input_channels, bias=False)

        nn.init.xavier_normal_(self.key_projection.weight)
        nn.init.xavier_normal_(self.value_projection.weight)

        self.attention_dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=1)

        self.out_projection = nn.Sequential(
            nn.Linear(input_channels, output_channels), nn.LeakyReLU(), nn.Dropout(dropout)
        )

    def forward(
        self,
        node_left: torch.FloatTensor,
        segmentation_index_left: torch.LongTensor,
        index_left: torch.LongTensor,
        node_right: torch.FloatTensor,
        segmentation_index_right: torch.LongTensor,
        index_right: torch.LongTensor,
    ):
        """Forward pass with the segmentation indices and node features.

        :param node_left: Left side node features.
        :param segmentation_index_left: Left side segmentation index.
        :param index_left: Left side indices.
        :param node_right: Right side node features.
        :param segmentation_index_right: Right side segmentation index.
        :param index_right: Right side indices.
        :returns: Left and right side messages and edge indices.
        """
        node_hidden_channels = node_left.size(1)

        segmentation_number_left = node_left.size(0)
        segmentation_number_right = node_right.size(0)

        node_left_center = self.key_projection(node_left).index_select(0, segmentation_index_left)
        node_right_center = self.key_projection(node_right).index_select(0, segmentation_index_right)

        node_left_neighbor = self.value_projection(node_right).index_select(0, segmentation_index_right)
        node_right_neighbor = self.value_projection(node_left).index_select(0, segmentation_index_left)

        translation = (node_left_center * node_right_center).sum(1)

        node_left_edge = self.attention_dropout(
            segment_softmax(
                translation, segmentation_number_left, segmentation_index_left, index_left, self.temperature
            )
        )
        node_right_edge = self.attention_dropout(
            segment_softmax(
                translation, segmentation_number_right, segmentation_index_right, index_right, self.temperature
            )
        )

        node_left_edge = node_left_edge.view(-1, 1)
        node_right_edge = node_right_edge.view(-1, 1)

        message_left = node_left.new_zeros((segmentation_number_left, node_hidden_channels)).index_add(
            0, segmentation_index_left, node_left_edge * node_left_neighbor
        )
        message_right = node_left.new_zeros((segmentation_number_right, node_hidden_channels)).index_add(
            0, segmentation_index_right, node_right_edge * node_right_neighbor
        )

        message_graph_left = self.out_projection(message_left)
        message_graph_right = self.out_projection(message_right)
        return message_graph_left, message_graph_right


class CoAttentionMessagePassingNetwork(nn.Module):
    """Coattention message passing layer."""

    def __init__(
        self,
        hidden_channels: int,
        readout_channels: int,
        dropout: float = 0.5,
    ):
        """Initialize a co-attention message passing network.

        :param hidden_channels: Input channel number.
        :param readout_channels: Readout channel number.
        :param dropout: Rate of dropout.
        """
        super().__init__()

        self.message_passing = MessagePassing(
            node_channels=hidden_channels,
            edge_channels=hidden_channels,
            hidden_channels=hidden_channels,
            dropout=dropout,
        )

        self.coattention = CoAttention(
            input_channels=hidden_channels,
            output_channels=hidden_channels,
            dropout=dropout,
        )

        self.linear = nn.LayerNorm(hidden_channels)

        # FIXME why is nn.LeakyReLU used inside init of Linear()
        self.prediction_readout_projection = nn.Linear(hidden_channels, readout_channels, nn.LeakyReLU())

    def update_fn(self, atom_features: torch.Tensor, inner_message: torch.Tensor, outer_message: torch.Tensor):
        """Update the representations."""
        return atom_features + inner_message + outer_message

    def forward(
        self,
        segmentation_molecule_left: torch.Tensor,
        atom_left: torch.Tensor,
        bond_left: torch.Tensor,
        inner_segmentation_index_left: torch.Tensor,
        inner_index_left: torch.Tensor,
        outer_segmentation_index_left: torch.Tensor,
        outer_index_left: torch.Tensor,
        segmentation_molecule_right: torch.Tensor,
        atom_right: torch.Tensor,
        bond_right: torch.Tensor,
        inner_segmentation_index_right: torch.Tensor,
        inner_index_right: torch.Tensor,
        outer_segmentation_index_right: torch.Tensor,
        outer_index_right: torch.Tensor,
    ):
        """Make a forward pass with the data.

        :param segmentation_molecule_left: Mapping from node id to graph id for the left drugs.
        :param atom_left: Atom features on the left hand side.
        :param bond_left: Bond features on the left hand side.
        :param inner_segmentation_index_left: Heads of edges connecting atoms within the left drug molecules.
        :param inner_index_left: Tails of edges connecting atoms within the left drug molecules.
        :param outer_segmentation_index_left: Heads of edges connecting atoms between left and right drug molecules
        :param outer_index_left: Tails of edges connecting atoms between left and right drug molecules.
        :param segmentation_molecule_right:  Mapping from node id to graph id for the right drugs.
        :param atom_right: Atom features on the right hand side.
        :param bond_right: Bond features on the right hand side.
        :param inner_segmentation_index_right: Heads of edges connecting atoms within the right drug molecules.
        :param inner_index_right: Tails of edges connecting atoms within the right drug molecules.
        :param outer_segmentation_index_right: Heads of edges connecting atoms between right and left drug molecules
        :param outer_index_right: Heads of edges connecting atoms between right and left drug molecules
        :returns: Graph level representations.
        """
        inner_message_left = self.message_passing(atom_left, bond_left, inner_segmentation_index_left, inner_index_left)
        inner_message_right = self.message_passing(
            atom_right, bond_right, inner_segmentation_index_right, inner_index_right
        )

        outer_message_left, outer_message_right = self.coattention(
            atom_left,
            outer_segmentation_index_left,
            outer_index_left,
            atom_right,
            outer_segmentation_index_right,
            outer_index_right,
        )

        atom_left = self.linear(self.update_fn(atom_left, inner_message_left, outer_message_left))
        atom_right = self.linear(self.update_fn(atom_right, inner_message_right, outer_message_right))

        graph_left = self.readout(atom_left, segmentation_molecule_left)
        graph_right = self.readout(atom_right, segmentation_molecule_right)

        return graph_left, graph_right

    def readout(self, atom_features: torch.Tensor, segmentation_molecule: torch.Tensor):
        """Aggregate node features.

        :param atom_features: Atom embeddings.
        :param segmentation_molecule: Molecular segmentation index.
        :returns: Graph readout vectors.
        """
        segmentation_max = segmentation_molecule.max() + 1

        atom_features = self.prediction_readout_projection(atom_features)
        hidden_channels = atom_features.size(1)

        readout_vectors = atom_features.new_zeros((segmentation_max, hidden_channels)).index_add(
            0, segmentation_molecule, atom_features
        )
        return readout_vectors


class MHCADDI(Model):
    """An implementation of the MHCADDI model from [deac2019]_.

    .. seealso:: This model was suggested in https://github.com/AstraZeneca/chemicalx/issues/13

    .. [deac2019] Deac, A., *et al.* (2019). `Drug-Drug Adverse Effect Prediction with
       Graph Co-Attention <http://arxiv.org/abs/1905.00534>`_. *arXiv*, 1905.00534.
    """

    def __init__(
        self,
        *,
        atom_feature_channels: int = 16,
        atom_type_channels: int = 16,
        bond_type_channels: int = 16,
        node_channels: int = 16,
        edge_channels: int = 16,
        hidden_channels: int = 16,
        readout_channels: int = 16,
        output_channels: int = 1,
        dropout: float = 0.5,
    ):
        """Instantiate the MHCADDI network.

        :param atom_feature_channels: Number of atom features.
        :param atom_type_channels: Number of atom types.
        :param bond_type_channels: Number of bonds.
        :param node_channels: Node feature number.
        :param edge_channels: Edge feature number.
        :param hidden_channels: Number of hidden layers.
        :param readout_channels: Readout dimensions.
        :param output_channels: Number of labels.
        :param dropout: Dropout rate.
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        self.atom_projection = nn.Linear(node_channels + atom_feature_channels, node_channels)
        self.atom_embedding = nn.Embedding(atom_type_channels, node_channels, padding_idx=0)
        self.bond_embedding = nn.Embedding(bond_type_channels, edge_channels, padding_idx=0)
        nn.init.xavier_normal_(self.atom_embedding.weight)
        nn.init.xavier_normal_(self.bond_embedding.weight)

        self.encoder = CoAttentionMessagePassingNetwork(
            hidden_channels=hidden_channels,
            readout_channels=readout_channels,
            dropout=dropout,
        )

        self.head_layer = nn.Linear(readout_channels * 2, output_channels)

    def forward(
        self,
        segmentation_molecule_left: torch.Tensor,
        atom_type_left: torch.Tensor,
        atom_features_left: torch.Tensor,
        bond_type_left: torch.Tensor,
        inner_segmentation_index_left: torch.Tensor,
        inner_index_left: torch.Tensor,
        graph_sizes_left: torch.Tensor,
        segmentation_molecule_right: torch.Tensor,
        atom_type_right: torch.Tensor,
        atom_features_right: torch.Tensor,
        bond_type_right: torch.Tensor,
        inner_segmentation_index_right: torch.Tensor,
        inner_index_right: torch.Tensor,
        graph_sizes_right: torch.Tensor,
    ) -> torch.FloatTensor:
        """Forward pass with the data.

        :param segmentation_molecule_left: Mapping from node id to graph id for the left drugs.
        :param atom_type_left: Atom types of the atoms in the left drug molecules.
        :param atom_features_left: Features of the atoms in the left drug molecules.
        :param bond_type_left: Bond types in the left drug molecules.
        :param inner_segmentation_index_left: Heads of edges connecting atoms within the left drug molecules.
        :param inner_index_left: Tails of edges connecting atoms within the left drug molecules.
        :param graph_sizes_left: Graph size vector on the left.
        :param segmentation_molecule_right:  Mapping from node id to graph id for the right drugs.
        :param atom_type_right: Atom types of the atoms in the right drug molecules.
        :param atom_features_right: Features of the atoms in the right drug molecules..
        :param bond_type_right: Bond types in the right drug molecules.
        :param inner_segmentation_index_right: Heads of edges connecting atoms within the right drug molecules.
        :param inner_index_right: Tails of edges connecting atoms within the right drug molecules.
        :param graph_sizes_right: Graph size vector on the right.
        :returns: A column vector of predicted scores.
        """
        outer_segmentation_index_left, outer_index_left = self.generate_outer_segmentation(
            graph_sizes_left, graph_sizes_right
        )
        outer_segmentation_index_right, outer_index_right = self.generate_outer_segmentation(
            graph_sizes_right, graph_sizes_left
        )

        atom_left = self.dropout(self.atom_comp(atom_features_left, atom_type_left))
        atom_right = self.dropout(self.atom_comp(atom_features_right, atom_type_right))

        bond_left = self.dropout(self.bond_embedding(bond_type_left))
        bond_right = self.dropout(self.bond_embedding(bond_type_right))

        drug_left, drug_right = self.encoder(
            segmentation_molecule_left,
            atom_left,
            bond_left,
            inner_segmentation_index_left,
            inner_index_left,
            outer_segmentation_index_left,
            outer_index_left,
            segmentation_molecule_right,
            atom_right,
            bond_right,
            inner_segmentation_index_right,
            inner_index_right,
            outer_segmentation_index_right,
            outer_index_right,
        )

        prediction_left = self.head_layer(torch.cat([drug_left, drug_right], dim=1))
        prediction_right = self.head_layer(torch.cat([drug_right, drug_left], dim=1))
        prediction_mean = (prediction_left + prediction_right) / 2
        return torch.sigmoid(prediction_mean)

    def atom_comp(self, atom_features: torch.Tensor, atom_index: torch.Tensor):
        """Compute atom projection, a linear transformation of a learned atom embedding and the atom features.

        :param atom_features: Atom input features
        :param atom_index: Index of atom type
        :returns: Node index.
        """
        atom_embedding = self.atom_embedding(atom_index)
        node_embedding = self.atom_projection(torch.cat([atom_embedding, atom_features], -1))
        return node_embedding

    def generate_outer_segmentation(self, graph_sizes_left: torch.LongTensor, graph_sizes_right: torch.LongTensor):
        """Calculate all pairwise edges between the atoms in a set of drug pairs.

        Example: Given two sets of drug sizes:

        graph_sizes_left = torch.tensor([1, 2])
        graph_sizes_right = torch.tensor([3, 4])

        Here the drug pairs have sizes (1,3) and (2,4)

        This results in:

        outer_segmentation_index = tensor([0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
        outer_index = tensor([0, 1, 2, 3, 4, 5, 6, 3, 4, 5, 6])

        :param graph_sizes_left: List of graph sizes in the left drug batch.
        :param graph_sizes_right: List of graph sizes in the right drug batch.
        :returns: Edge indices.
        """
        interactions = graph_sizes_left * graph_sizes_right

        left_shifted_graph_size_cum_sum = torch.cumsum(graph_sizes_left, 0) - graph_sizes_left
        shift_sums_left = torch.repeat_interleave(left_shifted_graph_size_cum_sum, interactions)
        outer_segmentation_index = [
            np.repeat(np.array(range(0, left_graph_size)), right_graph_size)
            for left_graph_size, right_graph_size in zip(graph_sizes_left, graph_sizes_right)
        ]
        outer_segmentation_index = functools.reduce(operator.iconcat, outer_segmentation_index, [])
        outer_segmentation_index = torch.tensor(outer_segmentation_index) + shift_sums_left

        right_shifted_graph_size_cum_sum = torch.cumsum(graph_sizes_right, 0) - graph_sizes_right
        shift_sums_right = torch.repeat_interleave(right_shifted_graph_size_cum_sum, interactions)
        outer_index = [
            list(range(0, right_graph_size)) * left_graph_size
            for left_graph_size, right_graph_size in zip(graph_sizes_left, graph_sizes_right)
        ]
        outer_index = functools.reduce(operator.iconcat, outer_index, [])
        outer_index = torch.tensor(outer_index) + shift_sums_right
        return outer_segmentation_index, outer_index

    def unpack(self, batch: DrugPairBatch):
        """Adjust drug pair batch to model design.

        :param batch: Molecular data in a drug pair batch.
        :returns: Tuple of data.
        """
        assert batch.drug_molecules_left is not None
        assert batch.drug_molecules_right is not None

        segmentation_molecule_left = batch.drug_molecules_left.node2graph
        atom_type_left = batch.drug_molecules_left.atom_type
        atom_features_left = batch.drug_molecules_left.node_feature
        bond_type_left = batch.drug_molecules_left.bond_type
        inner_segmentation_index_left = batch.drug_molecules_left.edge_list[:, 0]
        inner_index_left = batch.drug_molecules_left.edge_list[:, 1]
        graph_sizes_left = batch.drug_molecules_left.num_nodes

        segmentation_molecule_right = batch.drug_molecules_right.node2graph
        atom_type_right = batch.drug_molecules_right.atom_type
        atom_features_right = batch.drug_molecules_right.node_feature
        bond_type_right = batch.drug_molecules_right.bond_type
        inner_segmentation_index_right = batch.drug_molecules_right.edge_list[:, 0]
        inner_index_right = batch.drug_molecules_right.edge_list[:, 1]
        graph_sizes_right = batch.drug_molecules_right.num_nodes

        return (
            segmentation_molecule_left,
            atom_type_left,
            atom_features_left,
            bond_type_left,
            inner_segmentation_index_left,
            inner_index_left,
            graph_sizes_left,
            segmentation_molecule_right,
            atom_type_right,
            atom_features_right,
            bond_type_right,
            inner_segmentation_index_right,
            inner_index_right,
            graph_sizes_right,
        )
