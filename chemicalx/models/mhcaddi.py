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


def segment_max(logit, n_seg, seg_i, idx_j):
    max_seg_numel = idx_j.max().item() + 1
    seg_max = logit.new_full((n_seg, max_seg_numel), -np.inf)
    seg_max = seg_max.index_put_((seg_i, idx_j), logit).max(dim=1)[0]
    return seg_max[seg_i]


def segment_sum(logit, n_seg, seg_i):
    norm = logit.new_zeros(n_seg).index_add(0, seg_i, logit)
    return norm[seg_i]


def segment_softmax(logit, n_seg, seg_i, idx_j, temperature):
    logit_max = segment_max(logit, n_seg, seg_i, idx_j).detach()
    logit = torch.exp((logit - logit_max) / temperature)
    logit_norm = segment_sum(logit, n_seg, seg_i)
    prob = logit / (logit_norm + 1e-8)
    return prob


def segment_multihead_expand(seg_i, n_seg, n_head):
    i_head_shift = n_seg * seg_i.new_tensor(torch.arange(n_head))
    seg_i = (seg_i.view(-1, 1) + i_head_shift.view(1, -1)).view(-1)
    return seg_i


class MessagePassing(nn.Module):
    """A network for creating node representations based on internal message passing."""

    def __init__(self, d_node: int, d_edge: int, d_hid: int, dropout=0.1):
        """Instantiate the MessagePassing network.

        :param d_node: Dimension of node features
        :param d_edge: Dimension of edge features
        :param d_hid: Dimension of hidden layer
        :param dropout: Dropout probability
        """
        super(MessagePassing, self).__init__()

        dropout = nn.Dropout(p=dropout)

        self.node_proj = nn.Sequential(nn.Linear(d_node, d_hid, bias=False), dropout)
        self.edge_proj = nn.Sequential(
            nn.Linear(d_edge, d_hid), nn.LeakyReLU(), dropout, nn.Linear(d_hid, d_hid), nn.LeakyReLU(), dropout
        )
        self.msg_proj = nn.Sequential(
            nn.Linear(d_hid, d_hid), nn.LeakyReLU(), dropout, nn.Linear(d_hid, d_hid), dropout
        )

    def forward(self, node: torch.Tensor, edge: torch.Tensor, seg_i: torch.Tensor, idx_j: torch.Tensor):
        """Calculate forward pass of message passing network.

        :param node: N x d node feature matrix.
        :param edge: N x d edge feature matrix.
        :param seg_i: List of node indices from where edges in the molecular graph start.
        :param idx_j: List of node indices from where edges in the molecular graph end.
        :returns: Message between nodes.
        """
        edge = self.edge_proj(edge)
        msg = self.node_proj(node)
        msg = self.message_composing(msg, edge, idx_j)
        msg = self.message_aggregation(node, msg, seg_i)
        return msg

    def message_composing(self, msg, edge, idx_j):
        """Composes message based by elementwise multiplication of edge and node projections."""
        msg = msg.index_select(0, idx_j)
        msg = msg * edge
        return msg

    def message_aggregation(self, node, msg, seg_i):
        msg = torch.zeros_like(node).index_add(0, seg_i, msg)
        return msg


class CoAttention(nn.Module):
    """The co-attention network for MHCADDI model."""

    def __init__(self, d_in: int, d_out: int, n_head: int = 1, dropout: float = 0.1):
        """Instantiate the co-attention network.

        :param d_in: The number of atom features
        :param d_out: The number of output features
        :param n_head: The number of attention heads
        :param dropout: Dropout probability
        """
        super(CoAttention, self).__init__()
        self.temperature = np.sqrt(d_in)
        self.n_head = n_head
        self.multi_head = self.n_head > 1

        self.key_proj = nn.Linear(d_in, d_in * n_head, bias=False)
        self.val_proj = nn.Linear(d_in, d_in * n_head, bias=False)

        nn.init.xavier_normal_(self.key_proj.weight)
        nn.init.xavier_normal_(self.val_proj.weight)

        self.attn_drop = nn.Dropout(p=dropout)
        self.softmax = nn.Softmax(dim=1)

        self.out_proj = nn.Sequential(nn.Linear(d_in * n_head, d_out), nn.LeakyReLU(), nn.Dropout(p=dropout))

    def forward(self, node1, seg_i1, idx_j1, node2, seg_i2, idx_j2):

        d_h = node1.size(1)

        n_seg1 = node1.size(0)
        n_seg2 = node2.size(0)

        # Copy center for attention key
        node1_ctr = self.key_proj(node1).index_select(0, seg_i1)
        node2_ctr = self.key_proj(node2).index_select(0, seg_i2)

        # Copy neighbour for attention value
        node1_nbr = self.val_proj(node2).index_select(0, seg_i2)  # idx_j1 == seg_i2
        node2_nbr = self.val_proj(node1).index_select(0, seg_i1)  # idx_j2 == seg_i1

        arg_i1 = None
        arg_i2 = None

        if self.multi_head:
            # prepare copied and shifted index tensors
            seg_i1 = segment_multihead_expand(seg_i1, n_seg1, self.n_head)
            seg_i2 = segment_multihead_expand(seg_i2, n_seg2, self.n_head)

            idx_j1 = idx_j1.unsqueeze(1).expand(-1, self.n_head).contiguous().view(-1)
            idx_j2 = idx_j2.unsqueeze(1).expand(-1, self.n_head).contiguous().view(-1)

            arg_i1 = segment_multihead_expand(seg_i1.new_tensor(np.arange(n_seg1)), n_seg1, self.n_head)
            arg_i2 = segment_multihead_expand(seg_i2.new_tensor(np.arange(n_seg2)), n_seg2, self.n_head)

            # pile up as regular input
            node1_ctr = node1_ctr.view(-1, d_h)
            node2_ctr = node2_ctr.view(-1, d_h)

            node1_nbr = node1_nbr.view(-1, d_h)
            node2_nbr = node2_nbr.view(-1, d_h)

            n_seg1 = n_seg1 * self.n_head
            n_seg2 = n_seg2 * self.n_head

        translation = (node1_ctr * node2_ctr).sum(1)

        node1_edge = self.attn_drop(segment_softmax(translation, n_seg1, seg_i1, idx_j1, self.temperature))
        node2_edge = self.attn_drop(segment_softmax(translation, n_seg2, seg_i2, idx_j2, self.temperature))

        node1_edge = node1_edge.view(-1, 1)
        node2_edge = node2_edge.view(-1, 1)

        # Weighted sum
        msg1 = node1.new_zeros((n_seg1, d_h)).index_add(0, seg_i1, node1_edge * node1_nbr)
        msg2 = node2.new_zeros((n_seg2, d_h)).index_add(0, seg_i2, node2_edge * node2_nbr)

        if self.multi_head:
            msg1 = msg1[arg_i1].view(-1, d_h * self.n_head)
            msg2 = msg2[arg_i2].view(-1, d_h * self.n_head)

        msg1 = self.out_proj(msg1)
        msg2 = self.out_proj(msg2)
        return msg1, msg2, node1_edge, node2_edge


class CoAttentionMessagePassingNetwork(nn.Module):
    def __init__(
        self,
        d_hid: int,
        d_readout: int,
        n_prop_step: int,
        n_head: int = 1,
        dropout: float = 0.1,
    ):
        super(CoAttentionMessagePassingNetwork, self).__init__()

        self.n_prop_step = n_prop_step

        self.mps = nn.ModuleList(
            [
                MessagePassing(d_node=d_hid * self.x_d_node(step_i), d_edge=d_hid, d_hid=d_hid, dropout=dropout)
                for step_i in range(n_prop_step)
            ]
        )

        self.coats = nn.ModuleList(
            [
                CoAttention(d_in=d_hid * self.x_d_node(step_i), d_out=d_hid, n_head=n_head, dropout=dropout)
                for step_i in range(n_prop_step)
            ]
        )

        self.lns = nn.ModuleList([nn.LayerNorm(d_hid * self.x_d_node(step_i)) for step_i in range(n_prop_step)])

        self.pre_readout_proj = nn.Sequential(nn.Linear(d_hid * self.x_d_node(n_prop_step), d_readout, nn.LeakyReLU()))

    def x_d_node(self, x):
        return 1

    def update_fn(self, x, y, z):
        return x + y + z

    def forward(
        self,
        seg_g1,
        node1,
        edge1,
        inn_seg_i1,
        inn_idx_j1,
        out_seg_i1,
        out_idx_j1,
        seg_g2,
        node2,
        edge2,
        inn_seg_i2,
        inn_idx_j2,
        out_seg_i2,
        out_idx_j2,
    ):
        for step_i in range(self.n_prop_step):

            inner_msg1 = self.mps[step_i](node1, edge1, inn_seg_i1, inn_idx_j1)
            inner_msg2 = self.mps[step_i](node2, edge2, inn_seg_i2, inn_idx_j2)

            outer_msg1, outer_msg2, attn1, attn2 = self.coats[step_i](
                node1, out_seg_i1, out_idx_j1, node2, out_seg_i2, out_idx_j2
            )

            node1 = self.lns[step_i](self.update_fn(node1, inner_msg1, outer_msg1))
            node2 = self.lns[step_i](self.update_fn(node2, inner_msg2, outer_msg2))

        g1_vec = self.readout(node1, seg_g1)
        g2_vec = self.readout(node2, seg_g2)

        return g1_vec, g2_vec, attn1, attn2

    def readout(self, node, seg_g):
        sz_b = seg_g.max() + 1

        node = self.pre_readout_proj(node)
        d_h = node.size(1)

        encv = node.new_zeros((sz_b, d_h)).index_add(0, seg_g, node)
        return encv


class MHCADDI(Model):
    """An implementation of the MHCADDI model from [deac2019]_.

    .. seealso:: This model was suggested in https://github.com/AstraZeneca/chemicalx/issues/13

    .. [deac2019] Deac, A., *et al.* (2019). `Drug-Drug Adverse Effect Prediction with
       Graph Co-Attention <http://arxiv.org/abs/1905.00534>`_. *arXiv*, 1905.00534.
    """

    def __init__(
        self,
        *,
        d_atom_feat: int = 16,
        n_atom_type: int = 16,
        n_bond_type: int = 16,
        d_node: int = 16,
        d_edge: int = 16,
        d_hid: int = 16,
        d_readout: int = 16,
        n_prop_step: int = 1,
        n_lbls=1,
        n_head=1,
        dropout=0.5,
    ):
        """Instantiate the MHCADDI network.

        :param d_atom_feat: Number of atom features.
        :param n_atom_type: Number of atom types.
        :param n_bond_type: Number of bonds.
        :param d_node: Node feature number.
        :param d_edge: Edge feature number.
        :param d_hid: Number of hidden layers.
        :param d_readout: Readout dimensions.
        :param n_prop_step: GCN depth.
        :param n_lbls: Number of labels.
        :param n_head: Number of scoring head.
        :param dropout: Dropout rate.
        """
        super(MHCADDI, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        self.atom_proj = nn.Linear(d_node + d_atom_feat, d_node)
        self.atom_emb = nn.Embedding(n_atom_type, d_node, padding_idx=0)
        self.bond_emb = nn.Embedding(n_bond_type, d_edge, padding_idx=0)
        nn.init.xavier_normal_(self.atom_emb.weight)
        nn.init.xavier_normal_(self.bond_emb.weight)

        self.encoder = CoAttentionMessagePassingNetwork(
            d_hid=d_hid,
            d_readout=d_readout,
            n_head=n_head,
            n_prop_step=n_prop_step,
            dropout=dropout,
        )

        self.lbl_predict = nn.Linear(d_readout * 2, n_lbls)

    def forward(
        self,
        seg_m1,
        atom_type1,
        atom_feat1,
        bond_type1,
        inn_seg_i1,
        inn_idx_j1,
        out_seg_i1,
        out_idx_j1,
        seg_m2,
        atom_type2,
        atom_feat2,
        bond_type2,
        inn_seg_i2,
        inn_idx_j2,
        out_seg_i2,
        out_idx_j2,
    ) -> torch.FloatTensor:
        """
        :param seg_m1: Mapping from node id to graph id for the left drugs.
        :param atom_type1: Atom types of the atoms in the left drug molecules.
        :param atom_feat1: Features of the atoms in the left drug molecules.
        :param bond_type1: Bond types in the left drug molecules.
        :param inn_seg_i1: Heads of edges connecting atoms within the left drug molecules.
        :param inn_idx_j1: Tails of edges connecting atoms within the left drug molecules.
        :param out_seg_i1: Heads of edges connecting atoms between left and right drug molecules
        :param out_idx_j1: Tails of edges connecting atoms between left and right drug molecules.
        :param seg_m2:  Mapping from node id to graph id for the right drugs.
        :param atom_type2: Atom types of the atoms in the right drug molecules.
        :param atom_feat2: Features of the atoms in the right drug molecules..
        :param bond_type2: Bond types in the right drug molecules.
        :param inn_seg_i2: Heads of edges connecting atoms within the right drug molecules.
        :param inn_idx_j2: Tails of edges connecting atoms within the right drug molecules.
        :param out_seg_i2: Heads of edges connecting atoms between right and left drug molecules
        :param out_idx_j2: Heads of edges connecting atoms between right and left drug molecules
        :returns: A column vector of predicted scores.
        """
        atom1 = self.dropout(self.atom_comp(atom_feat1, atom_type1))
        atom2 = self.dropout(self.atom_comp(atom_feat2, atom_type2))

        bond1 = self.dropout(self.bond_emb(bond_type1))
        bond2 = self.dropout(self.bond_emb(bond_type2))

        d1_vec, d2_vec, attn1, attn2 = self.encoder(
            seg_m1,
            atom1,
            bond1,
            inn_seg_i1,
            inn_idx_j1,
            out_seg_i1,
            out_idx_j1,
            seg_m2,
            atom2,
            bond2,
            inn_seg_i2,
            inn_idx_j2,
            out_seg_i2,
            out_idx_j2,
        )

        pred1 = self.lbl_predict(torch.cat([d1_vec, d2_vec], dim=1))
        pred2 = self.lbl_predict(torch.cat([d2_vec, d1_vec], dim=1))
        return torch.sigmoid((pred1 + pred2) / 2)

    def atom_comp(self, atom_feat, atom_idx):
        """Computes atom projection, a linear transformation of a learned atom embedding and the atom features
        :param atom_feat: Atom input features
        :param atom_idx: Index of atom type
        :returns: Node index.
        """
        atom_emb = self.atom_emb(atom_idx)

        node = self.atom_proj(torch.cat([atom_emb, atom_feat], -1))
        return node

    def generate_outer_segmentation(self, graph_sizes_left, graph_sizes_right):
        """Calculates all pairwise edges between the atoms in a set of drug pairs.
        Example: Given two sets of drug sizes:

        graph_sizes_left = torch.tensor([1, 2])
        graph_sizes_right = torch.tensor([3, 4])

        Here the drug pairs have sizes (1,3) and (2,4)

        This results in:

        out_seg_i = tensor([0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
        out_idx_j = tensor([0, 1, 2, 3, 4, 5, 6, 3, 4, 5, 6])

        :param graph_sizes_left: List of graph sizes in the left drug batch.
        :param graph_sizes_right: List of graph sizes in the right drug batch.
        :returns: Edge indices
        """
        interactions = graph_sizes_left * graph_sizes_right

        left_shifted_graph_size_cum_sum = torch.cumsum(graph_sizes_left, 0) - graph_sizes_left
        shift_sums_left = torch.repeat_interleave(left_shifted_graph_size_cum_sum, interactions)
        out_seg_i = [
            np.repeat(np.array(range(0, left_graph_size)), right_graph_size)
            for left_graph_size, right_graph_size in zip(graph_sizes_left, graph_sizes_right)
        ]
        out_seg_i = functools.reduce(operator.iconcat, out_seg_i, [])
        out_seg_i = torch.tensor(out_seg_i) + shift_sums_left

        right_shifted_graph_size_cum_sum = torch.cumsum(graph_sizes_right, 0) - graph_sizes_right
        shift_sums_right = torch.repeat_interleave(right_shifted_graph_size_cum_sum, interactions)
        out_idx_j = [
            list(range(0, right_graph_size)) * left_graph_size
            for left_graph_size, right_graph_size in zip(graph_sizes_left, graph_sizes_right)
        ]
        out_idx_j = functools.reduce(operator.iconcat, out_idx_j, [])
        out_idx_j = torch.tensor(out_idx_j) + shift_sums_right

        return out_seg_i, out_idx_j

    def unpack(self, batch: DrugPairBatch):
        """Adjust drug pair batch to model design.

        :param batch: Molecular data in a drug pair batch.
        :returns: Tuple of data.
        """
        atom_type1 = batch.drug_molecules_left.atom_type
        bond_type1 = batch.drug_molecules_left.bond_type
        atom_feat1 = batch.drug_molecules_left.node_feature
        inn_seg_i1 = batch.drug_molecules_left.edge_list[:, 0]
        inn_idx_j1 = batch.drug_molecules_left.edge_list[:, 1]

        atom_type2 = batch.drug_molecules_right.atom_type
        atom_feat2 = batch.drug_molecules_right.node_feature
        bond_type2 = batch.drug_molecules_right.bond_type
        inn_seg_i2 = batch.drug_molecules_right.edge_list[:, 0]
        inn_idx_j2 = batch.drug_molecules_right.edge_list[:, 1]

        seg_m1 = batch.drug_molecules_left.node2graph
        seg_m2 = batch.drug_molecules_right.node2graph

        out_seg_i1, out_idx_j1 = self.generate_outer_segmentation(
            batch.drug_molecules_left.num_nodes, batch.drug_molecules_right.num_nodes
        )
        out_seg_i2, out_idx_j2 = self.generate_outer_segmentation(
            batch.drug_molecules_right.num_nodes, batch.drug_molecules_left.num_nodes
        )
        return (
            seg_m1,
            atom_type1,
            atom_feat1,
            bond_type1,
            inn_seg_i1,
            inn_idx_j1,
            out_seg_i1,
            out_idx_j1,
            seg_m2,
            atom_type2,
            atom_feat2,
            bond_type2,
            inn_seg_i2,
            inn_idx_j2,
            out_seg_i2,
            out_idx_j2,
        )
