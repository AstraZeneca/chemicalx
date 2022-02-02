r"""An implementation of the MHCADDI model."""

from functools import update_wrapper
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa:N812
from torchdrug.data import PackedGraph
import numpy as np
from torchdrug.models import MessagePassingNeuralNetwork

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
    """
    A network for creating node representations based on internal message passing
    between atoms in the molecule.
    """

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
        """
        Calculate forward pass of message passing network

        seg_i and idx_j are the same length

        :param node: Nxd node feature matrix
        :param edge: Nxd edge feature matrix
        :param seg_i: List of node indices from where edges in the molecular graph start
        :param idx_j: List of node indices from where edges in the molecular graph end
        """
        edge = self.edge_proj(edge)
        msg = self.node_proj(node)
        msg = self.message_composing(msg, edge, idx_j)
        msg = self.message_aggregation(node, msg, seg_i)
        return msg

    def message_composing(self, msg, edge, idx_j):
        """
        Composes message based by elementwise multiplication of edge and node projections.
        """
        msg = msg.index_select(0, idx_j)
        msg = msg * edge
        return msg

    def message_aggregation(self, node, msg, seg_i):
        msg = torch.zeros_like(node).index_add(0, seg_i, msg)
        return msg


class CoAttention(nn.Module):
    """
    The co-attention network for MHCADDI model. It calculates attentional coefficients between the atoms
    in two different drug molecules, i.e., each atom in drug X is expressed as a linear combination
    (weighed by the attentional coefficients) of the (projected) atom features in drug Y.
    """

    def __init__(self, d_in: int, d_out: int, n_head: int = 1, dropout: float = 0.1):
        """
        Instantiate the co-attention network

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

        # Copy neighbor for attention value
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
        update_method: str = "res",
    ):
        super(CoAttentionMessagePassingNetwork, self).__init__()

        self.n_prop_step = n_prop_step

        if update_method == "res":
            x_d_node = lambda step_i: 1
            self.update_fn = lambda x, y, z: x + y + z
        elif update_method == "den":
            x_d_node = lambda step_i: 1 + 2 * step_i
            self.update_fn = lambda x, y, z: torch.cat([x, y, z], -1)
        else:
            raise NotImplementedError

        self.mps = nn.ModuleList(
            [
                MessagePassing(d_node=d_hid * x_d_node(step_i), d_edge=d_hid, d_hid=d_hid, dropout=dropout)
                for step_i in range(n_prop_step)
            ]
        )

        self.coats = nn.ModuleList(
            [
                CoAttention(d_in=d_hid * x_d_node(step_i), d_out=d_hid, n_head=n_head, dropout=dropout)
                for step_i in range(n_prop_step)
            ]
        )

        self.lns = nn.ModuleList([nn.LayerNorm(d_hid * x_d_node(step_i)) for step_i in range(n_prop_step)])

        self.pre_readout_proj = nn.Sequential(nn.Linear(d_hid * x_d_node(n_prop_step), d_readout, nn.LeakyReLU()))

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
    r"""The Multi-Head Co-Attentive Drug-Drug Interaction (MHCADDI) model from [MHCADDI]_.

    .. [MHCADDI] `Drug-Drug Adverse Effect Prediction with Graph Co-Attention
       <https://arxiv.org/pdf/1905.00534.pdf>`_
    """

    def __init__(
        self,
        d_atom_feat,
        n_atom_type,
        n_bond_type,
        d_node: int = 32,
        d_edge: int = 32,
        d_hid: int = 32,
        d_readout: int = 32,
        n_prop_step: int = 3,
        n_side_effect=None,
        n_lbls=12,
        n_head=1,
        dropout=0.1,
        update_method="res",
        score_fn="trans",
    ):
        super(MHCADDI, self).__init__()

        self.dropout = nn.Dropout(p=dropout)

        self.atom_proj = nn.Linear(d_node + d_atom_feat, d_node)
        self.atom_emb = nn.Embedding(n_atom_type, d_node, padding_idx=0)
        self.bond_emb = nn.Embedding(n_bond_type, d_edge, padding_idx=0)
        nn.init.xavier_normal_(self.atom_emb.weight)
        nn.init.xavier_normal_(self.bond_emb.weight)

        self.side_effect_emb = None
        if n_side_effect is not None:
            self.side_effect_emb = nn.Embedding(n_side_effect, d_hid)
            nn.init.xavier_normal_(self.side_effect_emb.weight)

        self.encoder = CoAttentionMessagePassingNetwork(
            d_hid=d_hid,
            d_readout=d_readout,
            n_head=n_head,
            n_prop_step=n_prop_step,
            update_method=update_method,
            dropout=dropout,
        )

        self.head_proj = nn.Linear(d_hid, d_hid, bias=False)
        self.tail_proj = nn.Linear(d_hid, d_hid, bias=False)

        nn.init.xavier_normal_(self.head_proj.weight)
        nn.init.xavier_normal_(self.tail_proj.weight)

        self.lbl_predict = nn.Linear(d_readout, n_lbls)

        self.__score_fn = score_fn

    @property
    def score_fn(self):
        return self.__score_fn

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
        se_idx=None,
        drug_se_seg=None,
    ) -> torch.FloatTensor:

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

        if self.side_effect_emb is not None:
            d1_vec = d1_vec.index_select(0, drug_se_seg)
            d2_vec = d2_vec.index_select(0, drug_se_seg)

            se_vec = self.dropout(self.side_effect_emb(se_idx))

            fwd_score = self.cal_translation_score(head=self.head_proj(d1_vec), tail=self.tail_proj(d2_vec), rel=se_vec)
            bwd_score = self.cal_translation_score(head=self.head_proj(d2_vec), tail=self.tail_proj(d1_vec), rel=se_vec)
            score = fwd_score + bwd_score

            return (score,)
        else:
            pred1 = self.lbl_predict(d1_vec)
            pred2 = self.lbl_predict(d2_vec)

            return pred1, pred2, attn1, attn2

    def atom_comp(self, atom_feat, atom_idx):
        atom_emb = self.atom_emb(atom_idx)

        node = self.atom_proj(torch.cat([atom_emb, atom_feat], -1))
        return node

    def cal_translation_score(self, head, tail, rel):
        return torch.norm(head + rel - tail, dim=1)

    def unpack(self, batch: DrugPairBatch):
        """
        Adjust drug pair batch to model design
        """

        # Left Drugs
        atom_type1 = batch.drug_molecules_left.atom_type
        bond_type1 = batch.drug_molecules_left.bond_type
        atom_feat1 = batch.drug_molecules_left.node_feature  # TODO: Drug features or atom features?
        inn_seg_i1 = batch.drug_molecules_left.edge_list[:, 0]
        inn_idx_j1 = batch.drug_molecules_left.edge_list[:, 1]
        seg_m1 = torch.tensor([], dtype=torch.int64)
        out_seg_i1 = torch.tensor([], dtype=torch.int64)
        out_idx_j1 = torch.tensor([], dtype=torch.int64)

        # Right Drugs
        atom_type2 = batch.drug_molecules_right.atom_type
        atom_feat2 = batch.drug_molecules_right.node_feature
        bond_type2 = batch.drug_molecules_right.bond_type
        inn_seg_i2 = batch.drug_molecules_right.edge_list[:, 0]
        inn_idx_j2 = batch.drug_molecules_right.edge_list[:, 1]

        seg_m2 = torch.tensor([], dtype=torch.int64)
        out_seg_i2 = torch.tensor([], dtype=torch.int64)
        out_idx_j2 = torch.tensor([], dtype=torch.int64)

        ldrug_prev_max_ix = 0
        rdrug_prev_max_ix = 0
        for i_pair in range(batch.drug_molecules_left.batch_size):

            left_drug = batch.drug_molecules_left[i_pair]
            right_drug = batch.drug_molecules_right[i_pair]

            seg = torch.tensor([i_pair], dtype=torch.int64)
            seg_m1 = torch.cat([seg_m1, seg.repeat_interleave(left_drug.num_node)])
            seg_m2 = torch.cat([seg_m2, seg.repeat_interleave(right_drug.num_node)])

            # caclulate out_seg_i1
            ldrug_out_seg = torch.tensor(range(ldrug_prev_max_ix, ldrug_prev_max_ix + left_drug.num_node))
            ldrug_prev_max_ix = ldrug_out_seg.max() + 1
            ldrug_out_seg = ldrug_out_seg.repeat_interleave(right_drug.num_node)
            out_seg_i1 = torch.cat([out_seg_i1, ldrug_out_seg])

            # caclulate out_seg_i2
            rdrug_out_seg = torch.tensor(range(rdrug_prev_max_ix, rdrug_prev_max_ix + right_drug.num_node))
            rdrug_prev_max_ix = rdrug_out_seg.max() + 1
            rdrug_out_seg = rdrug_out_seg.repeat_interleave(left_drug.num_node)
            out_seg_i2 = torch.cat([out_seg_i2, rdrug_out_seg])

            # calculate out_idx_j1
            ldrug_idx_j1 = torch.tensor(range(right_drug.num_node))
            ldrug_idx_j1 = ldrug_idx_j1.repeat_interleave(left_drug.num_node)
            out_idx_j1 = torch.cat([out_idx_j1, ldrug_idx_j1])

            # calculate out_idx_j2
            rdrug_idx_j1 = torch.tensor(range(left_drug.num_node))
            rdrug_idx_j1 = rdrug_idx_j1.repeat_interleave(right_drug.num_node)
            out_idx_j2 = torch.cat([out_idx_j2, rdrug_idx_j1])

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
