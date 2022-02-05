"""An implementation of the SSI-DDI model."""

from .base import UnimplementedModel
import torch
from torch import nn
from torch.nn import Parameter, Linear
from torch.nn.modules.container import ModuleList
from torch import Tensor

from torchdrug.data import PackedGraph
from torchdrug.layers import GraphAttentionConv

from torch_scatter import scatter, scatter_add, scatter_max
from chemicalx.constants import TORCHDRUG_NODE_FEATURES
from chemicalx.data import DrugPairBatch
from chemicalx.models import Model
import torch.nn.functional as F


__all__ = [
    "SSIDDI",
]

def filter_adj(edge_index, edge_attr, perm, num_nodes):
    mask = perm.new_full((num_nodes, ), -1)
    i = torch.arange(perm.size(0), dtype=torch.long, device=perm.device)
    mask[perm] = i

    row, col = edge_index
    row, col = mask[row], mask[col]
    mask = (row >= 0) & (col >= 0)
    row, col = row[mask], col[mask]

    if edge_attr is not None:
        edge_attr = edge_attr[mask]

    return torch.stack([row, col], dim=0), edge_attr


def topk(x, ratio, batch, min_score=None, tol=1e-7):
    if min_score is not None:
        # Make sure that we do not drop all nodes in a graph.
        scores_max = scatter_max(x, batch)[0].index_select(0, batch) - tol
        scores_min = scores_max.clamp(max=min_score)

        perm = (x > scores_min).nonzero(as_tuple=False).view(-1)
    else:
        num_nodes = scatter_add(batch.new_ones(x.size(0)), batch, dim=0)
        batch_size, max_num_nodes = num_nodes.size(0), num_nodes.max().item()

        cum_num_nodes = torch.cat(
            [num_nodes.new_zeros(1),
             num_nodes.cumsum(dim=0)[:-1]], dim=0)

        index = torch.arange(batch.size(0), dtype=torch.long, device=x.device)
        index = (index - cum_num_nodes[batch]) + (batch * max_num_nodes)

        dense_x = x.new_full((batch_size * max_num_nodes, ),
                             torch.finfo(x.dtype).min)
        dense_x[index] = x
        dense_x = dense_x.view(batch_size, max_num_nodes)

        _, perm = dense_x.sort(dim=-1, descending=True)

        perm = perm + cum_num_nodes.view(-1, 1)
        perm = perm.view(-1)

        if isinstance(ratio, int):
            k = num_nodes.new_full((num_nodes.size(0), ), ratio)
            k = torch.min(k, num_nodes)
        else:
            k = (ratio * num_nodes.to(torch.float)).ceil().to(torch.long)

        mask = [
            torch.arange(k[i], dtype=torch.long, device=x.device) +
            i * max_num_nodes for i in range(batch_size)
        ]
        mask = torch.cat(mask, dim=0)

        perm = perm[mask]

    return perm


def softmax(src, batch):
    num_nodes = batch.shape[0]

    out = src - scatter_max(src, batch, dim=0, dim_size=num_nodes)[0][batch]
    out = out.exp()
    out = out / (scatter_add(out, batch, dim=0,
                             dim_size=num_nodes)[batch] + 1e-16)

    return out


def propagate(x, edge_index, edge_weight=None):

    row, col = edge_index

    if(edge_weight == None):
        edge_weight = torch.ones_like(row)

    out = edge_weight.view(-1, 1) * x[row]

    return scatter(out, col, dim=0, dim_size=x.size(0), reduce='add')


class GraphConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, aggr='add', bias=True):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_l = Linear(in_channels[0], out_channels, bias=bias)
        self.lin_r = Linear(in_channels[1], out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None, size=None):
        """"""
        if isinstance(x, Tensor):
            x = (x, x)

        out = propagate(x=x[0], edge_index=edge_index,
                        edge_weight=edge_weight)
        out = self.lin_l(out)

        x_r = x[1]
        if x_r is not None:
            out += self.lin_r(x_r)

        return out


class SAGPooling(torch.nn.Module):
    def __init__(self, in_channels, min_score, ratio=0.5, multiplier=1.0, nonlinearity=torch.tanh):
        super().__init__()

        self.in_channels = in_channels
        self.ratio = ratio
        self.gnn = GraphConv(in_channels, 1)  # need to implement
        self.min_score = min_score
        self.multiplier = multiplier
        self.nonlinearity = nonlinearity

        self.reset_parameters()

    def reset_parameters(self):
        self.gnn.reset_parameters()

    def forward(self, x, edge_index, edge_attr=None, batch=None, attn=None):
        """"""
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))

        attn = x if attn is None else attn
        attn = attn.unsqueeze(-1) if attn.dim() == 1 else attn

        score = self.gnn(attn, edge_index).view(-1)

        if self.min_score is None:
            score = self.nonlinearity(score)
        else:
            score = softmax(score, batch)

        perm = topk(score, self.ratio, batch, self.min_score)
        x = x[perm] * score[perm].view(-1, 1)
        x = self.multiplier * x if self.multiplier != 1 else x

        batch = batch[perm]
        edge_index, edge_attr = filter_adj(
            edge_index, edge_attr, perm, num_nodes=score.size(0))

        return x, batch


class LayerNorm(nn.Module):
    def __init__(self, in_channels, eps=1e-5, affine=True):
        super().__init__()

        self.in_channels = in_channels
        self.eps = eps

        if affine:
            self.weight = Parameter(torch.Tensor([in_channels]))
            self.bias = Parameter(torch.Tensor([in_channels]))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)
        torch.nn.init.zeros_(self.bias)

    def forward(self, x, batch):
        """"""
        if batch is None:
            x = x - x.mean()
            out = x / (x.std(unbiased=False) + self.eps)

        else:
            batch_size = int(batch.max()) + 1

            out = torch.zeros((batch_size, ), dtype=x.dtype,
                              device=batch.device)
            one = torch.ones((batch.size(0), ),
                             dtype=out.dtype, device=out.device)
            norm = out.scatter_add_(0, batch, one).clamp_(min=1)
            norm = norm.mul_(x.size(-1)).view(-1, 1)

            mean = scatter(x, batch, dim=0, dim_size=batch_size,
                           reduce='add').sum(dim=-1, keepdim=True) / norm

            x = x - mean.index_select(0, batch)

            var = scatter(x * x, batch, dim=0, dim_size=batch_size,
                          reduce='add').sum(dim=-1, keepdim=True)
            var = var / norm

            out = x / (var + self.eps).sqrt().index_select(0, batch)

        if self.weight is not None and self.bias is not None:
            out = out * self.weight + self.bias

        return out


class RESCAL(nn.Module):
    def __init__(self, n_features, n_rels=1):
        super().__init__()
        self.n_rels = n_rels
        self.n_features = n_features
        self.rel_emb = nn.Embedding(self.n_rels, n_features * n_features)
        nn.init.xavier_uniform_(self.rel_emb.weight)

    def forward(self, heads, tails, alpha_scores):
        # by default we have only one interaction
        rels = self.rel_emb(torch.tensor(0))
        rels = F.normalize(rels, dim=-1)
        heads = F.normalize(heads, dim=-1)
        tails = F.normalize(tails, dim=-1)
        rels = rels.view(-1, self.n_features, self.n_features)

        scores = heads @ rels @ tails.transpose(-2, -1)

        if alpha_scores is not None:
            scores = alpha_scores * scores
        scores = scores.sum(dim=(-2, -1))

        return scores


class CoAttentionLayer(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.n_features = n_features
        self.w_q = nn.Parameter(torch.zeros(n_features, n_features // 2))
        self.w_k = nn.Parameter(torch.zeros(n_features, n_features // 2))
        self.bias = nn.Parameter(torch.zeros(n_features // 2))
        self.a = nn.Parameter(torch.zeros(n_features // 2))

        nn.init.xavier_uniform_(self.w_q)
        nn.init.xavier_uniform_(self.w_k)
        nn.init.xavier_uniform_(self.bias.view(*self.bias.shape, -1))
        nn.init.xavier_uniform_(self.a.view(*self.a.shape, -1))

    def forward(self, receiver, attendant):
        keys = receiver @ self.w_k
        queries = attendant @ self.w_q

        values = receiver

        e_activations = queries.unsqueeze(-3) + keys.unsqueeze(-2) + self.bias
        e_scores = torch.tanh(e_activations) @ self.a

        attentions = e_scores

        return attentions


class SSIDDI_block(nn.Module):
    # single head out dimension = out/n_heads
    def __init__(self, n_heads: int, in_feats: int, out_feats: int):
        super().__init__()

        self.n_heads = n_heads
        self.in_feats = in_feats
        self.out_feats = out_feats

        self.conv = GraphAttentionConv(
            input_dim=in_feats, output_dim=out_feats, num_head=n_heads)

        self.readout = SAGPooling(out_feats, min_score=-1)

    def forward(self, molecules: PackedGraph):
        molecules.node_feature = self.conv(molecules, molecules.node_feature)

        att_h_nodes, att_batch = self.readout(
            molecules.node_feature, molecules.edge_list[:, :2].t(), batch=molecules.node2graph)

        h_graphs = scatter(att_h_nodes, att_batch, dim=0, dim_size=int(
            att_batch.max().item() + 1), reduce='add')
        
        return molecules, h_graphs
    

class SSIDDI(UnimplementedModel):
    """An implementation of the SSI-DDI model from [nyamabo2021]_.

    .. seealso:: This model was suggested in https://github.com/AstraZeneca/chemicalx/issues/11

    .. [nyamabo2021] Nyamabo, A. K., *et al.* (2021). `SSI–DDI: substructure–substructure interactions
       for drug–drug interaction prediction <https://doi.org/10.1093/bib/bbab133>`_.
       *Briefings in Bioinformatics*, 22(6).
    """
    def __init__(
            self,
            *,
            molecule_channels: int = TORCHDRUG_NODE_FEATURES,
            hidden_channels: list = [64, 64, 64, 64],
            n_heads: list = [2, 2, 2, 2],
    ):
        """Instantiate the SSIDDI model.

        :param molecule_channels: The number of molecular features.
        :param hidden_channels: The list of layer neurons for each hidden layer
        :param n_heads: The list of layer neurons for each hidden layer
        """
        super().__init__()
        self.in_feat = molecule_channels
        self.initial_norm = LayerNorm(molecule_channels)

        self.blocks = []
        self.net_norms = ModuleList()

        in_feat = molecule_channels
        for i, (hidden_channel, n_head) in enumerate(zip(hidden_channels, n_heads)):
            block = SSIDDI_block(n_head, in_feat, hidden_channel)
            self.add_module(f'block{i}', block)
            self.blocks.append(block)
            # here the hidden channel = n_heads * single_head_hidden_channel
            self.net_norms.append(LayerNorm(hidden_channel))
            in_feat = hidden_channel

        self.co_attention = CoAttentionLayer(in_feat)
        # we only have one interaction type
        self.KGE = RESCAL(in_feat, n_rels=1)

    def unpack(self, batch: DrugPairBatch):
        """Return the left molecular graph and right molecular graph."""

        return (
            batch.drug_molecules_left,
            batch.drug_molecules_right,
        )

    def forward(self, molecules_left: PackedGraph, molecules_right: PackedGraph) -> torch.FloatTensor:
        """Run a forward pass of the EPGCN-DS model.

        :param molecules_left: Batched molecules for the left side drugs.
        :param molecules_right: Batched molecules for the right side drugs.
        :returns: A column vector of predicted synergy scores.
        """

        molecules_left.node_feature = self.initial_norm(
            molecules_left.node_feature, molecules_left.node2graph)
        molecules_right.node_feature = self.initial_norm(
            molecules_right.node_feature, molecules_right.node2graph)

        repr_l, repr_r = [], []

        for i, block in enumerate(self.blocks):
            out1, out2 = block(molecules_left), block(molecules_right)

            molecules_left, molecules_right = out1[0], out2[0]
            h_graph_l, h_graph_r = out1[1], out2[1]

            repr_l.append(h_graph_l)
            repr_r.append(h_graph_r)

            molecules_left.node_feature = F.elu(self.net_norms[i](
                molecules_left.node_feature, molecules_left.node2graph))
            molecules_right.node_feature = F.elu(self.net_norms[i](
                molecules_right.node_feature, molecules_right.node2graph))

        repr_l = torch.stack(repr_l, dim=-2)
        repr_r = torch.stack(repr_r, dim=-2)

        kge_heads = repr_l
        kge_tails = repr_r

        attentions = self.co_attention(kge_heads, kge_tails)

        scores = torch.sigmoid(self.KGE(kge_heads, kge_tails, attentions))

        return scores.view(-1, 1)
