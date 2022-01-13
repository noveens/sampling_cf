''' Credit https://github.com/hengruizhang98/InfoGraph & https://github.com/fanyun-sun/InfoGraph '''

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, ModuleList, Linear, ReLU, BatchNorm1d

from dgl.nn import GINConv
from dgl.nn.pytorch.glob import SumPooling, AvgPooling, SortPooling

from data_genie.InfoGraph.infograph_utils import local_global_loss_

''' Feedforward neural network'''
class FeedforwardNetwork(nn.Module):

    '''
    3-layer feed-forward neural networks with jumping connections
    Parameters
    -----------
    in_dim: int, Input feature size.
    hid_dim: int, Hidden feature size.

    Functions
    -----------
    forward(feat):
        feat: Tensor, [N * D], input features
    '''

    def __init__(self, in_dim, hid_dim):
        super(FeedforwardNetwork, self).__init__()

        self.block = Sequential(
            Linear(in_dim, hid_dim),
            ReLU(),
            Linear(hid_dim, hid_dim),
            ReLU(),
            Linear(hid_dim, hid_dim),
            ReLU()
        )

        self.jump_con = Linear(in_dim, hid_dim)

    def forward(self, feat):
        block_out = self.block(feat)
        jump_out = self.jump_con(feat)

        out = block_out + jump_out

        return out


''' Unsupervised Setting '''
class GINEncoder(nn.Module):
    '''
    Encoder based on dgl.nn.GINConv &  dgl.nn.SortPooling
    Parameters
    -----------
    in_dim: int, Input feature size.
    hid_dim: int, Hidden feature size.
    n_layer: int, number of GIN layers.

    Functions
    -----------
    forward(graph, feat):
        graph: dgl.Graph,
        feat: Tensor, [N * D], node features
    '''

    def __init__(self, in_dim, hid_dim, n_layer):
        super(GINEncoder, self).__init__()

        self.n_layer = n_layer

        self.convs = ModuleList()
        self.bns = ModuleList()

        for i in range(n_layer):
            if i == 0: n_in = in_dim
            else: n_in = hid_dim
            
            n_out = hid_dim
            block = Sequential(
                Linear(n_in, n_out),
                ReLU(),
                Linear(hid_dim, hid_dim)
            )

            conv = GINConv(block, 'sum')
            bn = BatchNorm1d(hid_dim)

            self.convs.append(conv)
            self.bns.append(bn)

        # Pooling
        # self.pool = SumPooling()
        # self.pool = AvgPooling()
        self.pool = SortPooling(1)

    def forward(self, graph, feat):
        xs = []
        x = feat
        for i in range(self.n_layer):
            x = F.relu(self.convs[i](graph, x))
            x = self.bns[i](x)
            xs.append(x)

        local_emb = th.cat(xs, 1)                    # patch-level embedding
        global_emb = self.pool(graph, local_emb)     # graph-level embedding

        return global_emb, local_emb


class InfoGraph(nn.Module):
    r"""
        InfoGraph model for unsupervised setting

    Parameters
    -----------
    in_dim: int
        Input feature size.
    hid_dim: int
        Hidden feature size.

    Functions
    -----------
    forward(graph):
        graph: dgl.Graph

    """

    def __init__(self, hid_dim, n_layer):
        super(InfoGraph, self).__init__()

        self.in_dim = 32 # Keep it fixed to a reasonable value for our experiments
        self.hid_dim = hid_dim

        self.n_layer = n_layer
        embedding_dim = hid_dim * n_layer

        self.encoder = GINEncoder(self.in_dim, hid_dim, n_layer)

        self.local_d = FeedforwardNetwork(embedding_dim, embedding_dim)   # local discriminator (node-level)
        self.global_d = FeedforwardNetwork(embedding_dim, embedding_dim)  # global discriminator (graph-level)

    # get_embedding function for evaluation the learned embeddings
    def get_embedding(self, graph):
        with th.no_grad():
            feat = graph.ndata['attr']
            global_emb, _ = self.encoder(graph, feat)

        return global_emb

    def forward(self, graph):
        feat = graph.ndata['attr']
        graph_id = graph.ndata['graph_id']

        global_emb, local_emb = self.encoder(graph, feat)

        global_h = self.global_d(global_emb)    # global hidden representation
        local_h = self.local_d(local_emb)       # local hidden representation

        measure = 'JSD'
        loss = local_global_loss_(local_h, global_h, graph_id, measure)

        return loss
