# --------------------------------------------------------
# graph_net.py
# by SangHyuk Yoo, shyoo@yonsei.ac.kr
# last modified : Sat Aug 20 12:13:23 KST 2022
#
# Objectives
# build graph neural network
#
# Prerequisites library
# 1. ASE(Atomistic Simulation Environment)
# 2. DGL
# 3. PyTorch
# 4. scikit-learn
# --------------------------------------------------------

from typing import List

import dgl
import dgl.function as fn
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler


class ExplicitMLP(nn.Module):
    def __init__(
        self,
        in_feats: int,
        out_feats: int,
        hidden_feats: int = 128,
        hidden_lyrs: int = 3,
        activation: str = "gelu",
        actv_first: bool = False,
        init_param: bool = True,
    ):
        """Multilayer Perceptron in a Graph Neural Network.

        Parameters
        ----------
        in_feats : int
            the number of input feature
        out_feats : int
            the number of output feature
        hidden_feats : int
            the number of hidden feature
        hidden_lyrs : int
            the number of hidden layer
        activation : str
            activation function for multilayer perceptrons
        actv_first : bool
            prepend activation function or not
        init_param : bool

        """

        # initialize parent instance
        super(ExplicitMLP, self).__init__()

        # create activation function
        if activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "gelu":
            self.activation = nn.GELU()

        # append linear layers
        self.hidden_lyrs = hidden_lyrs
        mlp_layers = nn.ModuleList()
        for i in range(hidden_lyrs):
            if i == 0:
                if actv_first:
                    mlp_layers.append(self.activation)
                mlp_layers.append(nn.Linear(in_feats, hidden_feats))
                mlp_layers.append(self.activation)
            elif i == hidden_lyrs - 1:
                mlp_layers.append(nn.Linear(hidden_feats, out_feats))
            else:
                mlp_layers.append(nn.Linear(hidden_feats, hidden_feats))
                mlp_layers.append(self.activation)
        self.mlp_layers = mlp_layers
        self.hidden_lyrs = len(self.mlp_layers)

        # initialize parameters
        if init_param:
            for layer in self.mlp_layers:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)

    def forward(self, in_feats):
        x = in_feats
        for i in range(self.hidden_lyrs):
            x = self.mlp_layers[i](x)
        return x


class GNNBlock(nn.Module):
    def __init__(
        self,
        in_node_feats: int,
        in_edge_feats: int,
        out_node_feats: int,
        hidden_feats: int = 128,
        activation: str = "gelu",
        drop_edge: bool = False,
    ):
        """a Graph Neural Network block.

        Parameters
        ----------
        in_node_feats : int
            the number of input node feature
        in_edge_feats : int
            the number of input edge feature
        out_node_feats : int
            the number of output node feature
        hidden_feats : int
            the number of hidden feature of multilayer perceptrons
        activation : str
            activation function for multilayer perceptrons
        drop_edge : bool
            drop edge feature or not

        """
        # initialize instance
        super(GNNBlock, self).__init__()
        self.drop_edge = drop_edge

        # create MLPs
        self.affine_edge = ExplicitMLP(
            in_edge_feats, hidden_feats, hidden_feats, 2, activation, False
        )
        self.affine_node_src = nn.Linear(in_node_feats, hidden_feats)
        self.affine_node_dst = nn.Linear(in_node_feats, hidden_feats)

        self.phi = ExplicitMLP(
            hidden_feats, in_edge_feats, hidden_feats, 2, activation, True
        )
        self.lyr_norm_edge = nn.LayerNorm(in_edge_feats)

        self.theta_dst = nn.Linear(in_node_feats, hidden_feats)
        self.theta_msg = nn.Linear(in_node_feats, hidden_feats)
        self.theta = ExplicitMLP(
            hidden_feats, out_node_feats, hidden_feats, 1, activation, True
        )

    def forward(self, g: dgl.DGLGraph, node_feats: torch.Tensor) -> torch.Tensor:
        # copy node features
        h = node_feats.clone()

        # forward
        with g.local_scope():
            if self.drop_edge and self.training:
                raise NotImplementedError("drop-out procedure is not implemented.")
            if g.is_block:
                raise NotImplementedError("batched graph is not supported.")
            else:
                h_src = h_dst = h

            # prepare a message
            g.srcdata["h_n"] = h_src
            g.dstdata["h_n"] = h_dst
            edge_info = g.edges()
            src_idx = edge_info[0].long()
            dst_idx = edge_info[1].long()
            edge_embed = self.affine_edge(g.edata["h_e"])
            src_embed = self.affine_node_src(h_src[src_idx])
            dst_embed = self.affine_node_dst(h_dst[dst_idx])
            g.edata["h_e_emb"] = self.phi(edge_embed + src_embed + dst_embed)

            # message and reduce
            g.update_all(fn.src_mul_edge("h_n", "h_e_emb", "m"), fn.sum("m", "h"))
            m = g.ndata["h"]

        # edge normalization
        norm_e = self.lyr_norm_edge(g.edata["h_e"])
        g.edata["h_e"] = norm_e

        # update node
        node_feats = self.theta(self.theta_dst(h) + self.theta_msg(m))
        return node_feats


class GNNModel(nn.Module):
    def __init__(
        self,
        in_node_feats,
        in_edge_feats,
        out_node_feats,
        hidden_feats: int = 128,
        num_blocks: int = 3,
        activation: str = "gelu",
        drop_edge: bool = False,
        use_layer_norm: bool = False,
        use_batch_norm: bool = False,
    ):
        """Graph Neural Network.

        Parameters
        ----------
        in_node_feats : int
            the number of input node feature
        in_edge_feats : int
            the number of input edge feature
        out_node_feats : int
            the number of output node feature
        hidden_feats : int
            the number of hidden feature of multilayer perceptrons
        num_blocks : int
            the number of GNN block
        activation : str
            activation function for multilayer perceptrons
        drop_edge : bool
            drop edge feature or not
        use_layer_norm : bool

        use_batch_norm : bool

        """
        # initialize instance
        super(GNNModel, self).__init__()
        self.drop_edge = drop_edge
        self.use_layer_norm = use_layer_norm
        self.use_batch_norm = use_batch_norm

        if use_batch_norm == use_layer_norm and use_batch_norm:
            raise ValueError("Only one type of normalization can be used at a time.")

        # create GN blocks
        self.gn_blocks = nn.ModuleList()
        self.norm = nn.ModuleList()
        for i in range(num_blocks):
            if i == 0:
                self.gn_blocks.append(
                    GNNBlock(
                        in_node_feats,
                        in_edge_feats,
                        out_node_feats,
                        hidden_feats,
                        activation,
                        drop_edge,
                    )
                )
            else:
                self.gn_blocks.append(
                    GNNBlock(
                        out_node_feats,
                        in_edge_feats,
                        out_node_feats,
                        hidden_feats,
                        activation,
                        drop_edge,
                    )
                )

            if use_layer_norm:
                self.norm.append(nn.LayerNorm(out_node_feats))
            elif use_batch_norm:
                self.norm.append(nn.BatchNorm1d(out_node_feats))

    def forward(self, g: dgl.DGLGraph, node_feats: torch.Tensor) -> torch.Tensor:
        for i, block in enumerate(self.gn_blocks):
            if self.use_layer_norm or self.use_batch_norm:
                node_feats = block.forward(g, self.norm[i](node_feats)) + node_feats
            else:
                node_feats = block.forward(g, node_feats) + node_feats
        return node_feats


class RBFExpansion(nn.Module):
    def __init__(self, low: float = 0.0, high: float = 30.0, gap: float = 0.1):
        # initialize instance
        super(RBFExpansion, self).__init__()

        # get centers
        num_centers = np.int32(np.ceil((high - low) / gap))
        self.centers = torch.linspace(low, high, num_centers)
        self.centers = nn.Parameter(self.centers.float(), requires_grad=False)
        self.gamma = 1 / gap

    def forward(self, in_feats):
        radial = in_feats - self.centers
        return torch.exp(-self.gamma * (radial**2))


class MDNet(nn.Module):
    def __init__(
        self,
        in_node_feats,
        embed_feats,
        out_node_feats,
        hidden_feats,
        num_blocks,
        dropout: float = 0.1,
        drop_edge: bool = False,
        use_layer_norm: bool = True,
    ):
        # initialize instance
        super(MDNet, self).__init__()
        self.embed_feats = embed_feats

        # edge feature scaler
        self.edge_expand = RBFExpansion(high=1, gap=0.025)
        self.length_avg = nn.Parameter(torch.tensor([0.0]), requires_grad=False)
        self.length_std = nn.Parameter(torch.tensor([1.0]), requires_grad=False)
        self.length_scaler = StandardScaler()

        # drop or normalization
        if drop_edge:
            self.edge_drop_out = nn.Dropout(dropout)
        if use_layer_norm:
            self.edge_layer_norm = nn.LayerNorm(embed_feats)

        # node feature : one-hot vector
        # single element system uses random number vector
        self.node_feats = nn.Parameter(
            torch.randn(1, self.embed_feats), requires_grad=True
        )

        # create edge encoder
        in_edge_feats = 3 + 1 + len(self.edge_expand.centers)
        self.edge_encoder = ExplicitMLP(
            in_edge_feats, self.embed_feats, hidden_feats, 3, "gelu", False
        )

        # create node decoder
        self.node_decoder = ExplicitMLP(embed_feats, 3, hidden_feats, 2, "gelu", False)

        # create GNN
        self.net = GNNModel(
            in_node_feats=in_node_feats,
            in_edge_feats=embed_feats,
            out_node_feats=out_node_feats,
            hidden_feats=hidden_feats,
            num_blocks=num_blocks,
            activation="gelu",
            drop_edge=drop_edge,
            use_layer_norm=use_layer_norm,
            use_batch_norm=not use_layer_norm,
        )

    def forward(self, graph: dgl.DGLGraph) -> torch.Tensor:
        # create features for training
        # edge feature : concatanate unit vector, normalized distance and gaussian filtered distance
        if self.training:
            distance = graph.edata["distance"].detach().cpu()
            distance = np.asarray(distance).reshape(-1, 1)
            self.length_scaler.partial_fit(distance)
            self.length_avg[0] = self.length_scaler.mean_[0]
            self.length_std[0] = self.length_scaler.scale_[0]

        distance = graph.edata["distance"]
        distance = (distance - self.length_avg) / self.length_std
        edge_feats = torch.concat(
            (graph.edata["unit_vec"], distance, self.edge_expand(distance)), -1
        )
        graph.edata["h_e"] = edge_feats
        graph.apply_edges(self.encode_edge)

        node_feats = self.node_feats.repeat((graph.num_nodes(), 1))
        node_feats = self.net(graph, node_feats)
        pred = self.node_decoder(node_feats)
        return pred

    def encode_edge(self, edges):
        e_feat = self.edge_encoder.forward(edges.data["h_e"])
        e_feat = self.edge_layer_norm(e_feat)
        return {"h_e": e_feat}

    def decode_node(self, nodes):
        n_feat = self.node_decoder.forward(nodes.data["h_n"])
        return {"h_n": n_feat}
