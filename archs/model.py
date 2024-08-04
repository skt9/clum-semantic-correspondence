"""
MIT License

Permission is hereby granted, free of charge, to any person obtaining 
a copy of this software and associated documentation files (the "Software"), 
to deal in the Software without restriction, including without limitation the 
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell 
copies of the Software, and to permit persons to whom the Software is furnished to 
do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies 
or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR 
PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE 
FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, 
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

    Some functions are taken from the Blackbox Deep Graph Matching code base.
"""

import torch

import utils.backbone
from archs.affinity_layer import InnerProductWithWeightsAffinity
from archs.sconv_archs import SiameseSConvOnNodes, SiameseNodeFeaturesToEdgeFeatures
from utils.config import cfg
from utils.feature_align import feature_align
from utils.utils import lexico_iter
from utils.visualization import easy_visualize
from QAP_Solver import QAPSolverModule, LAPSolverModule
from archs.attention import RelationAwareMultiHeadAttention, MultiheadAttention
import torch.nn as nn
from layers.resnet import ResNet50

def normalize_over_channels(x):
    channel_norms = torch.norm(x, dim=1, keepdim=True)
    return x / channel_norms

def concat_features(embeddings, num_vertices):
    res = torch.cat([embedding[:, :num_v] for embedding, num_v in zip(embeddings, num_vertices)], dim=-1)
    return res.transpose(0, 1)

class SupervisedNet(utils.backbone.VGG16_bn):

    def __init__(self):
        super(SupervisedNet, self).__init__()
        self.message_pass_node_features = SiameseSConvOnNodes(input_node_dim=1024)
        self.build_edge_features_from_node_features = SiameseNodeFeaturesToEdgeFeatures(
            total_num_nodes=self.message_pass_node_features.num_node_features
        )
        self.feat_dim = 1024
        self.vertex_affinity = InnerProductWithWeightsAffinity(
            self.feat_dim, self.message_pass_node_features.num_node_features)
        self.edge_affinity1 = InnerProductWithWeightsAffinity(
            self.feat_dim,
            self.build_edge_features_from_node_features.num_edge_features)
        self.edge_affinity2 = InnerProductWithWeightsAffinity(
            self.feat_dim,
            self.build_edge_features_from_node_features.num_edge_features)

        self.self_attention =  nn.MultiheadAttention(1024, num_heads=2, dropout=0.2)
        self.cross_attention =  nn.MultiheadAttention(1024, num_heads=2, dropout=0.2)

        self.solver_mode = "lap"
        self.solver_params = {'lambda':70,'costMargin':1.0}
        if self.solver_mode == "lap":
            self.solver = LAPSolverModule(self.solver_params)
        elif self.solver_mode == "qap":
            self.solver = QAPSolverModule(self.solver_params)

    def forward(
        self,
        images,
        points,
        graphs,
        n_points,
        perm_mats
    ):

        global_list = []
        orig_graph_list = []
        for image, p, n_p, graph in zip(images, points, n_points, graphs):
            
            # Extract feature from VGG
            nodes = self.node_layers(image)
            #   Add a learnable layer between VGG and 
            edges = self.edge_layers(nodes)

            global_list.append(self.final_layers(edges)[0].reshape((nodes.shape[0], -1)))
            nodes = normalize_over_channels(nodes)
            edges = normalize_over_channels(edges)

            # arrange features
            U = concat_features(feature_align(nodes, p, n_p, (256, 256)), n_p)
            F = concat_features(feature_align(edges, p, n_p, (256, 256)), n_p)
            node_features = torch.cat((U, F), dim=-1).unsqueeze(0)
            node_features = self.self_attention(query=node_features, key=node_features, value=node_features)[0]
            graph.x = node_features.squeeze()

            graph = self.message_pass_node_features(graph)
            orig_graph = self.build_edge_features_from_node_features(graph)
            orig_graph_list.append(orig_graph)

        #   Convert global_weights_list
        global_weights_list = [
            torch.cat([global_src, global_tgt], axis=-1) for global_src, global_tgt in lexico_iter(global_list)
        ]
        global_weights_list = [normalize_over_channels(g) for g in global_weights_list]

        #   Unary costs list
        unary_costs_list = [
            self.vertex_affinity([item.x for item in g_1], [item.x for item in g_2], global_weights)
            for (g_1, g_2), global_weights in zip(lexico_iter(orig_graph_list), global_weights_list)
        ]

        # Similarities to costs
        unary_costs_list = [[-x for x in unary_costs] for unary_costs in unary_costs_list]
        unary_costs_list = unary_costs_list[0]
        # g_1, g_2 = orig_graph_list[0][0], orig_graph_list[1][0]
        # g1_edges = g_1.edge_index.T
        # g1_edges = torch.unique(torch.sort(g1_edges,dim=1)[0],dim=0)

        # un_1, un_2 = g_1.x,  g_2.x
        # pairwise_costs_list = []
        # for edge in g1_edges:
        #     U1 = un_1[edge[0],:].repeat(un_1.shape[0],1).unsqueeze(0)    # extract edge feature and repeat it to make [num_pts,feat_dim]
        #     U2 = un_1[edge[1],:].repeat(un_2.shape[0],1).unsqueeze(0)  # extract edge feature and repeat it to make [num_pts, feat_num]
        #     kv = un_2.unsqueeze(0)
        #     c_attn1 = self.cross_attention(U1, kv, kv).squeeze()
        #     c_attn2 = self.cross_attention(U2, kv, kv).squeeze()
        #     # e_attn = self.self_attention([U1],[un_2],[global_weights_list[0]])
        #     # affinity_matrix1 = self.edge_affinity1([U1],[un_2], global_weights_list[0])[0]
        #     # affinity_matrix2 = self.edge_affinity1([U2],[un_2], global_weights_list[0])[0]
        #     pw_cost = -(1 - torch.mm(c_attn1, c_attn2.t()))
        #     # total_cost = affinity_matrix1
        #     pairwise_costs_list.append(pw_cost)

        if self.solver_mode == "lap":
            predicted_matching = []
            for unary in unary_costs_list:
                pred_match = self.solver(unary)
                predicted_matching.append(pred_match)

        # pairwise_costs = torch.stack(pairwise_costs_list,dim=0)
        # unary_costs = unary_costs_list[0]
        # gt_perm_mat = perm_mats[0].squeeze()
        # unary_costs = unary_costs.squeeze()
        # unary_costs = unary_costs - self.solver_params["costMargin"]*gt_perm_mat


        # predicted_matching = self.qap_model(unary_costs,pairwise_costs,g1_edges)

        return predicted_matching


class UnsupervisedNet(utils.backbone.VGG16_bn):

    def __init__(self):
        super(UnsupervisedNet, self).__init__()
        self.message_pass_node_features = SiameseSConvOnNodes(input_node_dim=1024)
        self.build_edge_features_from_node_features = SiameseNodeFeaturesToEdgeFeatures(
            total_num_nodes=self.message_pass_node_features.num_node_features
        )
        self.feat_dim = 1024
        self.vertex_affinity = InnerProductWithWeightsAffinity(
            self.feat_dim, self.message_pass_node_features.num_node_features)
        self.edge_affinity1 = InnerProductWithWeightsAffinity(
            self.feat_dim,
            self.build_edge_features_from_node_features.num_edge_features)
        self.edge_affinity2 = InnerProductWithWeightsAffinity(
            self.feat_dim,
            self.build_edge_features_from_node_features.num_edge_features)

        self.self_attention =  nn.MultiheadAttention(1024, num_heads=2, dropout=0.2)
        self.cross_attention =  nn.MultiheadAttention(1024, num_heads=2, dropout=0.2)

        self.solver_mode = "qap"
        self.solver_params = {'lambda':70,'costMargin':1.0}
        if self.solver_mode == "lap":
            self.solver = LAPSolverModule(self.solver_params)
        elif self.solver_mode == "qap":
            self.solver = QAPSolverModule(self.solver_params)

    def forward(
        self,
        images,
        points,
        graphs,
        n_points,
        perm_mats
    ):

        global_list = []
        orig_graph_list = []
        for image, p, n_p, graph in zip(images, points, n_points, graphs):
            
            nodes = self.node_layers(image) #   Extract feature from VGG
            edges = self.edge_layers(nodes) #   Add a learnable layer between VGG and 
            global_list.append(self.final_layers(edges)[0].reshape((nodes.shape[0], -1)))
            nodes = normalize_over_channels(nodes)
            edges = normalize_over_channels(edges)

            # arrange features
            U = concat_features(feature_align(nodes, p, n_p, (256, 256)), n_p)
            F = concat_features(feature_align(edges, p, n_p, (256, 256)), n_p)
            node_features = torch.cat((U, F), dim=-1).unsqueeze(0)
            node_features = self.self_attention(query=node_features, key=node_features, value=node_features)[0]
            graph.x = node_features.squeeze()

            graph = self.message_pass_node_features(graph)
            orig_graph = self.build_edge_features_from_node_features(graph)
            orig_graph_list.append(orig_graph)

        #   Convert global_weights_list
        global_weights_list = [
            torch.cat([global_im1, global_im2], axis=-1) for global_im1, global_im2 in lexico_iter(global_list)
        ]
        global_weights_list = [normalize_over_channels(g) for g in global_weights_list]

        unary_costs_list = []
        for (g_1, g_2), global_weights in zip(lexico_iter(orig_graph_list), global_weights_list):
            unary = self.vertex_affinity([item.x for item in g_1], [item.x for item in g_2], global_weights)[0]
            unary_costs_list.append(unary)

        # Similarities to costs
        unary_costs_list = [-x for x in unary_costs_list]

        if self.solver_mode == "lap":
            predicted_matching = []
            for unary in unary_costs_list:
                pred_match = self.solver(unary)
                predicted_matching.append(pred_match)

        if self.solver_mode == "qap":
            """
                Pairwise costs list. lexico_iter.
            """
            pairwise_costs_list = []
            edges_list = []

            for (g_1,g_2) in lexico_iter(orig_graph_list):
                g1_edges = g_1[0].edge_index.T
                g1_edges = torch.unique(torch.sort(g1_edges,dim=1)[0],dim=0)
                edges_list.append(g1_edges) #   Add edges
                un_1, un_2 = g_1[0].x,  g_2[0].x
                instance_pairwise_costs = []
                for edge in g1_edges:
                    U1 = un_1[edge[0],:].repeat(un_1.shape[0],1).unsqueeze(0)    # extract edge feature and repeat it to make [num_pts,feat_dim]
                    U2 = un_1[edge[1],:].repeat(un_2.shape[0],1).unsqueeze(0)  # extract edge feature and repeat it to make [num_pts, feat_num]
                    kv = un_2.unsqueeze(0)
                    c_attn1 = self.cross_attention(U1, kv, kv)[0].squeeze()
                    c_attn2 = self.cross_attention(U2, kv, kv)[0].squeeze()
                    # e_attn = self.self_attention([U1],[un_2],[global_weights_list[0]])
                    # affinity_matrix1 = self.edge_affinity1([U1],[un_2], global_weights_list[0])[0]
                    # affinity_matrix2 = self.edge_affinity1([U2],[un_2], global_weights_list[0])[0]
                    pw_cost = -(1 - torch.mm(c_attn1, c_attn2.t()))
                    # total_cost = affinity_matrix1
                    instance_pairwise_costs.append(pw_cost)
                pairwise_costs_list.append(instance_pairwise_costs)
            predicted_matching = []
            for unary,pw_cost, edges in zip(unary_costs_list, pairwise_costs_list, edges_list):
                pw_cost = torch.stack(pw_cost,dim=0)
                pred_match = self.solver(unary, pw_cost, edges)
                predicted_matching.append(pred_match)


        return predicted_matching
