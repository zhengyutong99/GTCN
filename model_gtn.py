import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import torch_sparse
from gcn import GCNConv
from torch_geometric.utils import softmax
from utils import _norm, generate_non_local_graph

class Preprocess(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Preprocess, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        return x

class ScaledDotProductAttention(nn.Module):
    def __init__(self, args, embed_dim, num_heads):
        super(ScaledDotProductAttention, self).__init__()
        self.num_heads = num_heads
        self.scaling = float(embed_dim) ** -0.5
        self.embed_dim = embed_dim

        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)

        self.out_linear = nn.Linear(embed_dim, args.num_classes)

    def forward(self, x):
        Q = self.q_linear(x)
        K = self.k_linear(x)
        V = self.v_linear(x)
        
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) * self.scaling
        attn_weights = F.softmax(attn_weights, dim=-1)/(self.embed_dim^(0.5))

        attn_output = torch.matmul(attn_weights, V)
        
        output = self.out_linear(attn_output)
        
        return output

class GTNs(nn.Module):
    def __init__(self, args, 
                 num_edge_type, # 10+1 (D-D ~ I-D + Identity matrix)
                 w_in, # 64 (embedding dimension)
                 num_class, # 2 (分類數)
                 num_nodes, # 31778 (entity 數量)
                 preprocess_input_dim, # 512 (embedding input dimension)
                 preprocess_output_dim, # 64 (embedding output dimension)
                 feats_dim_list): # [2614, 20501, 8208, 455]
        super(GTNs, self).__init__()
        self.device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
        self.args = args
        self.num_nodes = num_nodes
        self.num_GTN_layers = args.num_GTN_layers
        GTNs = []
        for i in range(args.num_GTN_layers):
            if i == 0:
                GTNs.append(GTN(num_edge_type, w_in, num_class, num_nodes, args))
            else:
                GTNs.append(GTN(num_edge_type, args.node_dim, num_class, num_nodes, args))
        self.GTNs = nn.ModuleList(GTNs)
        self.loss = nn.CrossEntropyLoss()

        if args.embedding_method == 0:
            self.preprocess_nn = Preprocess(preprocess_input_dim, preprocess_output_dim)
        elif args.embedding_method == 1:
            self.fc_list = nn.ModuleList([nn.Linear(feats_dim, args.preprocess_output_dim, bias=True) for feats_dim in feats_dim_list])
            self.preprocess_output_dim = args.preprocess_output_dim
            self.feat_drop = nn.Dropout(0.5)
        
        if args.classifier == 'NN':
            self.linear = nn.Linear(args.node_dim*2, num_class)
        elif args.classifier == 'attention':
            self.attention = ScaledDotProductAttention(args, embed_dim=args.node_dim * 2, num_heads=4)

        if args.dataset == "PPI":
            self.m = nn.Sigmoid()
            self.loss = nn.BCELoss()
        else:
            self.loss = nn.CrossEntropyLoss()
        
        self.to(self.device)
        
    def forward(self, A, features_list, drug_indices, cancer_indices, target=None, num_nodes=None, eval=False, args=None, n_id=None, node_labels=None, type_mask = None, epoch=None):
        if self.args.embedding_method == 0:
            transformed_features = self.preprocess_nn(features_list).to(self.device)
        elif self.args.embedding_method == 1:
            transformed_features = torch.zeros(type_mask.shape[0], self.preprocess_output_dim, device=features_list[0].device)
            for i, fc in enumerate(self.fc_list):
                node_indices = np.where(type_mask == i)[0]
                transformed_features[node_indices] = fc(features_list[i])
            transformed_features = self.feat_drop(transformed_features)
        
        if num_nodes is None:
            num_nodes = self.num_nodes
        
        H_, Ws, result_A, New_adjs = self.GTNs[0](A, transformed_features, num_nodes=num_nodes, epoch=epoch)
        
        for i in range(1, self.num_GTN_layers):
            H_, Ws, result_A, New_adjs = self.GTNs[i](A, H_, num_nodes=num_nodes, epoch=epoch)

        drug_features = H_[drug_indices]
        cancer_features = H_[cancer_indices]
        pair_features = torch.cat([drug_features, cancer_features], dim=1)

        if self.args.classifier == 'NN':
            y_pair = self.linear(pair_features)
        elif self.args.classifier == 'attention':
            batch_size = 32768

            outputs = []
            for i in range(0, pair_features.shape[0], batch_size):
                batch_pair_features = pair_features[i:i+batch_size]
                y_batch = self.attention(batch_pair_features)
                
                outputs.append(y_batch)
        
        if eval or target is None:
            return y_pair, Ws, New_adjs, transformed_features
        else:
            if self.args.dataset == 'PPI':
                loss = self.loss(self.m(y_pair), target)
            else:
                loss = self.loss(y_pair, target.squeeze())
                
        return loss, y_pair, Ws, New_adjs, transformed_features

class GTN(nn.Module):
    def __init__(self, num_edge_type, w_in, num_class, num_nodes, args=None, pre_trained=None):
        super(GTN, self).__init__()
        if args.non_local:
            num_edge_type += 1
        self.num_edge_type = num_edge_type
        self.num_channels = args.num_channels
        self.num_nodes = num_nodes
        self.w_in = w_in
        args.w_in = w_in
        self.w_out = args.node_dim
        self.num_class = num_class
        self.num_layers = args.num_layers
        self.num_GCN_layers = args.num_GCN_layers
        
        if pre_trained is None:
            layers = []
            for i in range(self.num_layers):
                if i == 0:
                    layers.append(GTLayer(num_edge_type, self.num_channels, num_nodes, first=True, args=args))
                else:
                    layers.append(GTLayer(num_edge_type, self.num_channels, num_nodes, first=False, args=args))
            self.layers = nn.ModuleList(layers)
        else:
            layers = []
            for i in range(self.num_layers):
                if i == 0:
                    layers.append(GTLayer(num_edge_type, self.num_channels, num_nodes, first=True, args=args, pre_trained=pre_trained[i]))
                else:
                    layers.append(GTLayer(num_edge_type, self.num_channels, num_nodes, first=False, args=args, pre_trained=pre_trained[i]))
            self.layers = nn.ModuleList(layers)
        
        self.Ws = []
        for i in range(self.num_channels):
            self.Ws.append(GCNConv(in_channels=self.w_in, out_channels=self.w_out, args=args).weight)
        self.Ws = nn.ParameterList(self.Ws)

        self.linear1 = nn.Linear(self.w_out*self.num_channels, self.w_out)

        self.gcns = nn.ModuleList([
            GCNConv(in_channels=w_in, out_channels=w_in, args=args) for _ in range(self.num_GCN_layers)
        ])
        
        feat_trans_layers = []
        for i in range(self.num_layers+1):
            feat_trans_layers.append(nn.Sequential(nn.Linear(self.w_out, 128),
                                        nn.ReLU(),
                                        nn.Linear(128, 64)))
        self.feat_trans_layers = nn.ModuleList(feat_trans_layers)

        self.args = args
        self.out_norm = nn.LayerNorm(self.w_out)
        self.relu = torch.nn.ReLU()

    def forward(self, A, X, num_nodes, eval=False, node_labels=None, epoch=None):        
        Ws = []
        X_ = [X@W for W in self.Ws]
        H = [X@W for W in self.Ws]
        
        for i in range(self.num_layers):
            if self.args.non_local:
                g = generate_non_local_graph(self.args, self.feat_trans_layers[i], torch.stack(H).mean(dim=0), A, self.num_edge_type, num_nodes)
                deg_inv_sqrt, deg_row, deg_col = _norm(g[0].detach(), num_nodes, g[1])
                g[1] = softmax(g[1],deg_row)
                if len(A) < self.num_edge_type:
                    A.append(g)
                else:
                    A[-1] = g
            
            if i == 0 :
                H, W, result_A, New_adjs = self.layers[i](H, A, num_nodes, epoch=epoch, layer=i+1)
            elif i > 0:
                H, W, result_A, New_adjs = self.layers[i](H, A, num_nodes, epoch=epoch, layer=i+1, lastLayerA=result_A)
            Ws.append(W)
        
        for i in range(self.num_channels):
            edge_index, edge_weight = result_A[i][0], result_A[i][1]  # 從 result_A 抓 adjacency matrix
            for gcn_layer in self.gcns:
                H[i] = gcn_layer(H[i], edge_index=edge_index, edge_weight=edge_weight)
                
        for i in range(self.num_channels):
            if i == 0:
                H_ = F.relu(self.args.beta * (X_[i]) + (1-self.args.beta) * H[i])
            else:
                if self.args.channel_agg == 'concat':
                    H_ = torch.cat((H_, F.relu(self.args.beta * (X_[i]) + (1-self.args.beta) * H[i])), dim=1)
                elif self.args.channel_agg == 'mean':
                    H_ = H_ + F.relu(self.args.beta * (X_[i]) + (1-self.args.beta) * H[i])
        
        if self.args.channel_agg == 'concat':
            H_ = F.relu(self.linear1(H_))
        elif self.args.channel_agg == 'mean':
            H_ = H_ /self.args.num_channels
        
        return H_, Ws, result_A, New_adjs

class GTLayer(nn.Module):
    
    def __init__(self, in_channels, # 10+1 (D-D ~ I-D + Identity matrix)
                 out_channels, # 3 (head 數量)
                 num_nodes, # 31778 (entity 數量)
                 first=True, args=None, pre_trained=None):
        super(GTLayer, self).__init__()
        self.args = args
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.first = first
        self.num_nodes = num_nodes
        if pre_trained is not None:
            self.conv1 = GTConv(in_channels, out_channels, num_nodes, args=args, pre_trained=pre_trained.conv1)
        else:
            self.conv1 = GTConv(in_channels, out_channels, num_nodes, args=args)
        self.args = args
        self.feat_transfrom = nn.Sequential(nn.Linear(args.w_in, 128),
                                        nn.ReLU(),
                                        nn.Linear(128, 64))
    def forward(self, H_, A, num_nodes, epoch=None, layer=None, lastLayerA=None):
        result_A, W1 = self.conv1(A, num_nodes, epoch=epoch, layer=layer)
        W = [W1]
        Hs = []
        New_adjs = []
        
        block_save_dir = os.path.join(self.args.save_dir, 'matrix_blocks')
        os.makedirs(block_save_dir, exist_ok=True)
            
        for i in range(len(result_A)):
            a_edge, a_value = result_A[i]
            mat_a = torch.sparse_coo_tensor(a_edge, a_value, (num_nodes, num_nodes)).to(a_edge.device)
            mat_a = mat_a.coalesce()

            H = torch.sparse.mm(mat_a, H_[i])
            Hs.append(H)
        
        return Hs, W, result_A, New_adjs

class GTConv(nn.Module):
    def __init__(self, in_channels, 
                 out_channels, 
                 num_nodes, 
                 args=None, pre_trained=None):
        super(GTConv, self).__init__()
        self.args = args
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.Tensor(out_channels,in_channels))
        
        self.bias = None
        self.scale = nn.Parameter(torch.Tensor([0.1]), requires_grad=False)
        self.num_nodes = num_nodes

        self.reset_parameters()

        if pre_trained is not None:
            with torch.no_grad():
                self.weight.data = pre_trained.weight.data
        
    def reset_parameters(self):
        n = self.in_channels
        num_edges = self.weight.shape[1]
        nn.init.constant_(self.weight, 1 / num_edges)
        if self.args.non_local and self.args.non_local_weight != 0:
            with torch.no_grad():
                self.weight[:,-1] = self.args.non_local_weight
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, A, num_nodes, epoch=None, layer=None):
        weight = self.weight
        filter = F.softmax(weight, dim=1)
        num_channels = filter.shape[0]
        results = []
        for i in range(num_channels):
            for j, (edge_index,edge_value) in enumerate(A):
                if j == 0:
                    total_edge_index = edge_index
                    total_edge_value = edge_value*filter[i][j]
                else:
                    total_edge_index = torch.cat((total_edge_index, edge_index), dim=1)
                    total_edge_value = torch.cat((total_edge_value, edge_value*filter[i][j]))
            
            index, value = torch_sparse.coalesce(total_edge_index.detach(), total_edge_value, m=num_nodes, n=num_nodes, op='add')
            results.append((index, value))
        
        return results, filter