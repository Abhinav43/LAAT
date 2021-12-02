# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# import torch.nn.functional as F
# import torch_geometric.transforms as T
# from scipy import sparse
# from torch_geometric.nn import GCNConv
# import pickle as pk
# from torch.nn.init import xavier_uniform


# from torch_sparse import SparseTensor
# import numpy as np


# def gen_Adj(adj_file, t = 0.4, both = False):

#     """occurance with t"""
    
#     result = adj_file
#     _adj   = result['adj']
#     _nums  = result['nums']
#     _nums  = _nums[:, np.newaxis]
#     _adj   = _adj / _nums

#     _adj[_adj < t]  = 0
#     _adj[_adj >= t] = 1
    
#     if both:
#         _adj = _adj * 0.2 / (_adj.sum(0, keepdims=True) + 1e-6)
#     return _adj


# def load_emd(total_labels, adj_type, 
#              emd_type,     slag = './src/emd_data/'):
    
    
#     """ total labels : [40, 50, 1166, 8865]
#         adj_type     : either occ_t or occ_both
#         emd_type     : dim_300_10_half_raw_50_w2v_sum_w2v_no_tfidf
#         slag         : path"""
    
#     """ All emd path : Desktop/mtram/data/laat/icd9_data/
#     final_data_laat/models_f/gcn_final_data/all_gcn_emd_data"""
    
#     with open(f'{slag}all_gcn_data','rb') as f:
#         adj_data, emd_data = pk.load(f)
    
#     adj_m = adj_data[total_labels]
#     emd_m = emd_data[total_labels]
    
#     if adj_type == 'occ_t':
#         adj_final = gen_Adj(adj_m, both = False)
#     else:
#         adj_final = gen_Adj(adj_m, both = True)
#     emd_final     = emd_m[emd_type]
    
#     return adj_final, emd_final


# def get_gcn_data_train(total_labels, adj_type, 
#                        emd_type, device, slag = './src/emd_data/'):

    
#     adj_da, x_da     = load_emd(total_labels, adj_type, emd_type, slag)
#     x_da_f           = torch.nn.Parameter(torch.Tensor(x_da).to(device), requires_grad=True)
#     A                = torch.Tensor(adj_da).to(device)
#     edge_index       = A.nonzero(as_tuple=False).t()
#     edge_weight      = torch.nn.Parameter(A[edge_index[0], edge_index[1]],requires_grad=True)
#     adj              = SparseTensor(row=edge_index[0], col=edge_index[1], value=edge_weight,
#                           sparse_sizes=( total_labels, total_labels))
#     return x_da_f,   adj




# def gc_data(n_labels, adj_type, 
#             device, emd_type):
    
    
#     if 'half' in emd_type:
#         emd_dict   = {0: emd_type, 
#                       1: emd_type.replace('half', 'full')}
#     else:
#         emd_dict   = {0: emd_type.replace('full', 'half'), 
#                       1: emd_type}
    
#     gcn_d          = {label_lvl: get_gcn_data_train(n_labels[label_lvl],
#                                                  adj_type, emd_dict[label_lvl], 
#                                                  device) for label_lvl in range(len(n_labels))}
#     return gcn_d




# def gcn_l(n_labels, emd_dim, output_dim, 
#           inner_dims, drop, att, device):
    
    
#     if isinstance(output_dim, int):
#         gcn_layer    = nn.ModuleList([GCN(total_labels = n_labels[label_lvl],
#                                           emd_dim      = emd_dim,
#                                           out_dim      = output_dim,
#                                           inner_dims   = inner_dims,
#                                           dropout      = drop,
#                                           att= att).to(device) for label_lvl in range(len(n_labels))])
#     else:
#         gcn_layer    = nn.ModuleList([GCN(total_labels = n_labels[label_lvl],
#                                           emd_dim      = emd_dim,
#                                           out_dim      = output_dim[label_lvl],
#                                           inner_dims   = inner_dims,
#                                           dropout      = drop,
#                                           att= att).to(device) for label_lvl in range(len(n_labels))])
        
#     return gcn_layer



# class att_layer(nn.Module):
#     def __init__(self, dim_1, dim_2):
#         super(att_layer, self).__init__()
        
#         # if x is 50 x 300 then u will be 300 x 50
#         # dim_1 300, dim_2 50

#         self.U = nn.Linear(dim_1, dim_2)
#         xavier_uniform(self.U.weight)

#     def forward(self, x):
        
#         # x : 50  x 300 => 300 x 50
#         # u : 300 x 50  => 50  x 300 
        
#         x     = torch.tanh(x)
        
#         if x.dim() == 3:            
#             alpha = F.softmax(self.U.weight.matmul(x.transpose(1, 2)), dim=2)
#         else:
#             alpha = F.softmax(self.U.weight.matmul(x.t()), dim=1)
#         # 50 x 300 <> 300 x 50 ==> 50 x 50

#         m = alpha.matmul(x)
#         return m


# class GCN(torch.nn.Module):
#     def __init__(self,
#                  total_labels     = 50,
#                  emd_dim          = 300,
#                  out_dim          = 1024,
#                  inner_dims       = 1024,
#                  dropout          = False, 
#                  att              = False, 
#                  dropout_value    = 0.2):
        
        
#         super(GCN, self).__init__()
        
        
#         self.conv1       = GCNConv(emd_dim, inner_dims)
#         self.conv2       = GCNConv(inner_dims, out_dim)
#         self.dropout     = dropout
#         self.att         = att
#         self.drp         = dropout_value
        
#         if self.att:
#             self.att_laye = att_layer(out_dim, total_labels)
    
#     def forward(self, xm):
#         x, edge_index = xm
#         x             = self.conv1(x, edge_index)
#         x             = F.relu(x)
        
#         if self.dropout:
#             x = F.dropout(x, p=self.drp, training=self.training)
        
#         x             = self.conv2(x, edge_index)
#         if self.att:
#             x = self.att_laye(x)
#         return x
    
    
# def layer_ops(x, gcn_layer, gcn_data, lavel, linear_layer, con_dim, trans = True):
    
#     x_gcn              = gcn_layer[lavel](gcn_data[lavel])
#     weights            = torch.cat((linear_layer[lavel].weight, x_gcn), dim = con_dim)
#     if trans:
#         weights            = weights.t()
#         weights            = weights.expand(x.size()[0],  weights.size()[0], weights.size()[1])

#     return weights


# def res(data, desired_tensor, device):
#     target   = torch.zeros(data.size()[0], desired_tensor.size()[-1]).to(device)
#     target[:, :data.size()[-1]] = data
#     return target


# def layer_ops_sum(x, gcn_layer, gcn_data, lavel, linear_layer, device, con_dim, trans = True):
    
    
#     if trans:
#         x_gcn              = gcn_layer[lavel](gcn_data[lavel]).t()
#         li_data            = linear_layer[lavel].weight.t()
#     else:
#         x_gcn              = gcn_layer[lavel](gcn_data[lavel])
#         li_data            = linear_layer[lavel].weight
         
#     if x_gcn.size() != li_data.size():
#         if x_gcn.size()[-1] < li_data.size()[-1]:
#             x_gcn   = res(x_gcn, li_data, device)
            
#         else:
#             li_data = res(li_data, x_gcn, device)
    
#     weights            = x_gcn + li_data    
#     if trans:
#         weights            = weights.t()
#         weights            = weights.expand(x.size()[0],  weights.size()[0], weights.size()[1])

#     return weights
    
# # mm = get_gcn_data_train(50, 'occ_t', 'dim_300_10_full_raw_50_w2v_sum_w2v_no_tfidf', 'cpu', './')
# # yu = GCN(dropout = False, att = False)

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.nn.functional as F
import torch_geometric.transforms as T
from scipy import sparse
from torch_geometric.nn import GCNConv
import pickle as pk
from torch.nn.init import xavier_uniform


from torch_sparse import SparseTensor
import numpy as np


def gen_Adj(adj_file, t = 0.4, both = False):

    """occurance with t"""
    
    result = adj_file
    _adj   = result['adj']
    _nums  = result['nums']
    _nums  = _nums[:, np.newaxis]
    _adj   = _adj / _nums

    _adj[_adj < t]  = 0
    _adj[_adj >= t] = 1
    
    if both:
        _adj = _adj * 0.2 / (_adj.sum(0, keepdims=True) + 1e-6)
    return _adj


def load_emd(total_labels, adj_type, 
             emd_type,     slag = './src/emd_data/'):
    
    
    """ total labels : [40, 50, 1166, 8865]
        adj_type     : either occ_t or occ_both
        emd_type     : dim_300_10_half_raw_50_w2v_sum_w2v_no_tfidf
        slag         : path"""
    
    """ All emd path : Desktop/mtram/data/laat/icd9_data/
    final_data_laat/models_f/gcn_final_data/all_gcn_emd_data"""
    
    with open(f'{slag}all_gcn_data','rb') as f:
        adj_data, emd_data = pk.load(f)
    
    adj_m = adj_data[total_labels]
    emd_m = emd_data[total_labels]
    
    if adj_type == 'occ_t':
        adj_final = gen_Adj(adj_m, both = False)
    else:
        adj_final = gen_Adj(adj_m, both = True)
    emd_final     = emd_m[emd_type]
    
    return adj_final, emd_final


def get_gcn_data_train(total_labels, adj_type, 
                       emd_type, device, slag = './src/emd_data/'):

    
    adj_da, x_da     = load_emd(total_labels, adj_type, emd_type, slag)
    x_da_f           = torch.nn.Parameter(torch.Tensor(x_da).to(device), requires_grad=True)
    A                = torch.Tensor(adj_da).to(device)
    edge_index       = A.nonzero(as_tuple=False).t()
    edge_weight      = torch.nn.Parameter(A[edge_index[0], edge_index[1]],requires_grad=True)
    adj              = SparseTensor(row=edge_index[0], col=edge_index[1], value=edge_weight,
                          sparse_sizes=( total_labels, total_labels))
    return x_da_f,   adj




def gc_data(n_labels, adj_type, 
            device, emd_type):
    
    
    if 'half' in emd_type:
        emd_dict   = {0: emd_type, 
                      1: emd_type.replace('half', 'full')}
    else:
        emd_dict   = {0: emd_type.replace('full', 'half'), 
                      1: emd_type}
    
    gcn_d          = {label_lvl: get_gcn_data_train(n_labels[label_lvl],
                                                 adj_type, emd_dict[label_lvl], 
                                                 device) for label_lvl in range(len(n_labels))}
    return gcn_d




def gcn_l(n_labels, emd_dim, output_dim, 
          inner_dims, drop, att, device):
    
    
    if isinstance(output_dim, int):
        gcn_layer    = nn.ModuleList([GCN(total_labels = n_labels[label_lvl],
                                          emd_dim      = emd_dim,
                                          out_dim      = output_dim,
                                          inner_dims   = inner_dims,
                                          dropout      = drop,
                                          att= att).to(device) for label_lvl in range(len(n_labels))])
    else:
        gcn_layer    = nn.ModuleList([GCN(total_labels = n_labels[label_lvl],
                                          emd_dim      = emd_dim,
                                          out_dim      = output_dim[label_lvl],
                                          inner_dims   = inner_dims,
                                          dropout      = drop,
                                          att= att).to(device) for label_lvl in range(len(n_labels))])
        
    return gcn_layer



class att_layer(nn.Module):
    def __init__(self, dim_1, dim_2):
        super(att_layer, self).__init__()
        
        # if x is 50 x 300 then u will be 300 x 50
        # dim_1 300, dim_2 50

        self.U = nn.Linear(dim_1, dim_2)
        xavier_uniform(self.U.weight)

    def forward(self, x):
        
        # x : 50  x 300 => 300 x 50
        # u : 300 x 50  => 50  x 300 
        
        x     = torch.tanh(x)
        
        if x.dim() == 3:            
            alpha = F.softmax(self.U.weight.matmul(x.transpose(1, 2)), dim=2)
        else:
            alpha = F.softmax(self.U.weight.matmul(x.t()), dim=1)
        # 50 x 300 <> 300 x 50 ==> 50 x 50

        m = alpha.matmul(x)
        return m


class GCN(torch.nn.Module):
    def __init__(self,
                 total_labels     = 50,
                 emd_dim          = 300,
                 out_dim          = 1024,
                 inner_dims       = 1024,
                 dropout          = False, 
                 att              = False, 
                 dropout_value    = 0.2):
        
        
        super(GCN, self).__init__()
        
        
        self.conv1       = GCNConv(emd_dim, inner_dims)
        self.conv2       = GCNConv(inner_dims, out_dim)
        self.dropout     = dropout
        self.att         = att
        self.drp         = dropout_value
        
        if self.att:
            self.att_laye = att_layer(out_dim, total_labels)
    
    def forward(self, xm):
        x, edge_index = xm
        x             = self.conv1(x, edge_index)
        x             = F.relu(x)
        
        if self.dropout:
            x = F.dropout(x, p=self.drp, training=self.training)
        
        x             = self.conv2(x, edge_index)
        if self.att:
            x = self.att_laye(x)
        return x
    
    
def layer_ops(x, gcn_layer, gcn_data, lavel, linear_layer, con_dim, trans = True):
    
    x_gcn              = gcn_layer[lavel](gcn_data[lavel])
    
    print("x_gcnlevel_", linear_layer[lavel].weight.size(), x_gcn.shape)
    
    weights            = torch.cat((linear_layer[lavel].weight, x_gcn), dim = con_dim)
    if trans:
        weights            = weights.t()
        weights            = weights.expand(x.size()[0],  weights.size()[0], weights.size()[1])
    
    print("x_gcnlevel_return", linear_layer[lavel].weight.size(), x_gcn.shape)
    return weights


def res(data, desired_tensor, device):
    target   = torch.zeros(data.size()[0], desired_tensor.size()[-1]).to(device)
    target[:, :data.size()[-1]] = data
    return target


def layer_ops_sum(x, gcn_layer, gcn_data, lavel, linear_layer, device, con_dim, trans = True):
    
    
    if trans:
        x_gcn              = gcn_layer[lavel](gcn_data[lavel]).t()
        li_data            = linear_layer[lavel].weight.t()
    else:
        x_gcn              = gcn_layer[lavel](gcn_data[lavel])
        li_data            = linear_layer[lavel].weight
         
    if x_gcn.size() != li_data.size():
        if x_gcn.size()[-1] < li_data.size()[-1]:
            x_gcn   = res(x_gcn, li_data, device)
            
        else:
            li_data = res(li_data, x_gcn, device)
    
    weights            = x_gcn + li_data    
    if trans:
        weights            = weights.t()
        weights            = weights.expand(x.size()[0],  weights.size()[0], weights.size()[1])

    return weights


def layer_ops_matmul(x, gcn_layer, gcn_data, lavel, linear_layer, trans = True):
    
    x_gcn              = gcn_layer[lavel](gcn_data[lavel]).t()
    lin                = linear_layer[lavel].weight.t()
    
    weights            = lin @ x_gcn
    
    if trans:
        weights            = weights.expand(x.size()[0],  weights.size()[0], weights.size()[1])
    return weights


def res(data, desired_tensor, device):
    target   = torch.zeros(data.size()[0], desired_tensor.size()[-1]).to(device)
    target[:, :data.size()[-1]] = data
    return target
    
# mm = get_gcn_data_train(50, 'occ_t', 'dim_300_10_full_raw_50_w2v_sum_w2v_no_tfidf', 'cpu', './')
# yu = GCN(dropout = False, att = False)
