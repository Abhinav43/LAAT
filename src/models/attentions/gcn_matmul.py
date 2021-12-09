import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.gcn import *

adj_type   = 'occ_tboth'
emd_type   = 'dim_300_10_half_raw_50_w2v_avg_w2v_no_tfidf'



# concat version
# self.args.gcn_att
# self.args.drop
# self.args.concat_1_att
# self.args.concat_2_att
# self.args.concat_3_att
# final_att
# change output_size manual in argument_parser



class AttentionLayer(nn.Module):

    def __init__(self,
                 args,
                 size: int,
                 level_projection_size: int = 0,
                 n_labels=None,
                 n_level: int = 1
                 ):
        """
        The init function
        :param args: the input parameters from commandline
        :param size: the input size of the layer, it is normally the output size of other DNN models,
            such as CNN, RNN
        """
        super(AttentionLayer, self).__init__()
        self.attention_mode = args.attention_mode
        self.args = args
        device              = self.args.gpu_id
        
        self.size = size
        # For self-attention: d_a and r are the dimension of the dense layer and the number of attention-hops
        # d_a is the output size of the first linear layer
        self.d_a = args.d_a if args.d_a > 0 else self.size

        # r is the number of attention heads


        
        
        """ GCN data """

        self.gcn_1_data  = gc_data(n_labels, adj_type, device, emd_type)
        self.gcn_2_data  = gc_data(n_labels, adj_type, device, emd_type)
        self.gcn_3_data  = gc_data(n_labels, adj_type, device, emd_type)
        
        
        """ Gcn data """
        
        
        
        
        
        """ Gcn layers """
        
        self.gcn_1_layer = gcn_l(n_labels,  self.gcn_1_data[0][0].shape[-1], 
                                 self.d_a, inner_dims = 1024, 
                                 drop = self.args.gcn_drop, 
                                 att  = self.args.gcn_att, device = device)
        
        self.gcn_2_layer = gcn_l(n_labels,  self.gcn_1_data[0][0].shape[-1], 
                                 n_labels,  inner_dims = 1024, 
                                 drop = self.args.gcn_drop, 
                                 att  = self.args.gcn_att, device = device)
        
        self.gcn_3_layer = gcn_l(n_labels,  self.gcn_1_data[0][0].shape[-1],
                                 n_labels, inner_dims = 1024, 
                                 drop = self.args.gcn_drop, 
                                 att  = self.args.gcn_att, device = device)
        
        """ Gcn layers """
        
        
        
        """ Attention layers concat """
        
        self.concat_1_att    = nn.ModuleList([att_layer(n_labels[label_lvl], 
                                                        self.size) for label_lvl in range(len(n_labels))])
        self.concat_2_att    = nn.ModuleList([att_layer(n_labels[label_lvl],
                                                        n_labels[label_lvl])
                                              for label_lvl in range(len(n_labels))])
        
        self.concat_3_att    = nn.ModuleList([att_layer(self.size + (level_projection_size if label_lvl > 0 else 0),
                                                        n_labels[label_lvl])
                                              for label_lvl in range(len(n_labels))])
        self.final_att       = nn.ModuleList([att_layer(n_labels[label_lvl], self.args.batch_size)
                                              for label_lvl in range(len(n_labels))])
        
        """ Attention layers concat """
        
        

        self.n_labels = n_labels
        self.n_level = n_level
        self.r = [args.r if args.r > 0 else n_labels[label_lvl] for label_lvl in range(n_level)]

        self.level_projection_size = level_projection_size

        self.linear = nn.Linear(self.size, self.size, bias=False)
        if self.attention_mode == "caml":
            self.d_a = self.size
        
        self.first_linears = nn.ModuleList([nn.Linear(self.size, self.d_a, bias=False) for _ in range(self.n_level)])
        self.second_linears = nn.ModuleList([nn.Linear(self.n_labels[label_lvl], self.n_labels[label_lvl], bias=False) for label_lvl in range(self.n_level)])
        self.third_linears = nn.ModuleList([nn.Linear(self.size +
                                           (self.level_projection_size if label_lvl > 0 else 0),
                                           self.n_labels[label_lvl], bias=True) for label_lvl in range(self.n_level)])
        
        
        self._init_weights(mean=0.0, std=0.03)

    def _init_weights(self, mean=0.0, std=0.03) -> None:
        """
        Initialise the weights
        :param mean:
        :param std:
        :return: None
        """
        for first_linear in self.first_linears:
            torch.nn.init.normal(first_linear.weight, mean, std)
            if first_linear.bias is not None:
                first_linear.bias.data.fill_(0)

        for linear in self.second_linears:
            torch.nn.init.normal(linear.weight, mean, std)
            if linear.bias is not None:
                linear.bias.data.fill_(0)
        if self.attention_mode == "label" or self.attention_mode == "caml":
            for linear in self.third_linears:
                torch.nn.init.normal(linear.weight, mean, std)

    def forward(self, x, previous_level_projection=None, label_level=0):
        """
        :param x: [batch_size x max_len x dim (i.e., self.size)]
        :param previous_level_projection: the embeddings for the previous level output
        :param label_level: the current label level
        :return:
            Weighted average output: [batch_size x dim (i.e., self.size)]
            Attention weights
        """
        if self.attention_mode == "caml":
            weights = F.tanh(x)
        else:

            """ First layer attention """
            
            weights        = layer_ops_matmul(x, self.gcn_1_layer, 
                                                   self.gcn_1_data, 
                                                   label_level, 
                                                 self.first_linears)
            
            if self.args.concat_1_att:
                weights            = self.concat_1_att[label_level](weights)
            
            
            weights           = x @ weights
            weights           = F.tanh(weights)
        
            """ End of First layer attention """
        
        """ Second attention layer """
        
        att_weights    = layer_ops_matmul(weights, self.gcn_2_layer, 
                                   self.gcn_2_data, 
                                   label_level, self.second_linears)
        
        if self.args.concat_2_att:
            att_weights    = self.concat_2_att[label_level](att_weights)
            
        att_weights    = weights @ att_weights
        att_weights    = F.softmax(att_weights, 1).transpose(1, 2)
        

        
        """ End of Second attention layer """
        
        if len(att_weights.size()) != len(x.size()):
            att_weights = att_weights.squeeze()
        weighted_output = att_weights @ x
        
        
        batch_size = weighted_output.size(0)


        gcn_weighted_output    = layer_ops_matmul(weighted_output, self.gcn_3_layer, 
                                   self.gcn_3_data, 
                                   label_level, self.third_linears).transpose(1,2)
                        
        if self.args.concat_3_att:
            gcn_weighted_output    = self.concat_3_att[label_level](gcn_weighted_output)
                
        if previous_level_projection is not None:
            temp = [weighted_output,
                    previous_level_projection.repeat(1, 
                                                     self.n_labels[label_level]).view(batch_size, 
                                                                                      self.n_labels[label_level], -1)]
            weighted_output = torch.cat(temp, dim=2)
        
        
        weighted_output     = gcn_weighted_output.mul(weighted_output).sum(dim=2)
                
        if self.args.final_att:
            weighted_output = self.final_att[label_level](weighted_output)
            
        
        return weighted_output, att_weights

    # Using when use_regularisation = True
    @staticmethod
    def l2_matrix_norm(m):
        """
        Frobenius norm calculation
        :param m: {Variable} ||AAT - I||
        :return: regularized value
        """
        return torch.sum(torch.sum(torch.sum(m ** 2, 1), 1) ** 0.5)
