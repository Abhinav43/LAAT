import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_ as xavier_uniform
import json
from math import floor
from torch import nn

class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, kernel_size, stride, use_res, dropout):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv1d(inchannel, outchannel, kernel_size=kernel_size, stride=stride, padding=int(floor(kernel_size / 2)), bias=False),
            nn.BatchNorm1d(outchannel),
            nn.Tanh(),
            nn.Conv1d(outchannel, outchannel, kernel_size=kernel_size, stride=1, padding=int(floor(kernel_size / 2)), bias=False),
            nn.BatchNorm1d(outchannel)
        )

        self.use_res = use_res
        if self.use_res:
            self.shortcut = nn.Sequential(
                        nn.Conv1d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                        nn.BatchNorm1d(outchannel)
                    )

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        out = self.left(x)
        if self.use_res:
            out += self.shortcut(x)
        out = torch.tanh(out)
        out = self.dropout(out)
        return out

class OutputLayer(nn.Module):
    def __init__(self, Y, input_size, cnn_att):
        super(OutputLayer, self).__init__()
        
        self.cnn_att = cnn_att
        
        if self.cnn_att:
            self.U = nn.Linear(input_size, Y)
            xavier_uniform(self.U.weight)
        
        

    def forward(self, x):
        
        if self.cnn_att:
            alpha = F.softmax(self.U.weight.matmul(x.transpose(1, 2)), dim=2)

            m = alpha.matmul(x)
            return m
        else:
            return x

class MultiResCNN(nn.Module):

    def __init__(self, emd_dim, num_filter, cnn_att):
        super(MultiResCNN, self).__init__()
        
        
        feature_size         = emd_dim
        num_filter_maps      = num_filter
        filter_size         = "3,5,9,15,19,25"
        self.conv_layer      = 1
        
        self.word_rep = {1: [feature_size, num_filter_maps],
                         2: [feature_size, 100, num_filter_maps],
                         3: [feature_size, 150, 100, num_filter_maps],
                         4: [feature_size, 200, 150, 100, num_filter_maps]
                         }

        self.conv    = nn.ModuleList()
        filter_sizes = filter_size.split(',')

        self.filter_num = len(filter_sizes)
        for filter_size in filter_sizes:
            filter_size = int(filter_size)
            one_channel = nn.ModuleList()
            tmp = nn.Conv1d(feature_size, feature_size, kernel_size=filter_size,
                            padding=int(floor(filter_size / 2)))
            xavier_uniform(tmp.weight)
            one_channel.add_module('baseconv', tmp)

            conv_dimension = self.word_rep[self.conv_layer]
            for idx in range(self.conv_layer):
                tmp = ResidualBlock(conv_dimension[idx], conv_dimension[idx + 1], filter_size, 1, True,
                                    0.2)
                one_channel.add_module('resconv-{}'.format(idx), tmp)

            self.conv.add_module('channel-{}'.format(filter_size), one_channel)

        self.output_layer = OutputLayer(4000, self.filter_num * num_filter_maps, cnn_att)


    def forward(self, x):

        x = x.transpose(1, 2)

        conv_result = []
        for conv in self.conv:
            tmp = x
            for idx, md in enumerate(conv):
                if idx == 0:
                    tmp = torch.tanh(md(tmp))
                else:
                    tmp = md(tmp)
            tmp = tmp.transpose(1, 2)
            conv_result.append(tmp)
        x = torch.cat(conv_result, dim=2)
        y = self.output_layer(x)

        return y

    def freeze_net(self):
        for p in self.word_rep.embed.parameters():
            p.requires_grad = False
