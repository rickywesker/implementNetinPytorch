import math
import torch

from torch.nn.parameter import Parameter
from torch.nn.modules import Module

class GraphConvolution(Module):
    """
    Implementation of Graph Convolution

    - Attributes
        - in_features: int
        the size of input features, i.e |H^(l)|
        - out_features: int
        the size of output features, i.e |H^(l+1)|
        - bias: bool
        default as True
        - weight: Parameter
        trainable param in GC

    - Methods
        - reset_parameters(self)
        -forward(self, input, adj)
            - Feed forward func, adjacency matrix after transformation N(A)

    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias',None)
        self.reset_parameters()
    
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv,stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv,stdv)

    def forward(self, input, adj):
        # H * W
        support = torch.mm(input, self.weight)
        # N(A) * H * W
        output = torch.spmm() #sparse matrix mul
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.in_features) + '->' + str(self.out_features) + ')'

import torch.nn as nn
import torch.nn.functional as F


class GCN(Module):

    """
    two layer GCN model
    ...
    Attributes
    -----------
        - n_feat: int
        - n_hid: int
        - n_class: int
        - dropout: float

    Methods
    --------
        - forward(self, x, adj)
    """

    def __init__(self, n_feat, n_hid, n_class, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(n_feat, n_hid)
        self.gc2 = GraphConvolution(n_hid, n_class)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x,adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)
