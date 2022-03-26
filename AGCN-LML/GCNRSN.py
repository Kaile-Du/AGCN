
import torch.nn as nn
from torch.nn import Parameter
import torch
import math
import numpy as np





device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def gen_P(A):
    D = torch.pow(A.sum(1).float(), -0.5)
    D = torch.diag(D)
    adj = torch.matmul(torch.matmul(A, D).t(), D)
    return adj
def gen_A1(num_classes, t, adj_file):
    import pickle
    result = pickle.load(open(adj_file, 'rb'))
    _adj = result['adj']
    _nums = result['nums']
    _nums = _nums[:, np.newaxis]
    _adj = _adj / _nums
    return _adj

def gen_A2(num_classes, t,_adj):
    _adj[_adj < t] = 0
    _adj[_adj >= t] = 1

    #ps:这个地方好像跟论文里面的公式有出入，但是它代码是这样写的，我也就按照它代码来处理
    _adj = _adj * 0.25 / (_adj.sum(0, keepdims=True) + 1e-6)
    _adj = _adj + np.identity(num_classes, np.int)
    return _adj

#gen_adj()相当于通过A得到A_hat矩阵
def gen_adj(A):
    D = torch.pow(A.sum(1).float(), -0.5)
    D = torch.diag(D)
    adj = torch.matmul(torch.matmul(A, D).t(), D)
    return adj




class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        # print('GCN is running')
        # print(input.shape)
        # input = torch.Tensor(input)
        input = input.to(device)
        support = torch.matmul(input, self.weight)
        support = support.to(device)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class gnn(nn.Module):
    #in_channel是指词向量的维度，即一个词由300维的向量表示，t表示阈值，adj_file是我们上面生成adj_file的文件地址
    def __init__(self,in_channel=300):
        super(gnn, self).__init__()

        self.gc1 = GraphConvolution(in_channel, 1024)
        self.gc1 = self.gc1.to(device)
        self.gc2 = GraphConvolution(1024, 2048)
        self.gc2 = self.gc2.to(device)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, P, inp):
        # 2层的GCN网络
        adj = P
        # print('p matrix',adj)
        adj = adj.to(device)
        adj = adj.detach()
        h = self.gc1(inp, adj).to(device)
        h = h.to(device)
        h = self.relu(h)
        h = h.to(device)
        h = self.gc2(h, adj)
        h = h.to(device)
        h = h.transpose(0, 1)
        return h




