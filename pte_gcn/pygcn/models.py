import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid[0]) # num of input channels and num of outputchannels
        self.gc2 = GraphConvolution(nhid[0], nhid[1])
        # self.gcls = []
        # self.gcls.append(self.gc1)
        # for i in range(len(nhid)-1):
        #     self.gcls.append(GraphConvolution(nhid[i], nhid[i+1]))
        # self.gc3 = GraphConvolution(nhid[1], nhid[2])
        # self.gc4 = GraphConvolution(nhid[2], nhid[3])
        self.fc = GraphConvolution(nhid[-1], nclass)

        self.dropout = dropout

    def forward(self, x, adj):
        # for gcl in self.gcls:
        #     x = F.relu(gcl(x, adj))
        x = F.relu(self.gc1(x, adj))
        x = F.relu(self.gc2(x, adj))
        # x = F.relu(self.gc3(x, adj))
        # x = F.relu(self.gc4(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        return self.fc(x, adj)
