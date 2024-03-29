import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution
import pdb

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid[0]) # num of input channels and num of outputchannels
        self.bc1 = nn.BatchNorm1d(nhid[0])

        self.gc2 = GraphConvolution(nhid[0], nhid[1])
        self.bc2 = nn.BatchNorm1d(nhid[1])
        # self.gcls = []
        # self.gcls.append(self.gc1)
        # for i in range(len(nhid)-1):
        #     self.gcls.append(GraphConvolution(nhid[i], nhid[i+1]))
        self.gc3 = GraphConvolution(nhid[1], nhid[2])
        self.bc3 = nn.BatchNorm1d(nhid[2])

        # self.gc4 = GraphConvolution(nhid[2], nhid[3])
        # self.bc4 = nn.BatchNorm1d(nhid[3])

        self.fc = GraphConvolution(nhid[-1], nclass)
        self.activation = nn.PReLU()

        self.mlp = nn.Sequential(
            # nn.Linear(400, 200), 
            # nn.PReLU(),
            nn.Linear(100, 50),
            # nn.BatchNorm1d(50),
            nn.PReLU(),
            # nn.Linear(200, 50),
            # nn.PReLU(),
            # nn.Dropout(),
            nn.Linear(50, nclass))

        self.dropout = dropout

    def forward(self, x, adj):
        # for gcl in self.gcls:
        #     x = F.relu(gcl(x, adj))
        x1 = self.activation(self.bc1(self.gc1(x, adj).permute(0, 2, 1)).permute(0, 2, 1))
        x2 = self.activation(self.bc2(self.gc2(x1, adj).permute(0, 2, 1)).permute(0, 2, 1))
       
        # x = self.activation(self.bc4(self.gc4(x, adj).permute(0, 2, 1)).permute(0, 2, 1))
        x2_dpt = F.dropout(x2, self.dropout, training=self.training) # batch_sizex16x200
        x3 = self.activation(self.bc3(self.gc3(x2_dpt, adj).permute(0, 2, 1)).permute(0, 2, 1))
        # x = F.dropout(x, self.dropout, training=self.training)
        # x = self.activation(self.bc4(self.gc4(x, adj).permute(0, 2, 1)).permute(0, 2, 1))

        flattened_x = torch.cat([torch.mean(x3, dim=1), torch.max(x3, dim=1)[0]], 1)
        # flattened_x_pt1= torch.cat([torch.mean(x3, dim=1), torch.max(x3, dim=1)[0]], 1) # batch_size x 2xlast layer units
        # flattened_x_pt2 = torch.cat([torch.mean(x2, dim=1), torch.max(x2, dim=1)[0]], 1)
        # flattened_x = torch.cat([flattened_x_pt1, flattened_x_pt2], 1)
        # flattened_x = torch.mean(x, dim=1)
        return self.mlp(flattened_x) # batch_size x 2
        # return torch.mean(self.fc(x, adj), dim=1)
