import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution
import pdb
# from torch.nn.modules.batchnorm import _BatchNorm
import math
# from bn_lib.nn.modules import SynchronizedBatchNorm2d
# import settings
from functools import partial

BN_MOM = 0.1
norm_layer = partial(nn.modules.batchnorm.SyncBatchNorm, momentum=BN_MOM)


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.5, proj_drop=0.5):
        super().__init__()
        self.num_heads = num_heads
        assert dim % num_heads == 0
        self.coef = 4
        self.trans_dims = nn.Linear(dim, dim * self.coef)
        self.num_heads = self.num_heads * self.coef
        self.k = 256 // self.coef
        self.linear_0 = nn.Linear(dim * self.coef // self.num_heads, self.k)
        self.linear_1 = nn.Linear(self.k, dim * self.coef // self.num_heads)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim * self.coef, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        # x = x.view(x.shape[0], x.shape[1], 1)
        B, N, C = x.shape

        x = self.trans_dims(x)  # B, N, C
        x = x.view(B, N, self.num_heads, -1).permute(0, 2, 1, 3)

        attn = self.linear_0(x)

        attn = attn.softmax(dim=-2)
        attn = attn / (1e-9 + attn.sum(dim=-1, keepdim=True))
        attn = self.attn_drop(attn)
        x = self.linear_1(attn).permute(0, 2, 1, 3).reshape(B, N, -1)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class External_attention(nn.Module):
    '''
    Arguments:
        c (int): The input and output channel number.
    '''

    def __init__(self, c):
        super(External_attention, self).__init__()

        self.conv1 = nn.Conv2d(c, c, 1)

        self.k = 64
        self.linear_0 = nn.Conv1d(c, self.k, 1, bias=False)

        self.linear_1 = nn.Conv1d(self.k, c, 1, bias=False)
        self.linear_1.weight.data = self.linear_0.weight.data.permute(1, 0, 2)

        self.conv2 = nn.Sequential(
            nn.Conv2d(c, c, 1, bias=False),
            norm_layer(c))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.modules.batchnorm._BatchNorm):

                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        idn = x
        x = self.conv1(x)

        b, c, h, w = x.size()
        n = h * w
        x = x.view(b, c, h * w)  # b * c * n

        attn = self.linear_0(x)  # b, k, n
        attn = F.softmax(attn, dim=-1)  # b, k, n

        attn = attn / (1e-9 + attn.sum(dim=1, keepdim=True))  # # b, k, n
        x = self.linear_1(attn)  # b, c, n

        x = x.view(b, c, h, w)
        x = self.conv2(x)
        x = x + idn
        x = F.relu(x)
        return x



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
        self.EA_module = MultiHeadAttention(200, 8)

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
        att_x = self.EA_module(x2_dpt)
        x3 = self.activation(self.bc3(self.gc3(att_x, adj).permute(0, 2, 1)).permute(0, 2, 1))
        # x = F.dropout(x, self.dropout, training=self.training)
        # x = self.activation(self.bc4(self.gc4(x, adj).permute(0, 2, 1)).permute(0, 2, 1))

        flattened_x = torch.cat([torch.mean(x3, dim=1), torch.max(x3, dim=1)[0]], 1)
        # print(flattened_x.shape)
        # pdb.set_trace()
        # flattened_x_pt1= torch.cat([torch.mean(x3, dim=1), torch.max(x3, dim=1)[0]], 1) # batch_size x (2 xlast layer units)
        # flattened_x_pt2 = torch.cat([torch.mean(x2, dim=1), torch.max(x2, dim=1)[0]], 1)
        # flattened_x = torch.cat([flattened_x_pt1, flattened_x_pt2], 1)
        # flattened_x = torch.mean(x, dim=1)

        return self.mlp(flattened_x) # batch_size x 2
        # return torch.mean(self.fc(x, adj), dim=1)
