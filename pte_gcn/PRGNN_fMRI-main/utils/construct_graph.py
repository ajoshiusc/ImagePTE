'''
Author: Xiaoxiao Li
Date: 2019/02/24
'''

import os.path as osp
from os import listdir
import glob
import h5py

import torch
import numpy as np
from torch_geometric.data import Data
import networkx as nx
from networkx.convert_matrix import from_numpy_matrix
import multiprocessing
from torch_sparse import coalesce
from torch_geometric.utils import remove_self_loops
from functools import partial
import pdb

def split(data, batch):
    node_slice = torch.cumsum(torch.from_numpy(np.bincount(batch)), 0)
    node_slice = torch.cat([torch.tensor([0]), node_slice])

    row, _ = data.edge_index
    edge_slice = torch.cumsum(torch.from_numpy(np.bincount(batch[row])), 0)
    edge_slice = torch.cat([torch.tensor([0]), edge_slice])

    # Edge indices should start at zero for every graph.
    data.edge_index -= node_slice[batch[row]].unsqueeze(0)

    slices = {'edge_index': edge_slice}
    if data.x is not None:
        slices['x'] = node_slice
    if data.edge_attr is not None:
        slices['edge_attr'] = edge_slice
    if data.y is not None:
        if data.y.size(0) == batch.size(0):
            slices['y'] = node_slice
        else:
            slices['y'] = torch.arange(0, batch[-1] + 2, dtype=torch.long)
    if data.pos is not None:
        slices['pos'] = node_slice

    return data, slices


def cat(seq):
    seq = [item for item in seq if item is not None]
    seq = [item.unsqueeze(-1) if item.dim() == 1 else item for item in seq]
    return torch.cat(seq, dim=-1).squeeze() if len(seq) > 0 else None

class NoDaemonProcess(multiprocessing.Process):
    @property
    def daemon(self):
        return False

    @daemon.setter
    def daemon(self, value):
        pass


class NoDaemonContext(type(multiprocessing.get_context())):
    Process = NoDaemonProcess



def read_data(data_dir):
    # onlyfiles = [f for f in listdir(data_dir) if osp.isfile(osp.join(data_dir, f))]
    # onlyfiles.sort()
    ADHD_data = np.load("/home/wenhuicu/ImagePTE/pte_gcn/PRGNN_fMRI-main/ADHD_parPearson_BCI-DNI.npz")
    TDC_data = np.load("/home/wenhuicu/ImagePTE/pte_gcn/PRGNN_fMRI-main/TDC_parPearson_BCI-DNI.npz")

    batch = []
    y_list = []
    pseudo = []
    edge_att_list, edge_index_list, att_list = [], [], []

    # # parallar computing
    # cores = multiprocessing.cpu_count()
    # pool = multiprocessing.Pool(processes=cores)
    # func = partial(read_sigle_data, data_dir)
    #
    # import timeit
    #
    # start = timeit.default_timer()
    #
    # res = pool.map(func, onlyfiles)
    #
    # pool.close()
    # pool.join()
    #
    # stop = timeit.default_timer()
    #
    # print('Time: ', stop - start)

    label = 1
    temp = ADHD_data
    num_sub_adhd = ADHD_data["conn_mat"].shape[0]
    for j in range(num_sub_adhd + TDC_data["conn_mat"].shape[0]):
        if j >= num_sub_adhd:
            temp = TDC_data
            label = 0
            ind = j - num_sub_adhd
        else:
            ind = j

        res = process_sigle_data(temp, ind)
        edge_att_list.append(res[0])
        edge_index_list.append(res[1]+j*res[-1])
        att_list.append(res[2]) # node attribute
        y_list.append(label)
        batch.append([j]*res[-1])
        pseudo.append(np.diag(np.ones(res[-1])))

    edge_att_arr = np.concatenate(edge_att_list)
    edge_index_arr = np.concatenate(edge_index_list, axis=1)
    att_arr = np.concatenate(att_list, axis=0)
    y_arr = np.stack(y_list)
    pseudo_arr = np.concatenate(pseudo, axis=0)
    #edge_att_torch = torch.from_numpy(edge_att_arr.reshape(len(edge_att_arr), 1)).float()

    edge_att_torch = torch.from_numpy(edge_att_arr).float()
    att_torch = torch.from_numpy(att_arr).float()
    y_torch = torch.from_numpy(y_arr).long()  # classification
    batch_torch = torch.from_numpy(np.hstack(batch)).long()
    edge_index_torch = torch.from_numpy(edge_index_arr).long()
    #data = Data(x=att_torch, edge_index=edge_index_torch, y=y_torch, edge_attr=edge_att_torch)
    pseudo_torch = torch.from_numpy(pseudo_arr).float()

    data = Data(x=att_torch, edge_index=edge_index_torch, y=y_torch, edge_attr=edge_att_torch, pos=pseudo_torch )
    # x: node feature matrix, edge_index: graph connectivity in COO format with shape[2, num_edges], y: labels/targets
    # edge_attr: edge feature matrix with shape[num_e, num_e_feat];
    # pos: node position matrix with shape[num_nodes, num_dimensions]

    data, slices = split(data, batch_torch)

    return data, slices


def process_sigle_data(data, index):
    # key to how we adapt to our model
    # read edge and edge attribute, partial correlation
    pcorr = np.abs(data['partial_mat'][index])
    # only keep the top 10% edges
    th = np.percentile(pcorr.reshape(-1), 95)
    pcorr[pcorr < th] = 0  # set a threshold
    num_nodes = pcorr.shape[0]

    G = from_numpy_matrix(pcorr)
    A = nx.to_scipy_sparse_matrix(G)
    adj = A.tocoo()
    edge_att = np.zeros((len(adj.row)))
    for i in range(len(adj.row)):
        edge_att[i] = pcorr[adj.row[i], adj.col[i]]
    edge_index = np.stack([adj.row, adj.col])
    edge_index, edge_att = remove_self_loops(torch.from_numpy(edge_index).long(), torch.from_numpy(edge_att).float())
    edge_index, edge_att = coalesce(edge_index, edge_att, num_nodes, num_nodes)

    # node attribute, Pearson correlation
    node_att = data["conn_mat"][index]
    # print(edge_att.data.numpy(), edge_index.data.numpy(), node_att, num_nodes)
    # pdb.set_trace()

    return edge_att.data.numpy(), edge_index.data.numpy(), node_att, num_nodes


read_data("")