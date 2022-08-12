import sys
import os
import numpy as np
import pandas as pd
import random
sys.path.append('/ImagePTE1/ajoshi/code_farm/ImagePTE/src/stats/')

#from surfproc import patch_color_attrib, smooth_surf_function
from brainsync import normalizeData, groupBrainSync, brainSync

#from dfsio import readdfs, writedfs
from scipy import io as spio
from read_data_utils import load_bfp_data
import pdb
import nilearn.image as ni
from nilearn.connectome import ConnectivityMeasure
from tqdm import tqdm
from sklearn.svm import OneClassSVM
from sklearn.cross_decomposition import CCA
#%%

def get_connectivity(data, labels, label_ids): # compute adj matrix
    if type(data) == str:
        df = spio.loadmat(data)
        data = df['dtseries'].T

    num_time = data.shape[0]
    num_rois = len(label_ids)
    rtseries = np.zeros((num_time, num_rois))  # 171x16 / 95 / 158
    #   3, 100, 101, 184, 185, 200, 201, 300, 301, 400, 401, 500, 501,                            800, 850, 900
    # 8051
    a = 0
    for i, id in enumerate(label_ids):
        idx = labels == id
        print(i, np.sum(idx))
        a = a + np.sum(idx)
        rtseries[:, i] = np.mean(data[:, idx], axis=1)
    print(a)
    pdb.set_trace()
    rtseries, _, _ = normalizeData(rtseries)
    ##================================================================##
    partial_measure = ConnectivityMeasure(kind='partial correlation')
    partial_corrM = partial_measure.fit_transform([rtseries])[0]

    conn_measure = ConnectivityMeasure(kind='correlation')
    conn_mat = conn_measure.fit_transform([rtseries])[0]
    conn_measure = ConnectivityMeasure(kind='tangent')
    connectivity_fit = conn_measure.fit([conn_mat])
    TPE_connectivity = connectivity_fit.transform([conn_mat])[0]

    connectivity_fit_TE = conn_measure.fit([rtseries])
    TE_connectivity = connectivity_fit_TE.transform([rtseries])[0]
    ##================================================================##

    # conn = np.corrcoef(rtseries.T)
    
    # cca = CCA(n_components=1)
    # # for i in range(rtseries.T.shape[0]):
    # #     for i in range(rtseries.T.)
    # cca.fit(rtseries.T, rtseries.T)

    # conn = cca.transform(rtseries.T)
    # print(conn)
    # pdb.set_trace()

    conn = conn_mat
    conn[~np.isfinite(conn)] = 0  # define the infinite value edges as no connection

    ##===================Added===========================##
    for i in range(conn.shape[0]):
        conn[i, i] = 1.0
        for j in range(conn.shape[1]):
            conn[i, j] = conn[j, i]
    ##================##
    ## the adjacency matrix here is not binary. we use the correlation coefficient directly.
    # print(conn.shape, rtseries.T.shape)
    return conn, partial_corrM, TPE_connectivity, TE_connectivity, rtseries.T  # 16x171, ROI/Node. 16*16 for conn


def load_all_data(studydir, epi_txt, test_epi_txt, nonepi_txt, test_nonepi_txt, atlas_labels):

    atlas = spio.loadmat(atlas_labels)

    gord_labels = atlas['labels'].squeeze()
    
    label_ids = np.unique(gord_labels)  # unique label ids
    # remove WM label from connectivity analysis
    label_ids = np.setdiff1d(label_ids, (2000, 0))
    
    idst = atlas['LookUpTable'][0][0][0][0]
    names = atlas['LookUpTable'][0][0][1][0]
    table = dict()
    for i, idl in enumerate(label_ids):
        tmp = names[np.where(idst == idl)]
        table[i] = tmp
    print(table)
    pdb.set_trace()
    # pick_ids = np.random.randint(0, 156, size=16)
    # label_ids = label_ids[pick_ids]
    

    # for i, tid in enumerate(label_ids):
    #     random_labels[np.where(gord_labels==tid)] = random_label_ids[i]
    # pdb.set_trace()


    with open(epi_txt) as f:
        epiIds = f.readlines()

    with open(nonepi_txt) as f:
        nonepiIds = f.readlines()

    epiIds = list(map(lambda x: x.strip(), epiIds))
    nonepiIds = list(map(lambda x: x.strip(), nonepiIds))
    # random.shuffle(epiIds)
    # random.shuffle(nonepiIds)
    # print(len(epiIds), epiIds)
    epi_files = list()
    nonepi_files = list()

    for sub in epiIds:
        fname = os.path.join(studydir, sub, 'BFP', sub, 'func',
                             sub + '_rest_bold.32k.GOrd.mat')
        if os.path.isfile(fname):
            epi_files.append(fname)

    for sub in nonepiIds:
        fname = os.path.join(studydir, sub, 'BFP', sub, 'func',
                             sub + '_rest_bold.32k.GOrd.mat')
        if os.path.isfile(fname):
            nonepi_files.append(fname)

    epi_data = load_bfp_data(epi_files[:1], 171)
    nonepi_data = load_bfp_data(nonepi_files[:1], 171)

    # nsub = epi_data.shape[2]
    #==============================================================
    # nsub = min(epi_data.shape[2], nonepi_data.shape[2])
    nsub = epi_data.shape[2]

    # epiIds = epiIds[:nsub]
    # nonepiIds = nonepiIds[:nsub]
    #===============================================================

    conn_mat = np.zeros((nsub, len(label_ids), len(label_ids)))
    parcorr_mat = np.zeros((nsub, len(label_ids), len(label_ids)))
    TE_conn = np.zeros((nsub, len(label_ids), len(label_ids)))
    TPE_conn = np.zeros((nsub, len(label_ids), len(label_ids)))
    cent_mat = np.zeros((nsub, len(label_ids)))
    input_feat = np.zeros((nsub, len(label_ids), epi_data.shape[0]))
    print(conn_mat.shape, input_feat.shape)
    print(epi_data.shape, nonepi_data.shape, gord_labels.shape)
    # print(label_ids)
    _,_, _, _, ref_sub = get_connectivity(nonepi_data[:, :, 0],
                            labels=gord_labels,
                            label_ids=label_ids)


    for subno in range(nsub): # num of subjects
        conn_mat[subno, :, :], parcorr_mat[subno, :, :], TPE_conn[subno, :, :], TE_conn[subno, :, :], time_series = get_connectivity(epi_data[:, :, subno],
                                                 labels=gord_labels,
                                                 label_ids=label_ids)
        #G = nx.convert_matrix.from_numpy_array(np.abs(conn_mat[subno, :, :]))
        #cent = nx.eigenvector_centrality(G, weight='weight')
        #cent_mat[subno, :] = np.array(list(cent.items()))[:,1]
        # print(ref_sub.shape, time_series.shape)
        # input_feat[subno, :, :, :] = np.transpose(brainSync(ref_sub.T, time_series.T)[0])
        # input_feat[subno, :, :, :] = time_series

    np.savez('/home/wenhuicu/data_npz/PTE_Allconn_brain_all.npz',
             conn_mat=conn_mat,
             partial_mat=parcorr_mat,
             TPE_mat=TPE_conn,
             TE_mat=TE_conn,
             features=input_feat, # 36x16x171
             label_ids=label_ids,
             cent_mat=cent_mat)
##============================================================================
    print("non_epi")
    nsub = nonepi_data.shape[2]
    conn_mat = np.zeros((nsub, len(label_ids), len(label_ids)))
    parcorr_mat = np.zeros((nsub, len(label_ids), len(label_ids)))
    TE_conn = np.zeros((nsub, len(label_ids), len(label_ids)))
    TPE_conn = np.zeros((nsub, len(label_ids), len(label_ids)))

    cent_mat = np.zeros((nsub, len(label_ids)))
    input_feat = np.zeros((nsub, len(label_ids), epi_data.shape[0]))
    print(conn_mat.shape, input_feat.shape)
    # here we are using same number of training subjects for epi and nonepi.
    for subno in range(nsub):
        conn_mat[subno, :, :], parcorr_mat[subno, :, :], TPE_conn[subno, :, :], TE_conn[subno, :, :], time_series = get_connectivity(nonepi_data[:, :, subno],
                                                 labels=gord_labels,
                                                 label_ids=label_ids)
        #G = nx.convert_matrix.from_numpy_array(np.abs(conn_mat[subno, :, :]))
        #cent = nx.eigenvector_centrality(G, weight='weight')
       # cent_mat[subno, :] = np.array(list(cent.items()))[:,1]
       #  input_feat[subno, :, :] = np.transpose(brainSync(ref_sub.T, time_series.T)[0])

    #We are not using time series directly as input features, so the input feature here is just zeros.
    np.savez('/home/wenhuicu/data_npz/NONPTE_Allconn_brain_all.npz',
             conn_mat=conn_mat, # n_subjects*16*16
             partial_mat=parcorr_mat,
             TPE_mat=TPE_conn,
             TE_mat=TE_conn,
             features=input_feat, # n_subjects * 16 x 171
             label_ids=label_ids,
             cent_mat=cent_mat)

    print('done')


if __name__ == "__main__":

    studydir = '/ImagePTE1/ajoshi/fitbir/preproc/maryland_rao_v1'

    # epi_txt = '/ImagePTE1/ajoshi/fitbir/preproc/maryland_rao_v1_epilepsy.txt'
    test_epi_txt = '/ImagePTE1/ajoshi/fitbir/preproc/maryland_rao_v1_epilepsy_test.txt'
    # nonepi_txt = '/ImagePTE1/ajoshi/fitbir/preproc/maryland_rao_v1_nonepilepsy_imgs.txt'
    test_nonepi_txt = '/ImagePTE1/ajoshi/fitbir/preproc/maryland_rao_v1_nonepilepsy_test.txt'
    
    epi_txt = '/ImagePTE1/ajoshi/fitbir/preproc/maryland_rao_v1_epilepsy_imgs.txt'
    # nonepi_txt = '/ImagePTE1/ajoshi/fitbir/preproc/maryland_rao_v1_nonepilepsy_imgs_37.txt'
    nonepi_txt = '/ImagePTE1/ajoshi/fitbir/preproc/maryland_rao_v1_nonepilepsy_imgs.txt'
    # atlas_labels = '/ImagePTE1/ajoshi/code_farm/bfp/supp_data/USCBrain_grayordinate_labels.mat'
    atlas_labels = '/ImagePTE1/ajoshi/code_farm/bfp/supp_data/USCLobes_grayordinate_labels.mat'
    # atlas_labels = '/ImagePTE1/ajoshi/code_farm/bfp/supp_data/AAL_grayordinate_labels.mat'
    # atlas_labels = '../../BCI-DNI_brain_grayordinate_labels.mat'
    load_all_data(studydir, epi_txt, test_epi_txt, nonepi_txt, test_nonepi_txt, atlas_labels)
    input('press any key')