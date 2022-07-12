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
#%%

# aug_times = 30

def get_connectivity(data, labels, label_ids): # compute adj matrix
    if type(data) == str:
        df = spio.loadmat(data)
        data = df['dtseries'].T

    num_time = data.shape[0]
    subseq_len = num_time // 2
    slide_len = subseq_len // 4
    aug_times = (num_time - subseq_len) // slide_len + 1
    num_rois = len(label_ids)

    rtseries = np.zeros((aug_times, subseq_len, num_rois)) # 171x16/ 95 /158
    partial_corrM = np.zeros((aug_times, num_rois, num_rois))
    conn = np.zeros((aug_times, num_rois, num_rois))

    j = 0
    for k in range(aug_times):
        # start_point = np.random.randint(0, num_time - subseq_len)
        for i, id in enumerate(label_ids):
            ##============================
            idx = labels == id
            if num_time - j < subseq_len:
                j = num_time - subseq_len
            rtseries[k, :, i] = np.mean(data[j:j+subseq_len, idx], axis=1)
        j = j + slide_len
            # pdb.set_trace()

        rtseries[k], _, _ = normalizeData(rtseries[k])

        ##================================================================##
        partial_measure = ConnectivityMeasure(kind='partial correlation')
        partial_corrM[k] = partial_measure.fit_transform([rtseries[k]])[0]
        ##================================================================##

        conn[k] = np.corrcoef(rtseries[k].T)
        conn[~np.isfinite(conn)] = 0  # define the infinite value edges as no connection

        ##===================Added===========================##
        for i in range(conn[k].shape[0]):
            conn[k, i, i] = 1.0
            for j in range(conn[k].shape[1]):
                conn[k, i, j] = conn[k, j, i]
    ##================##
    ## the adjacency matrix here is not binary. we use the correlation coefficient directly.
    #print(conn.shape, rtseries.T.shape)

    return conn, partial_corrM, np.transpose(rtseries, (0, 2, 1)) # 16x171, ROI/Node. 16*16 for conn


def load_all_data(studydir, epi_txt, test_epi_txt, nonepi_txt, test_nonepi_txt, atlas_labels):

    atlas = spio.loadmat(atlas_labels)

    gord_labels = atlas['labels'].squeeze()

    label_ids = np.unique(gord_labels)  # unique label ids

    # remove WM label from connectivity analysis
    label_ids = np.setdiff1d(label_ids, (2000, 0))

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

    epi_data = load_bfp_data(epi_files, 171)
    nonepi_data = load_bfp_data(nonepi_files, 171)

    # nsub = epi_data.shape[2]
    #==============================================================
    nsub = min(epi_data.shape[2], nonepi_data.shape[2])
    epiIds = epiIds[:nsub]
    nonepiIds = nonepiIds[:nsub]
    #===============================================================
    num_time = epi_data.shape[0]
    subseq_len = num_time // 2
    slide_len = subseq_len // 4
    aug_times = (num_time - subseq_len) // slide_len + 1

    conn_mat = np.zeros((nsub*aug_times, len(label_ids), len(label_ids)))
    parcorr_mat = np.zeros((nsub*aug_times, len(label_ids), len(label_ids)))
    cent_mat = np.zeros((nsub*aug_times, len(label_ids)))
    input_feat = np.zeros((nsub*aug_times, len(label_ids), epi_data.shape[0]))
    print(conn_mat.shape, input_feat.shape)
    print(epi_data.shape, nonepi_data.shape, gord_labels.shape)

    _,_, ref_sub = get_connectivity(nonepi_data[:, :, 0],
                            labels=gord_labels,
                            label_ids=label_ids)


    for subno in range(nsub): # num of subjects
        conn_mat[subno:subno+aug_times, :, :], parcorr_mat[subno:subno+aug_times, :, :], \
        time_series = get_connectivity(epi_data[:, :, subno],
                                                 labels=gord_labels,
                                                 label_ids=label_ids)
        #G = nx.convert_matrix.from_numpy_array(np.abs(conn_mat[subno, :, :]))
        #cent = nx.eigenvector_centrality(G, weight='weight')
        #cent_mat[subno, :] = np.array(list(cent.items()))[:,1]
        # print(ref_sub.shape, time_series.shape)
        # input_feat[subno, :, :, :] = np.transpose(brainSync(ref_sub.T, time_series.T)[0])
        # input_feat[subno, :, :, :] = time_series

    np.savez('../PTE_parPearson_BCI-DNI_ovlp.npz',
             conn_mat=conn_mat,
             partial_mat=parcorr_mat,
             features=input_feat, # 36x16x171
             label_ids=label_ids,
             cent_mat=cent_mat)
##============================================================================
    print("non_epi")
    # nsub = nonepi_data.shape[2]

    conn_mat = np.zeros((nsub * aug_times, len(label_ids), len(label_ids)))
    parcorr_mat = np.zeros((nsub * aug_times, len(label_ids), len(label_ids)))
    cent_mat = np.zeros((nsub * aug_times, len(label_ids)))
    input_feat = np.zeros((nsub * aug_times, len(label_ids), epi_data.shape[0]))
    print(conn_mat.shape, input_feat.shape)
    # here we are using same number of training subjects for epi and nonepi.
    for subno in range(nsub):
        conn_mat[subno:subno+aug_times, :, :], parcorr_mat[subno:subno+aug_times, :, :], \
        time_series = get_connectivity(nonepi_data[:, :, subno],
                                                 labels=gord_labels,
                                                 label_ids=label_ids)
        #G = nx.convert_matrix.from_numpy_array(np.abs(conn_mat[subno, :, :]))
        #cent = nx.eigenvector_centrality(G, weight='weight')
       # cent_mat[subno, :] = np.array(list(cent.items()))[:,1]
       #  input_feat[subno, :, :] = np.transpose(brainSync(ref_sub.T, time_series.T)[0])

    #We are not using time series directly as input features, so the input feature here is just zeros.
    np.savez('../NONPTE_parPearson_BCI-DNI_ovlp.npz',
             conn_mat=conn_mat, # n_subjects*16*16
             partial_mat=parcorr_mat,
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
    nonepi_txt = '/ImagePTE1/ajoshi/fitbir/preproc/maryland_rao_v1_nonepilepsy_imgs_37.txt'
    # atlas_labels = '/ImagePTE1/ajoshi/code_farm/bfp/supp_data/USCBrain_grayordinate_labels.mat'
    # atlas_labels = '/ImagePTE1/ajoshi/code_farm/bfp/supp_data/USCLobes_grayordinate_labels.mat'
    atlas_labels = '../../BCI-DNI_brain_grayordinate_labels.mat'
    load_all_data(studydir, epi_txt, test_epi_txt, nonepi_txt, test_nonepi_txt, atlas_labels)
    input('press any key')