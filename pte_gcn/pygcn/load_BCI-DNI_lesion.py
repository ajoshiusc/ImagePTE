import sys
import os
import numpy as np
import random
sys.path.append('/ImagePTE1/ajoshi/code_farm/ImagePTE/src/stats/')

#from surfproc import patch_color_attrib, smooth_surf_function
from brainsync import normalizeData, groupBrainSync, brainSync

#from dfsio import readdfs, writedfs
from scipy import io as spio
from read_data_utils import load_bfp_data
import pdb
import nilearn.image as ni
from tqdm import tqdm
from sklearn.svm import OneClassSVM
#%%


def get_connectivity(data, labels, labels_lobes, label_ids, label_ids_lobes): # compute adj matrix
    if type(data) == str:
        df = spio.loadmat(data)
        data = df['dtseries'].T

    num_time = data.shape[0]
    num_rois = len(label_ids)

    rtseries = np.zeros((num_time, num_rois))# 171x16/ 95 /158

    for i, id in enumerate(label_ids):
        region = labels == id
        coords = np.where(region == True)[0]
        for id_lobes in label_ids_lobes:
            region_lobes = labels_lobes == id_lobes
            coords_lobes = np.where(region_lobes == True)[0]
            print(coords_lobes, coords)
            if coords in coords_lobes:
                print("yes\n")
            pdb.set_trace()
        idx = labels == id
        rtseries[:, i] = np.mean(data[:, idx], axis=1)

    rtseries, _, _ = normalizeData(rtseries) 

    conn = abs(np.corrcoef(rtseries.T))

    conn[~np.isfinite(conn)] = 0  # define the infinite value edges as no connection

    ##===================Added===========================##
    for i in range(conn.shape[0]):
        conn[i, i] = 1.0
        for j in range(conn.shape[1]):
            conn[i, j] = conn[j, i]
    ##================##
    ## the adjacency matrix here is not binary. we use the correlation coefficient directly.
    #print(conn.shape, rtseries.T.shape)
    return conn, rtseries.T # 16x171, ROI/Node. 16*16 for conn


def load_all_data(studydir, epi_txt, test_epi_txt, nonepi_txt, test_nonepi_txt, atlas_labels, atlas_labels_lobes):

    atlas = spio.loadmat(atlas_labels)
    gord_labels = atlas['labels'].squeeze()
    label_ids = np.unique(gord_labels)  # unique label ids

    atlas_lobes = spio.loadmat(atlas_labels_lobes)
    gord_labels_lobes = atlas_lobes['labels'].squeeze()
    label_ids_lobes = np.unique(gord_labels_lobes)

    # remove WM label from connectivity analysis
    label_ids = np.setdiff1d(label_ids, (2000, 0))
    label_ids_lobes = np.setdiff1d(label_ids_lobes, (2000, 0))
    print(label_ids, len(label_ids))
    pdb.set_trace()
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
    nsub = min(epi_data.shape[2], nonepi_data.shape[2])
    epiIds = epiIds[:nsub]
    nonepiIds = nonepiIds[:nsub]
    #===============================================================

    conn_mat = np.zeros((nsub, len(label_ids), len(label_ids)))
    cent_mat = np.zeros((nsub, len(label_ids)))
    input_feat = np.zeros((nsub, len(label_ids), epi_data.shape[0]))
    print(conn_mat.shape, input_feat.shape)
    print(epi_data.shape, nonepi_data.shape, gord_labels.shape)

    _, ref_sub = get_connectivity(nonepi_data[:, :, 0],
                            labels=gord_labels,
                            labels_lobes=gord_labels_lobes,
                            label_ids=label_ids,
                            label_ids_lobes=label_ids_lobes)


    for subno in range(nsub): # num of subjects
        conn_mat[subno, :, :], time_series = get_connectivity(epi_data[:, :, subno],
                                                 labels=gord_labels,
                                                  labels_lobes=gord_labels_lobes,
                                                  label_ids=label_ids,
                                                  label_ids_lobes=label_ids_lobes)

        #G = nx.convert_matrix.from_numpy_array(np.abs(conn_mat[subno, :, :]))
        #cent = nx.eigenvector_centrality(G, weight='weight')
        #cent_mat[subno, :] = np.array(list(cent.items()))[:,1]
        # print(ref_sub.shape, time_series.shape)
        input_feat[subno, :, :] = np.transpose(brainSync(ref_sub.T, time_series.T)[0])

    np.savez('PTE_graphs_gcn_BCI-DNI.npz',
             conn_mat=conn_mat,
             features=input_feat, # 36x16x171
             label_ids=label_ids,
             cent_mat=cent_mat)
##============================================================================
    print("non_epi")
    # nsub = nonepi_data.shape[2]

    conn_mat = np.zeros((nsub, len(label_ids), len(label_ids)))
    cent_mat = np.zeros((nsub, len(label_ids)))
    input_feat = np.zeros((nsub, len(label_ids), nonepi_data.shape[0]))
    print(conn_mat.shape, input_feat.shape)
    # here we are using same number of training subjects for epi and nonepi.
    for subno in range(nsub):
        conn_mat[subno, :, :], time_series = get_connectivity(nonepi_data[:, :, subno],
                                                 labels=gord_labels,
                                                  labels_lobes=gord_labels_lobes,
                                                  label_ids=label_ids,
                                                  label_ids_lobes=label_ids_lobes)
        #G = nx.convert_matrix.from_numpy_array(np.abs(conn_mat[subno, :, :]))
        #cent = nx.eigenvector_centrality(G, weight='weight')
       # cent_mat[subno, :] = np.array(list(cent.items()))[:,1]
        input_feat[subno, :, :] = np.transpose(brainSync(ref_sub.T, time_series.T)[0])

    np.savez('NONPTE_graphs_gcn_BCI-DNI.npz',
             conn_mat=conn_mat, # n_subjects*16*16
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
    atlas_labels_lobes = '/ImagePTE1/ajoshi/code_farm/bfp/supp_data/USCLobes_grayordinate_labels.mat'
    atlas_labels = '../BCI-DNI_brain_grayordinate_labels.mat'
    # atlas_labels = "/home/ajosshi/BrainSuite21a/svreg/BCI-DNI_brain_atlas/BCI-DNI_brain.label.nii.gz"
    load_all_data(studydir, epi_txt, test_epi_txt, nonepi_txt, test_nonepi_txt, atlas_labels, atlas_labels_lobes)
    input('press any key')
