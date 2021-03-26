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
#%%


def get_connectivity(data, labels, label_ids): # compute adj matrix
    #%%
    if type(data) == str:
        df = spio.loadmat(data)
        data = df['dtseries'].T

    num_time = data.shape[0]

    num_rois = len(label_ids)

    rtseries = np.zeros((num_time, num_rois)) # 171x158
    cent_coords = np.zeros(num_rois)

    for i, id in enumerate(label_ids):

        idx = labels == id
        rtseries[:, i] = np.mean(data[:, idx], axis=1)
        coords = np.where(idx == True)[0]
        cent_coords[i] = coords[len(coords)//2]
        # pdb.set_trace()

    # rtseries, _, _ = normalizeData(rtseries) 
    
    conn = abs(np.corrcoef(rtseries.T))

    conn[~np.isfinite(conn)] = 0  # define the infinite value edges as no connection

    ##======Added=======##
    for i in range(conn.shape[0]):
        conn[i, i] = 1.0
        for j in range(conn.shape[1]):
            conn[i, j] = conn[j, i]
    ##================##
    ## the adjacency matrix here is not binary. we use the correlation coefficient directly.
    #print(conn.shape, rtseries.T.shape)
    return conn, rtseries.T, cent_coords # 16x171, ROI/Node. 16*16 for conn


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
    print(epi_data.shape)
    pdb.set_trace()
    nonepi_data = load_bfp_data(nonepi_files, 171)

    # nsub = epi_data.shape[2]
    #==============================================================
    nsub = min(epi_data.shape[2], nonepi_data.shape[2])
    epiIds = epiIds[:nsub]
    nonepiIds = nonepiIds[:nsub]
    #===============================================================

    conn_mat = np.zeros((nsub, len(label_ids), len(label_ids)))
    cent_mat = np.zeros((nsub, len(label_ids)))
    input_feat = np.zeros((nsub, len(label_ids), epi_data.shape[0]))
    cent_coords = np.zeros((nsub, len(label_ids)))

    _, ref_sub, _ = get_connectivity(nonepi_data[:, :, 0],
                            labels=gord_labels,
                            label_ids=label_ids)


    for subno in range(nsub): # num of subjects
        conn_mat[subno, :, :], time_series, cent_coords[subno, :] = get_connectivity(epi_data[:, :, subno],
                                                 labels=gord_labels,
                                                 label_ids=label_ids)

        #G = nx.convert_matrix.from_numpy_array(np.abs(conn_mat[subno, :, :]))
        #cent = nx.eigenvector_centrality(G, weight='weight')
        #cent_mat[subno, :] = np.array(list(cent.items()))[:,1]
        # print(ref_sub.shape, time_series.shape)
        # input_feat[subno, :, :] = time_series
        input_feat[subno, :, :] = np.transpose(brainSync(ref_sub.T, time_series.T)[0])

    print(cent_coords, cent_coords.shape)

    np.savez('PTE_graphs_gcn_hcf.npz',
             conn_mat=conn_mat,
             features=input_feat, # 36x16x171
             label_ids=label_ids,
             cent_mat=cent_mat,
             cent_coords=cent_coords)
##============================================================================
    print("non_epi")
    # nsub = nonepi_data.shape[2]

    conn_mat = np.zeros((nsub, len(label_ids), len(label_ids)))
    cent_mat = np.zeros((nsub, len(label_ids)))
    input_feat = np.zeros((nsub, len(label_ids), nonepi_data.shape[0]))
    cent_coords = np.zeros((nsub, len(label_ids)))
    # here we are using same number of training subjects for epi and nonepi.
    for subno in range(nsub):
        conn_mat[subno, :, :], time_series, cent_coords[subno, :] = get_connectivity(nonepi_data[:, :, subno],
                                                 labels=gord_labels,
                                                 label_ids=label_ids)
        #G = nx.convert_matrix.from_numpy_array(np.abs(conn_mat[subno, :, :]))
        #cent = nx.eigenvector_centrality(G, weight='weight')
       # cent_mat[subno, :] = np.array(list(cent.items()))[:,1]
        # input_feat[subno, :, :] = time_series
        input_feat[subno, :, :] = np.transpose(brainSync(ref_sub.T, time_series.T)[0])

    np.savez('NONPTE_graphs_gcn_hcf.npz',
             conn_mat=conn_mat, # n_subjects*16*16
             features=input_feat, # n_subjects * 16 x 171
             label_ids=label_ids,
             cent_mat=cent_mat,
             cent_coords=cent_coords) # n_subx16

    print('done')


if __name__ == "__main__":

     # BFPPATH = '/ImagePTE1/ajoshi/code_farm/bfp'
     # BrainSuitePath = '/home/ajoshi/BrainSuite19b/svreg'
     # NDim = 31

     # p_dir = '/ImagePTE1/ajoshi/fitbir/preproc/maryland_rao_v1/TBI_INVZV163RWK/BFP/TBI_INVZV163RWK/func/'
     # sub = 'TBI_INVZV163RWK'
     # atlas_labels = '/ImagePTE1/ajoshi/code_farm/bfp/supp_data/USCBrain_grayordinate_labels.mat'

     # atlas = spio.loadmat(atlas_labels)

     # gord_labels = atlas['labels'].squeeze()

     # label_ids = np.unique(gord_labels)  # unique label ids

     # # remove WM label from connectivity analysis
     # label_ids = np.setdiff1d(label_ids, 2000)  # 158 ROIs

     # fname = os.path.join(p_dir, sub + '_rest_bold.32k.GOrd.mat')
#     conn = get_connectivity(fname, gord_labels, label_ids)

    studydir = '/ImagePTE1/ajoshi/fitbir/preproc/maryland_rao_v1'

    # epi_txt = '/ImagePTE1/ajoshi/fitbir/preproc/maryland_rao_v1_epilepsy.txt'
    test_epi_txt = '/ImagePTE1/ajoshi/fitbir/preproc/maryland_rao_v1_epilepsy_test.txt'
    # nonepi_txt = '/ImagePTE1/ajoshi/fitbir/preproc/maryland_rao_v1_nonepilepsy_imgs.txt'
    test_nonepi_txt = '/ImagePTE1/ajoshi/fitbir/preproc/maryland_rao_v1_nonepilepsy_test.txt'
    epi_txt = '/ImagePTE1/ajoshi/fitbir/preproc/maryland_rao_v1_epilepsy_imgs.txt'
    nonepi_txt = '/ImagePTE1/ajoshi/fitbir/preproc/maryland_rao_v1_nonepilepsy_imgs_37.txt'
    # atlas_labels = '/ImagePTE1/ajoshi/code_farm/bfp/supp_data/USCBrain_grayordinate_labels.mat'
    atlas_labels = '/ImagePTE1/ajoshi/code_farm/bfp/supp_data/USCLobes_grayordinate_labels.mat'
    load_all_data(studydir, epi_txt, test_epi_txt, nonepi_txt, test_nonepi_txt, atlas_labels)
    input('press any key')
