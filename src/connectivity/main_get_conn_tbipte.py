import os
import time
from multiprocessing import Pool
from shutil import copy, copyfile

import nilearn.image as ni
import numpy as np
import pandas as pd
import scipy as sp
import scipy.stats as ss
from scipy.stats import shapiro
from statsmodels.stats.multitest import fdrcorrection
from statsmodels.stats.weightstats import ttest_ind
# from multivariate. import TBM_t2
from tqdm import tqdm

from brainsync import groupBrainSync, normalizeData
from get_connectivity import get_connectivity
#from statsmodels.stats import wilcoxon
from read_data_utils import load_bfp_data
import scipy.io as spio
import networkx as nx


def main():

    studydir = '/ImagePTE1/ajoshi/fitbir/preproc/maryland_rao_v1'

    epi_txt = '/ImagePTE1/ajoshi/fitbir/preproc/maryland_rao_v1_epilepsy_imgs.txt'
    nonepi_txt = '/ImagePTE1/ajoshi/fitbir/preproc/maryland_rao_v1_nonepilepsy_imgs_37.txt'

    atlas_labels = '/ImagePTE1/ajoshi/code_farm/bfp/supp_data/USCLobes_grayordinate_labels.mat'
    atlas = spio.loadmat(atlas_labels)

    gord_labels = atlas['labels'].squeeze()

    label_ids = np.unique(gord_labels)  # unique label ids
    #label_ids = [301, 300, 401, 400, 101, 100, 201, 200, 501, 500, 900]

    # remove WM label from connectivity analysis
    label_ids = np.setdiff1d(label_ids, (2000, 0))

    with open(epi_txt) as f:
        epiIds = f.readlines()

    with open(nonepi_txt) as f:
        nonepiIds = f.readlines()

    epiIds = list(map(lambda x: x.strip(), epiIds))
    nonepiIds = list(map(lambda x: x.strip(), nonepiIds))

    epi_files = list()
    nonepi_files = list()
    epi_ids = list()
    nonepi_ids = list()

    for sub in epiIds:
        fname = os.path.join(studydir, sub, 'BFP', sub, 'func',
                             sub + '_rest_bold.32k.GOrd.mat')
        if os.path.isfile(fname):
            epi_files.append(fname)
            epi_ids.append(sub)
        else:
            print('File does not exist: %s' % fname)

    for sub in nonepiIds:
        fname = os.path.join(studydir, sub, 'BFP', sub, 'func',
                             sub + '_rest_bold.32k.GOrd.mat')
        if os.path.isfile(fname):
            nonepi_files.append(fname)
            nonepi_ids.append(sub)
        else:
            print('File does not exist: %s' % fname)

    epi_data = load_bfp_data(epi_files, 171)
    nonepi_data = load_bfp_data(nonepi_files, 171)

    nsub = min(epi_data.shape[2], nonepi_data.shape[2])
    epi_ids = epi_ids[:nsub]
    nonepi_ids = nonepi_ids[:nsub]
    conn_mat = np.zeros((len(label_ids), len(label_ids), nsub))
    cent_mat = np.zeros((len(label_ids), nsub))

    for subno in range(nsub):
        conn_mat[:, :, subno] = get_connectivity(epi_data[:, :, subno],
                                                 labels=gord_labels,
                                                 label_ids=label_ids)

        G = nx.convert_matrix.from_numpy_array(np.abs(conn_mat[:, :, subno]))
        cent = nx.eigenvector_centrality(G, weight='weight')
        cent_mat[:, subno] = np.array(list(cent.items()))[:, 1]

    np.savez('PTE_graphs.npz',
             conn_mat=conn_mat,
             label_ids=label_ids,
             labels=gord_labels,
             cent_mat=cent_mat,
             sub_ids=epi_ids)

    conn_mat = np.zeros((len(label_ids), len(label_ids), nsub))
    cent_mat = np.zeros((len(label_ids), nsub))

    for subno in range(nsub):
        conn_mat[:, :, subno] = get_connectivity(nonepi_data[:, :, subno],
                                                 labels=gord_labels,
                                                 label_ids=label_ids)
        G = nx.convert_matrix.from_numpy_array(np.abs(conn_mat[:, :, subno]))
        cent = nx.eigenvector_centrality(G, weight='weight')
        cent_mat[:, subno] = np.array(list(cent.items()))[:, 1]

    np.savez('NONPTE_graphs.npz',
             conn_mat=conn_mat,
             label_ids=label_ids,
             labels=gord_labels,
             cent_mat=cent_mat,
             sub_ids=nonepi_ids)

    print('done')


if __name__ == "__main__":
    main()
