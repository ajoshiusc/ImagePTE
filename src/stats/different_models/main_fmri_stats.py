import os
from multiprocessing import Pool
import numpy as np
from shutil import copyfile, copy
import time
import nilearn.image as ni
#from multivariate. import TBM_t2
from tqdm import tqdm
import scipy as sp
import numpy as np
from statsmodels.stats.multitest import fdrcorrection
from statsmodels.stats.weightstats import ttest_ind
import scipy.stats as ss
from scipy.stats import shapiro
#from statsmodels.stats import wilcoxon
from read_data_utils import load_bfp_data
from brainsync import groupBrainSync, normalizeData
import time
import pandas as pd

def main():

    studydir = '/ImagePTE1/ajoshi/fitbir/preproc/maryland_rao_v1'

    epi_txt = '/ImagePTE1/ajoshi/fitbir/preproc/maryland_rao_v1_epilepsy_imgs.txt'
    nonepi_txt = '/ImagePTE1/ajoshi/fitbir/preproc/maryland_rao_v1_nonepilepsy_imgs_37.txt'

    with open(epi_txt) as f:
        epiIds = f.readlines()

    with open(nonepi_txt) as f:
        nonepiIds = f.readlines()

    epiIds = list(map(lambda x: x.strip(), epiIds))
    nonepiIds = list(map(lambda x: x.strip(), nonepiIds))

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

    t = time.time()
    X2, Os, Costdif, TotalError = groupBrainSync(nonepi_data)

    elapsed = time.time() - t

    np.savez('grp_atlas2.npz', X2=X2, Os=Os)

    atlas_data, _, _ = normalizeData(np.mean(X2, axis=1))

    np.savez('grp_atlas.npz', atlas_data=atlas_data)

    # Do Pointwise stats
    #    pointwise_stats(epi_data, nonepi_data)

    vis_grayord_sigcorr(pval, rval, cf.outname, cf.out_dir,
                        int(cf.smooth_iter), cf.save_surfaces, cf.save_figures,
                        'True')

    print('done')


if __name__ == "__main__":
    main()
