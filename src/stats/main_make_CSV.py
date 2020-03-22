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
#from read_data_utils import load_bfp_data
#from brainsync import groupBrainSync, normalizeData
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

    epi_id = list()
    nonepi_id = list()

    for sub in epiIds:
        fname = os.path.join(studydir, sub, 'BFP', sub, 'func',
                             sub + '_rest_bold.32k.GOrd.mat')
        if os.path.isfile(fname):
            epi_files.append(fname)
            epi_id.append(sub)

    for sub in nonepiIds:
        fname = os.path.join(studydir, sub, 'BFP', sub, 'func',
                             sub + '_rest_bold.32k.GOrd.mat')
        if os.path.isfile(fname):
            nonepi_files.append(fname)
            nonepi_id.append(sub)

    #epi_data = load_bfp_data(epi_files, 171)
    #nonepi_data = load_bfp_data(nonepi_files, 171)

    ids = epi_id + nonepi_id
    pte = [1] * len(epi_id) + [0] * len(nonepi_id)

    exclude = np.zeros(len(ids), np.int16)

    p = pd.read_csv(
        '/ImagePTE1/ajoshi/fitbir/maryland_rao/FITBIR Demographics_314/FITBIRdemographics_prospective_modified.csv',
        index_col='Main Group.GUID')

    #p.loc["TBI_INVVL624DAG"]['Main Group.AgeYrs']
    #p.loc["TBI_INVVL624DAG"]['Subject Demographics.GenderTyp']

    age = list(p.loc[ids]['Main Group.AgeYrs'])
    gender = list(p.loc[ids]['Subject Demographics.GenderTyp'])

    df = pd.DataFrame(list(zip(ids, pte, age, gender, exclude)),
                      columns=['subID', 'PTE', 'Age', 'Gender', 'Exclude'])

    df.to_csv('ImagePTE_Maryland_demographics.csv', index=False)

    print('done')


if __name__ == "__main__":
    main()
