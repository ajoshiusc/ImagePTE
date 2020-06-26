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

    train_txt = '/ImagePTE1/ajoshi/fitbir/preproc/maryland_rao_v1_nonepilepsy_imgs_training.txt'


    with open(train_txt) as f:
        trainIds = f.readlines()

    trainIds = list(map(lambda x: x.strip(), trainIds))

    train_files = list()

    train_id = list()

    for sub in trainIds:
        fname = os.path.join(studydir, sub, 'BFP', sub, 'func',
                             sub + '_rest_bold.32k.GOrd.mat')
        if os.path.isfile(fname):
            train_files.append(fname)
            train_id.append(sub)

    #epi_data = load_bfp_data(epi_files, 171)
    #train_data = load_bfp_data(train_files, 171)

    ids = train_id
    pte = [0] * len(train_id)

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

    df.to_csv('ImagePTE_Maryland_training_demographics.csv', index=False)

    print('done')


if __name__ == "__main__":
    main()
