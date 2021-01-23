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


    txt = '/ImagePTE1/ajoshi/fitbir/preproc/tracktbi_pilot_nonepilepsy_imgs_training.txt'

    with open(txt) as f:
        subIds = f.readlines()

    subIds = list(map(lambda x: x.strip(), subIds))

    sub_files = list()


    p = pd.read_csv(
        '/ImagePTE1/ajoshi/fitbir/tracktbi_pilot/Baseline Subject_246/TrackTBI_Subject_modified.csv',
        index_col='Main.GUID')

    #p.loc["TBI_INVVL624DAG"]['Main Group.AgeYrs']
    #p.loc["TBI_INVVL624DAG"]['Subject Demographics.GenderTyp']

    age = list(p.loc[subIds]['Main.AgeYrs'])
    gender = list(p.loc[subIds]['Subject Demographics.GenderTyp'])

    df = pd.DataFrame(list(zip(subIds, age, gender)),
                      columns=['subID', 'Age', 'Gender'])

    df.to_csv('ImagePTE_TrackTBI_pilot_testing_demographics.csv', index=False)

    print('done')


if __name__ == "__main__":
    main()
