import nilearn.image as ni
import os
from bfp_utils import load_bfp_data
from brainsync import groupBrainSync, normalizeData
import numpy as np
import time


studydir = '/ImagePTE1/ajoshi/fitbir/preproc/maryland_rao_v1'
nonpte_training_txt = '/ImagePTE1/ajoshi/fitbir/preproc/maryland_rao_v1_epilepsy_imgs.txt'


with open(nonpte_training_txt) as f:
    subids = f.readlines()

subids = list(map(lambda x: x.strip(), subids))

sub_files = list()
for sub in subids:
    fname = os.path.join(studydir, sub, 'BFP', sub, 'func',
                         sub + '_rest_bold.32k.GOrd.mat')

    if os.path.isfile(fname):
        sub_files.append(fname)
    else:
        print('File does not exist: %s' % fname)


nonepi_data = load_bfp_data(sub_files, 171)

t = time.time()
X2, Os, Costdif, TotalError = groupBrainSync(nonepi_data)

elapsed = time.time() - t

np.savez('grp_atlas_unnormalized.npz', X2=X2, Os=Os)

atlas_data, _, _ = normalizeData(X2)

np.savez('grp_atlas.npz', atlas_data=atlas_data)


# print(subids)
