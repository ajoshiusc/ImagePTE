import nilearn.image as ni
import os
from bfp_utils import load_bfp_data
from brainsync import groupBrainSync, normalizeData
import numpy as np
import time
import scipy.io as spio

studydir = '/ImagePTE1/ajoshi/fitbir/preproc/maryland_rao_v1'
nonpte_training_txt = '/ImagePTE1/ajoshi/fitbir/preproc/maryland_rao_v1_nonepilepsy_imgs_training.txt'


with open(nonpte_training_txt) as f:
    subids = f.readlines()

subids = list(map(lambda x: x.strip(), subids))

sub_files = list()
for subdir in subids:
    sub = os.path.basename(subdir)
    fname = os.path.join(subdir, 'func', sub + '_rest_bold.BOrd.mat')

    if os.path.isfile(fname):
        sub_files.append(fname)
    else:
        print('File does not exist: %s' % fname)

num_sub = len(sub_files)

f = spio.loadmat(sub_files[0])['dtseries']

numT = f.shape[1]
fmri_data = np.zeros((numT, f.shape[0], num_sub))

for i, f in enumerate(tqdm(sub_files)):

    d = spio.loadmat(f)['dtseries'].T
    fmri_data[:, :, i] = d[:numT, :]

t = time.time()
X2, Os, Costdif, TotalError = groupBrainSync(fmri_data)

elapsed = time.time() - t


base = os.path.basename(studydir)
np.savez(base+'_grp_atlas_unnormalized_bord.npz', X2=X2, Os=Os)

atlas_data, _, _ = normalizeData(X2)

np.savez(base+'_grp_atlas_bord.npz', atlas_data=atlas_data)


# print(subids)
