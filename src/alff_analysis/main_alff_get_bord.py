from matplotlib.pyplot import axis
import numpy as np
import os
#from bfp_utils import load_bfp_data
import scipy.io as spio

measure = 'ALFF_Z'
studydir = '/ImagePTE1/ajoshi/maryland_rao_v1_bfp'
epi_txt = '/ImagePTE1/ajoshi/fitbir/preproc/maryland_rao_v1_epilepsy_imgs.txt'
nonepi_txt = '/ImagePTE1/ajoshi/fitbir/preproc/maryland_rao_v1_nonepilepsy_imgs_37.txt'
nonepi_train_txt = '/ImagePTE1/ajoshi/fitbir/preproc/maryland_rao_v1_nonepilepsy_imgs_training.txt'

with open(epi_txt) as f:
    epiIds = f.readlines()

with open(nonepi_txt) as f:
    nonepiIds = f.readlines()

with open(nonepi_train_txt) as f:
    nonepitrainIds = f.readlines()


epiIds = list(map(lambda x: x.strip(), epiIds))
nonepiIds = list(map(lambda x: x.strip(), nonepiIds))
nonepitrainIds = list(map(lambda x: x.strip(), nonepitrainIds))

epi_files = list()
nonepi_files = list()
nonepitrain_files = list()

epi_ids = list()
nonepi_ids = list()
nonepitrain_ids = list()

for sub in epiIds:

    fname = os.path.join(studydir, sub, 'func', sub +
                         '_rest_bold_'+measure+'.BOrd.mat')

    if os.path.isfile(fname):
        epi_files.append(fname)
        epi_ids.append(sub)
    else:
        print('File does not exist: %s' % fname)

for sub in nonepiIds:

    fname = os.path.join(studydir, sub, 'func', sub +
                         '_rest_bold_'+measure+'.BOrd.mat')

    if os.path.isfile(fname):
        nonepi_files.append(fname)
        nonepi_ids.append(sub)
    else:
        print('File does not exist: %s' % fname)

for sub in nonepitrainIds:

    fname = os.path.join(studydir, sub, 'func', sub +
                         '_rest_bold_'+measure+'.BOrd.mat')

    if os.path.isfile(fname):
        nonepitrain_files.append(fname)
        nonepitrain_ids.append(sub)
    else:
        print('File does not exist: %s' % fname)

nsub = min(len(epi_files), len(nonepi_files))
epi_ids = epi_ids[:nsub]
epi_files = epi_files[:nsub]

nonepi_ids = nonepi_ids[:nsub]
nonepi_files = nonepi_files[:nsub]

num_bord = 46961

# calculate vertiexwise mean and variance for trainnon epi data

nsub_nonepi_train = len(nonepitrainIds)
# fmri diff for epilepsy
data = np.zeros((num_bord, nsub_nonepi_train))

for i, fname in enumerate(nonepitrain_files):
    data[:, i] = spio.loadmat(fname)['dtseries'].squeeze()


np.savez('NONPTE_TRAINING_'+measure+'_BOrd.npz',
         data=data,
         sub_ids=nonepitrain_ids)


nsub_epi = len(epiIds)
data = np.zeros((num_bord, nsub_epi))

for i, fname in enumerate(epi_files):
    data[:, i] = spio.loadmat(fname)['dtseries'].squeeze()

np.savez('PTE_'+measure+'_BOrd.npz', data=data)

nsub_nonepi = len(nonepiIds)
data = np.zeros((num_bord, nsub_nonepi))

for subno in range(nsub_nonepi):
    data[:, i] = spio.loadmat(fname)['dtseries'].squeeze()


np.savez('NONPTE_'+measure+'_BOrd.npz', data=data, sub_ids=nonepi_ids)


print('done')
