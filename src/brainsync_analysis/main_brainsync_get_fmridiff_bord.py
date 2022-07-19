from matplotlib.pyplot import axis
import numpy as np
from brainsync import normalizeData, brainSync
import os
from bfp_utils import load_bfp_data
import scipy.io as spio

# get atlas
atlas_data = np.load('maryland_rao_v1_bfp_grp_atlas_bord.npz')['atlas_data']  # , X2=X2, Os=Os)
atlas_data2, _, _ = normalizeData(atlas_data)
print(np.max(atlas_data2-atlas_data))


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

    fname = os.path.join(studydir, sub, 'func', sub + '_rest_bold.BOrd.mat')

    if os.path.isfile(fname):
        epi_files.append(fname)
        epi_ids.append(sub)
    else:
        print('File does not exist: %s' % fname)

for sub in nonepiIds:

    fname = os.path.join(studydir, sub, 'func', sub + '_rest_bold.BOrd.mat')

    if os.path.isfile(fname):
        nonepi_files.append(fname)
        nonepi_ids.append(sub)
    else:
        print('File does not exist: %s' % fname)

for sub in nonepitrainIds:

    fname = os.path.join(studydir, sub, 'func', sub + '_rest_bold.BOrd.mat')

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


epi_data = load_bfp_data(epi_files, 171)
nonepi_data = load_bfp_data(nonepi_files, 171)
nonepitrain_data = load_bfp_data(nonepitrain_files, 171)


'''nsub = min(epi_data.shape[2], nonepi_data.shape[2])
epi_ids = epi_ids[:nsub]
nonepi_ids = nonepi_ids[:nsub]
#nonepitrain_ids = nonepitrain_ids[:nsub]
'''

num_vtx = epi_data.shape[1]


# calculate vertiexwise mean and variance for trainnon epi data

nsub_nonepi_train = nonepitrain_data.shape[2]
# fmri diff for epilepsy
fdiff_sub = np.zeros((num_vtx, nsub_nonepi_train))

for subno in range(nsub_nonepi_train):
    d, _ = brainSync(atlas_data, nonepitrain_data[:, :, subno])
    fdiff_sub[:, subno] =  np.linalg.norm(atlas_data - d,axis=0)


np.savez('NONPTE_TRAINING_fmridiff_BOrd.npz',
         fdiff_sub=fdiff_sub,
         sub_ids=nonepitrain_ids)


fdiff_mean = np.mean(fdiff_sub, axis=1)
fdiff_std = np.std(fdiff_sub, axis=1)


nsub_epi = epi_data.shape[2]
# fmri diff for epilepsy
fdiff_sub = np.zeros((num_vtx, nsub_epi))
fdiff_sub_z = np.zeros((num_vtx, nsub_epi))

for subno in range(nsub_epi):
    d, _ = brainSync(atlas_data, epi_data[:, :, subno])
    data = np.linalg.norm(atlas_data - d,axis=0)
    fdiff_sub[:, subno] = data
    fdiff_sub_z[:, subno] = (data - fdiff_mean)/fdiff_std

np.savez('PTE_fmridiff_BOrd.npz',
         fdiff_sub=fdiff_sub,
         fdiff_sub_z=fdiff_sub_z,
         sub_ids=epi_ids)


# fmri diff for nonepilepsy
nsub_nonepi = nonepi_data.shape[2]
fdiff_sub = np.zeros((num_vtx, nsub_nonepi))
fdiff_sub_z = np.zeros((num_vtx, nsub_nonepi))

for subno in range(nsub_nonepi):
    d, _ = brainSync(atlas_data, nonepi_data[:, :, subno])
    data = np.linalg.norm(atlas_data - d,axis=0)
    fdiff_sub[:, subno] = data
    fdiff_sub_z[:, subno] = (data - fdiff_mean)/fdiff_std


np.savez('NONPTE_fmridiff_BOrd.npz',
         fdiff_sub=fdiff_sub,
         fdiff_sub_z=fdiff_sub_z,
         sub_ids=nonepi_ids)





print('done')


