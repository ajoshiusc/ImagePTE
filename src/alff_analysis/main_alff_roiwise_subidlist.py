from matplotlib.pyplot import axis
import numpy as np
import os
import scipy.io as spio


measure = 'fALFF'
atlas_name = 'USCLobes'

#studydir = '/ImagePTE1/ajoshi/maryland_rao_v1_bfp'
studydir = '/ImagePTE1/ajoshi/fitbir/preproc/maryland_rao_v1'
epi_txt = '/ImagePTE1/ajoshi/fitbir/preproc/maryland_rao_v1_epilepsy_imgs.txt'
nonepi_txt = '/ImagePTE1/ajoshi/fitbir/preproc/maryland_rao_v1_nonepilepsy_imgs_37.txt'
nonepi_train_txt = '/ImagePTE1/ajoshi/fitbir/preproc/maryland_rao_v1_nonepilepsy_imgs_training.txt'


atlas_labels = '/home/ajoshi/projects/bfp/supp_data/' + \
    atlas_name+'_grayordinate_labels.mat'
atlas = spio.loadmat(atlas_labels)

gord_labels = atlas['labels'].squeeze()

label_ids = np.unique(gord_labels)  # unique label ids
#label_ids = [301, 300, 401, 400, 101, 100, 201, 200, 501, 500, 900]
# label_ids = [3, 100, 101, 184, 185, 200, 201, 300,
#             301, 400, 401, 500, 501, 800, 850, 900]
# remove WM label from connectivity analysis
label_ids = np.setdiff1d(label_ids, (2000, 0))

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
    fnamefmri = os.path.join(studydir, sub, 'BFP', sub, 'func',  sub + '_rest_bold.' +'32k'+ '.GOrd.mat')
    fname = os.path.join(studydir, sub, 'BFP', sub, 'func',
                         sub + '_rest_bold.' + measure + '.GOrd.mat')
    if os.path.isfile(fname) and os.path.isfile(fnamefmri):
        epi_files.append(fname)
        epi_ids.append(sub)
    else:
        print('File does not exist: %s\n%s' % (fname, fname))

for sub in nonepiIds:
    fname = os.path.join(studydir, sub, 'BFP', sub, 'func',
                         sub + '_rest_bold.' + measure + '.GOrd.mat')
    if os.path.isfile(fname):
        nonepi_files.append(fname)
        nonepi_ids.append(sub)
    else:
        print('File does not exist: %s' % fname)

for sub in nonepitrainIds:
    fname = os.path.join(studydir, sub, 'BFP', sub, 'func',
                         sub + '_rest_bold.' + measure + '.GOrd.mat')
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


# Now read the data


num_gord = 96854

nsub_nonepi_train = len(nonepitrainIds)
nonepitrain_data = np.zeros((num_gord, nsub_nonepi_train))
for i, fname in enumerate(nonepitrain_files):
    nonepitrain_data[:, i] = spio.loadmat(fname)['data'].squeeze()


nsub_epi = len(epiIds)
epi_data = np.zeros((num_gord, nsub_epi))
for i, fname in enumerate(epi_files):
    epi_data[:, i] = spio.loadmat(fname)['data'].squeeze()


nsub_nonepi = len(nonepiIds)
nonepi_data = np.zeros((num_gord, nsub_nonepi))
for i, fname in enumerate(nonepi_files):
    nonepi_data[:, i] = spio.loadmat(fname)['data'].squeeze()


'''nsub = min(epi_data.shape[2], nonepi_data.shape[2])
epi_ids = epi_ids[:nsub]
nonepi_ids = nonepi_ids[:nsub]
#nonepitrain_ids = nonepitrain_ids[:nsub]
'''

num_vtx = epi_data.shape[0]


# calculate vertiexwise mean and variance for trainnon epi data

nsub_nonepi_train = nonepitrain_data.shape[1]
roiwise_data = np.zeros((len(label_ids), nsub_nonepi_train))

for i, id in enumerate(label_ids):
    idx = gord_labels == id
    data = np.mean(nonepitrain_data[idx, :], axis=0)
    roiwise_data[i, :] = data

np.savez('NONPTE_TRAINING_'+measure+'_'+atlas_name+'.npz',
         roiwise_data=roiwise_data, label_ids=label_ids, sub_ids=nonepitrainIds)


nsub_epi = epi_data.shape[1]
roiwise_data = np.zeros((len(label_ids), nsub_epi))

for i, id in enumerate(label_ids):
    idx = gord_labels == id
    data = np.mean(epi_data[idx, :], axis=0)
    roiwise_data[i, :] = data

np.savez('PTE_'+measure+'_'+atlas_name+'.npz',
         roiwise_data=roiwise_data, label_ids=label_ids, sub_ids=epiIds)


nsub_nonepi = nonepi_data.shape[1]
roiwise_data = np.zeros((len(label_ids), nsub_nonepi))

for i, id in enumerate(label_ids):
    idx = gord_labels == id
    data = np.mean(nonepi_data[idx, :], axis=0)
    roiwise_data[i, :] = data

np.savez('NONPTE_'+measure+'_'+atlas_name+'.npz',
         roiwise_data=roiwise_data, label_ids=label_ids, sub_ids=nonepiIds)


print('done')
