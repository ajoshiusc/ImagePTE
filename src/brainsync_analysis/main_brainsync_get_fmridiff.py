from matplotlib.pyplot import axis
import numpy as np
from brainsync import normalizeData, brainSync
import os
from bfp_utils import load_bfp_data
import scipy.io as spio

# get atlas
atlas_data=np.load('grp_atlas.npz')['atlas_data'] #, X2=X2, Os=Os)
atlas_data2, _, _ = normalizeData(atlas_data)
print(np.max(atlas_data2-atlas_data))


studydir = '/ImagePTE1/ajoshi/fitbir/preproc/maryland_rao_v1'
epi_txt = '/ImagePTE1/ajoshi/fitbir/preproc/maryland_rao_v1_epilepsy_imgs.txt'
nonepi_txt = '/ImagePTE1/ajoshi/fitbir/preproc/maryland_rao_v1_nonepilepsy_imgs_37.txt'

atlas_labels = '/ImagePTE1/ajoshi/code_farm/bfp/supp_data/USCLobes_grayordinate_labels.mat'
atlas = spio.loadmat(atlas_labels)

gord_labels = atlas['labels'].squeeze()

label_ids = np.unique(gord_labels)  # unique label ids
#label_ids = [301, 300, 401, 400, 101, 100, 201, 200, 501, 500, 900]
label_ids = [3, 100, 101, 184, 185, 200, 201, 300,
                301, 400, 401, 500, 501, 800, 850, 900]
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

num_vtx = epi_data.shape[1]

# fmri diff for epilepsy 
fdiff_sub = np.zeros((len(label_ids), nsub))

for subno in range(nsub):
    d, _ = brainSync(atlas_data, epi_data[:, :, subno])

    for i, id in enumerate(label_ids):
        idx = gord_labels == id
        data = np.linalg.norm(atlas_data[:, idx] - d[:, idx], axis = 0)
        fdiff_sub[i, subno] = np.mean(data, axis=1)






np.savez('PTE_fmridiff.npz',
            fdiff_sub=fdiff_sub,
            label_ids=label_ids,
            labels=gord_labels,
            cent_mat=cent_mat,
            sub_ids=epi_ids)


np.savez('NONPTE_fmridiff.npz',
            conn_mat=conn_mat,
            label_ids=label_ids,
            labels=gord_labels,
            cent_mat=cent_mat,
            sub_ids=nonepi_ids)

print('done')

