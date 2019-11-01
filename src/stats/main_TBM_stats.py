import os
from multiprocessing import Pool
import numpy as np
from shutil import copyfile, copy
import time
import nilearn.image as ni
from multivariate.hotelling import hotelling_t2
from tqdm import tqdm
import scipy as sp


def check_imgs_exist(studydir, sub_ids):
    subids_imgs = list()

    for id in sub_ids:
        fname = os.path.join(studydir, id, 'BrainSuite',
                             'T1mni.svreg.inv.jacobian.nii.gz')

        if os.path.isfile(fname):

            subids_imgs.append(id)

    return subids_imgs


def readsubs(studydir, sub_ids, nsub=10000):

    print(len(sub_ids))

    sub_ids = check_imgs_exist(studydir, sub_ids)

    print('Reading Subjects')

    for n, id in enumerate(sub_ids):
        if n >= nsub:
            break

        fname = os.path.join(studydir, id, 'BrainSuite',
                             'T1mni.svreg.inv.jacobian.nii.gz')
        print('sub:', n, 'Reading', id)
        jac = ni.load_img(fname)

        if n == 0:
            data = np.zeros((min(len(sub_ids), nsub), ) + jac.shape)

            data[n, :, :, :] = jac.get_data()

    return data, sub_ids


def main():

    studydir = '/big_disk/ajoshi/fitbir/preproc/maryland_rao_v1'

    epi_txt = '/big_disk/ajoshi/fitbir/preproc/maryland_rao_v1_epilepsy_imgs.txt'
    nonepi_txt = '/big_disk/ajoshi/fitbir/preproc/maryland_rao_v1_nonepilepsy_imgs.txt'

    with open(epi_txt) as f:
        epiIds = f.readlines()

    with open(nonepi_txt) as f:
        nonepiIds = f.readlines()

    epiIds = list(map(lambda x: x.strip(), epiIds))
    nonepiIds = list(map(lambda x: x.strip(), nonepiIds))

    epi_data, epi_subids = readsubs(studydir, epiIds, nsub=20)

    nonepi_data, nonepi_subids = readsubs(studydir, nonepiIds, nsub=20)

    epi_data = epi_data.reshape(epi_data.shape[0], -1)
    nonepi_data = nonepi_data.reshape(nonepi_data.shape[0], -1)

    msk = nonepi_data[0, :].flatten() > 0

    numV = msk.sum()

    rval = sp.zeros(numV)
    pval = sp.zeros(numV)

    edat1 = epi_data[:, msk].squeeze().T
    edat2 = nonepi_data[:, msk].squeeze().T

    for nv in tqdm(range(numV)):
        rval[nv], pval[nv] = sp.stats.ranksums(edat1[nv, :], edat2[nv, :])

    np.savez('TBM_results.npz', rval=rval, pval=pval, msk=msk)

    print('done')


if __name__ == "__main__":
    main()
