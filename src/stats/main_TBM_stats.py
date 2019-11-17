import os
from multiprocessing import Pool
import numpy as np
from shutil import copyfile, copy
import time
import nilearn.image as ni
#from multivariate. import TBM_t2
from tqdm import tqdm
import scipy as sp
from statsmodels.stats.multitest import fdrcorrection
from statsmodels.stats.weightstats import ttest_ind
#from statsmodels.stats import wilcoxon

SM = 'smooth3mm'


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
    nsub = min(nsub, len(sub_ids))

    print('Reading Subjects')

    for n, id in enumerate(sub_ids):
        if n >= nsub:
            break

        fname = os.path.join(studydir, id, 'BrainSuite',
                             'T1mni.svreg.inv.jacobian.nii.gz')

        fname_sm = os.path.join(studydir, id, 'BrainSuite',
                                'T1mni.svreg.inv.jacobian.' + SM + '.nii.gz')

        # Smooth the Jacobian image
        if os.path.isfile(fname_sm):
            print('File exists :' + fname_sm)
        else:
            print('Applying Smoothing:' + id)
            os.system(
                '/home/ajoshi/BrainSuite19a/svreg/bin/svreg_smooth_vol_function.sh '
                + fname + ' 3 3 3 ' + fname_sm)

        print('sub:', n, 'Reading', id)
        jac = ni.load_img(fname_sm)

        if n == 0:
            data = np.zeros((min(len(sub_ids), nsub), ) + jac.shape)

        data[n, :, :, :] = jac.get_data()

    return data, sub_ids


def main():

    studydir = '/big_disk/ajoshi/fitbir/preproc/maryland_rao_v1'

    epi_txt = '/big_disk/ajoshi/fitbir/preproc/maryland_rao_v1_epilepsy_imgs.txt'
    nonepi_txt = '/big_disk/ajoshi/fitbir/preproc/maryland_rao_v1_nonepilepsy_imgs.txt'
    atlas = '/home/ajoshi/BrainSuite19a/svreg/BCI-DNI_brain_atlas/BCI-DNI_brain.bfc.nii.gz'

    ati = ni.load_img(atlas)

    with open(epi_txt) as f:
        epiIds = f.readlines()

    with open(nonepi_txt) as f:
        nonepiIds = f.readlines()

    epiIds = list(map(lambda x: x.strip(), epiIds))
    nonepiIds = list(map(lambda x: x.strip(), nonepiIds))

    epi_data, epi_subids = readsubs(studydir, epiIds, nsub=36)

    nonepi_data, nonepi_subids = readsubs(studydir, nonepiIds, nsub=36)

    # Save mean over the epilepsy subjects
    epi_data_mean = ni.new_img_like(ati, epi_data.mean(axis=0))
    epi_data_mean.to_filename('epi_mean.' + SM + '.nii.gz')

    # Save std-dev over the epilepsy subjects
    epi_data_mean = ni.new_img_like(ati, epi_data.std(axis=0))
    epi_data_mean.to_filename('epi_std.' + SM + '.nii.gz')

    # Save mean over the non epilepsy subjects
    nonepi_data_mean = ni.new_img_like(ati, nonepi_data.mean(axis=0))
    nonepi_data_mean.to_filename('nonepi_mean.' + SM + '.nii.gz')

    # Save std-dev over the non epilepsy subjects
    nonepi_data_mean = ni.new_img_like(ati, nonepi_data.std(axis=0))
    nonepi_data_mean.to_filename('nonepi_std.' + SM + '.nii.gz')

    # Save diff of mean over the non epilepsy subjects
    nonepi_data_mean = ni.new_img_like(
        ati,
        epi_data.mean(axis=0) - nonepi_data.mean(axis=0))
    nonepi_data_mean.to_filename('diffepi_mean.' + SM + '.nii.gz')

    # Save diff of std-dev over the non epilepsy subjects
    nonepi_data_mean = ni.new_img_like(
        ati,
        epi_data.std(axis=0) - nonepi_data.std(axis=0))
    nonepi_data_mean.to_filename('diffepi_std.' + SM + '.nii.gz')

    epi_data = epi_data.reshape(epi_data.shape[0], -1)
    nonepi_data = nonepi_data.reshape(nonepi_data.shape[0], -1)

    msk = ati.get_data().flatten() > 0

    numV = msk.sum()

    rval = sp.zeros(numV)
    pval = sp.ones(numV)
    pval_vol = np.ones(ati.shape)

    edat1 = epi_data[:, msk].squeeze().T
    edat2 = nonepi_data[:, msk].squeeze().T

    rval, pval, _ = ttest_ind(edat1.T, edat2.T)
    #    for nv in tqdm(range(numV), mininterval=30, maxinterval=90):
    #        rval[nv], pval[nv] = sp.stats.ranksums(edat1[nv, :], edat2[nv, :])

    np.savez('TBM_results.npz', rval=rval, pval=pval, msk=msk)

    pval_vol = pval_vol.flatten()
    pval_vol[msk] = pval
    pval_vol = pval_vol.reshape(ati.shape)

    p = ni.new_img_like(ati, pval_vol)
    p.to_filename('pval_TBM.' + SM + '.nii.gz')

    pval_vol = 0 * pval_vol.flatten()
    pval_vol[msk] = (pval < 0.05)
    pval_vol = pval_vol.reshape(ati.shape)

    _, pval_fdr = fdrcorrection(pval)
    pval_vol = 0 * pval_vol.flatten()
    pval_vol[msk] = (pval < 0.05)
    pval_vol = pval_vol.reshape(ati.shape)

    p = ni.new_img_like(ati, pval_vol)
    p.to_filename('pval_TBM.sig.mask.' + SM + '.nii.gz')

    # Significance masks
    p1 = ni.smooth_img(p, 5)
    p1.to_filename('pval_TBM_sig_mask.smooth5.' + SM + '.nii.gz')

    p1 = ni.smooth_img(p, 10)
    p1.to_filename('pval_TBM_sig_mask.smooth10.' + SM + '.nii.gz')

    p1 = ni.smooth_img(p, 15)
    p1.to_filename('pval_TBM_sig_mask.smooth15.' + SM + '.nii.gz')

    print('done')


if __name__ == "__main__":
    main()
