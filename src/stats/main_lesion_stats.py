import os
import sys
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

NSUB = 37 * 2
ATLAS = '/home/ajoshi/BrainSuite19a/svreg/BCI-DNI_brain_atlas/BCI-DNI_brain.bfc.nii.gz'


def warpsubs(studydir, lesion_studydir, sub_ids, nonepi):

    print(len(sub_ids))

    check_imgs_exist(studydir, sub_ids)

    nsub = 37

    print('Reading Subjects')

    for n, id in enumerate(sub_ids):

        invmap = os.path.join(studydir, id, 'BrainSuite',
                              'T1mni.svreg.inv.map.nii.gz')

        # Warp Lesion Map
        fname_lesion = os.path.join(
            lesion_studydir, id, 'MSE_T1_' + str(n + 37 * nonepi) + '.nii.gz')

        fname_lesion_w = os.path.join(studydir, id, 'lesion_vae.atlas.nii.gz')
        os.system('/home/ajoshi/BrainSuite19a/svreg/bin/svreg_apply_map.sh ' +
                  invmap + ' ' + fname_lesion + ' ' + fname_lesion_w + ' ' + ATLAS)


def check_imgs_exist(studydir, sub_ids):

    for id in sub_ids:
        fname = os.path.join(studydir, id, 'BrainSuite',
                             'T1mni.svreg.inv.jacobian.nii.gz')

        if not os.path.isfile(fname):
            err_msg = 'the inv map does not exist for: ' + id, ' in dir: ' + studydir
            sys.exit(err_msg)


def readsubs(studydir):

    print('Reading Subjects')

    for isub in range(NSUB):
        fname = os.path.join(studydir, 'MSE_FLAIR_' + str(isub) + '.nii.gz')

        print(isub, end=',')
        err = ni.load_img(fname)

        if isub == 0:
            data = np.zeros((NSUB, ) + err.shape)

        data[isub, :, :, :] = err.get_data()

    return data


def main():

    studydir = '/big_disk/akrami/git_repos_new/ImagePTE/src/Lesion Detection/models/3D_out'
    studydir_imgs = '/big_disk/ajoshi/fitbir/preproc/maryland_rao_v1'

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

    # Warp maps to match the atlas
    warpsubs(studydir=studydir_imgs,
             lesion_studydir=studydir,
             sub_ids=epiIds,
             nonepi=0)

    warpsubs(studydir=studydir_imgs,
             lesion_studydir=studydir,
             sub_ids=nonepiIds,
             nonepi=1)

    data = readsubs(studydir)

    epi_data = data[:int(NSUB / 2), :, :, :]
    nonepi_data = data[int(NSUB / 2):, :, :, :]
    #    readsubs(studydir, nonepiIds, NSUB=36)

    # Save mean over the epilepsy subjects
    epi_data_mean = ni.new_img_like(ati, epi_data.mean(axis=0))
    epi_data_mean.to_filename('epi_mean_err.nii.gz')

    # Save std-dev over the epilepsy subjects
    epi_data_mean = ni.new_img_like(ati, epi_data.std(axis=0))
    epi_data_mean.to_filename('epi_std_err.nii.gz')

    # Save mean over the non epilepsy subjects
    nonepi_data_mean = ni.new_img_like(ati, nonepi_data.mean(axis=0))
    nonepi_data_mean.to_filename('nonepi_mean_err.nii.gz')

    # Save std-dev over the non epilepsy subjects
    nonepi_data_mean = ni.new_img_like(ati, nonepi_data.std(axis=0))
    nonepi_data_mean.to_filename('nonepi_std_err.nii.gz')

    # Save diff of mean over the non epilepsy subjects
    nonepi_data_mean = ni.new_img_like(
        ati,
        epi_data.mean(axis=0) - nonepi_data.mean(axis=0))
    nonepi_data_mean.to_filename('diffepi_mean_err.nii.gz')

    # Save diff of std-dev over the non epilepsy subjects
    nonepi_data_mean = ni.new_img_like(
        ati,
        epi_data.std(axis=0) - nonepi_data.std(axis=0))
    nonepi_data_mean.to_filename('diffepi_std_err.nii.gz')

    epi_data = epi_data.reshape(epi_data.shape[0], -1)
    nonepi_data = nonepi_data.reshape(nonepi_data.shape[0], -1)

    msk = ati.get_data().flatten() > 0
    #******warp to atlas*********
    numV = msk.sum()

    rval = sp.zeros(numV)
    pval = sp.ones(numV)
    pval_vol = np.ones(ati.shape)

    edat1 = epi_data[:, msk].squeeze().T
    edat2 = nonepi_data[:, msk].squeeze().T

    rval, pval, _ = ttest_ind(edat1.T, edat2.T)
    #    for nv in tqdm(range(numV), mininterval=30, maxinterval=90):
    #        rval[nv], pval[nv] = sp.stats.ranksums(edat1[nv, :], edat2[nv, :])

    np.savez('vae_flair_error_results.npz', rval=rval, pval=pval, msk=msk)

    pval_vol = pval_vol.flatten()
    pval_vol[msk] = pval
    pval_vol = pval_vol.reshape(ati.shape)

    p = ni.new_img_like(ati, pval_vol)
    p.to_filename('pval_vae.nii.gz')

    pval_vol = 0 * pval_vol.flatten()
    pval_vol[msk] = (pval < 0.05)
    pval_vol = pval_vol.reshape(ati.shape)

    p = ni.new_img_like(ati, pval_vol)
    p.to_filename('pval_vae.sig.mask.nii.gz')

    # Significance masks
    p1 = ni.smooth_img(p, 5)
    p1.to_filename('pval_vae_sig_mask.smooth5.nii.gz')

    p1 = ni.smooth_img(p, 10)
    p1.to_filename('pval_vae_sig_mask.smooth10.nii.gz')

    p1 = ni.smooth_img(p, 15)
    p1.to_filename('pval_vae_sig_mask.smooth15.nii.gz')

    print('done')


if __name__ == "__main__":
    main()
