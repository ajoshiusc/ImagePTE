import os
import sys
import time
from multiprocessing import Pool
from shutil import copy, copyfile

import nilearn.image as ni
import numpy as np
import scipy as sp
import scipy.stats as ss
from scipy.stats import shapiro
from sklearn.svm import OneClassSVM
from statsmodels.stats.multitest import fdrcorrection
from statsmodels.stats.weightstats import ttest_ind
#from multivariate. import TBM_t2
from tqdm import tqdm
from matplotlib import pyplot
from scipy.stats import ks_2samp

#from statsmodels.stats import wilcoxon


def check_imgs_exist(studydir, sub_ids, sm='smooth3mm.'):
    subids_imgs = list()

    for id in sub_ids:
        fname = os.path.join(studydir, id,
                             'vae_mse.flair.atlas.' + sm + 'nii.gz')

        if not os.path.isfile(fname):
            err_msg = 'the file does not exist: ' + fname
            print(err_msg)
        else:
            subids_imgs.append(id)

    return subids_imgs


def readsubs(studydir, sub_ids, read_mask=False, sm='smooth3mm.'):

    print(len(sub_ids))

    sub_ids = check_imgs_exist(studydir, sub_ids)
    nsub = len(sub_ids)

    print('Reading Subjects')

    for n, id in enumerate(sub_ids):

        if read_mask:
            fname = os.path.join(studydir, id,
                                 'vae_mse.flair.atlas.mask' + '.nii.gz')

        else:
            fname = os.path.join(studydir, id,
                                 'vae_mse.flair.atlas.' + sm + 'nii.gz')

        print('sub:', n, 'Reading', id)
        im = ni.load_img(fname)

        if n == 0:
            data = np.zeros((min(len(sub_ids), nsub), ) + im.shape)

        data[n, :, :, :] = im.get_data()

    return data, sub_ids


def find_lesions_OneclassSVM(studydir, epi_subids, epi_data, nonepi_subids,
                             nonepi_data):

    atlas_bfc = '/ImagePTE1/ajoshi/code_farm/svreg/USCLobes/BCI-DNI_brain.bfc.nii.gz'
    ati = ni.load_img(atlas_bfc)
    atlas_labels = '/ImagePTE1/ajoshi/code_farm/svreg/USCLobes/BCI-DNI_brain.label.nii.gz'
    at_labels = ni.load_img(atlas_labels).get_data()

    epi_data = epi_data.reshape([epi_data.shape[0], -1])
    nonepi_data = nonepi_data.reshape([nonepi_data.shape[0], -1])

    epi_data_lesion = np.zeros(epi_data.shape)
    nonepi_data_lesion = np.zeros(nonepi_data.shape)

    msk = at_labels.flatten() > 0

    edat1 = epi_data[:, msk].squeeze().T
    edat2 = nonepi_data[:, msk].squeeze().T
    X = np.concatenate((edat1, edat2), axis=1)
    Xout = np.zeros(X.shape)

    for j in tqdm(range(X.shape[0])):
        Xout[j, ] = OneClassSVM(gamma=0.001).fit_predict(X[[j], ].T) == -1

    epi_data_lesion[:, msk] = Xout[:, :edat1.shape[1]].T
    nonepi_data_lesion[:, msk] = Xout[:, edat2.shape[1]:].T

    #    epi_data_lesion = epi_data_lesion.reshape(ati.shape)
    #    nonepi_data_lesion = nonepi_data_lesion.reshape(ati.shape)

    for i, id in enumerate(epi_subids):
        fname = os.path.join(studydir, id,
                             'vae_mse.flair.atlas.mask' + '.nii.gz')
        img = ni.new_img_like(ati, epi_data_lesion[i, ].reshape(ati.shape))
        img.to_filename(fname)

        fwdmap = os.path.join(studydir, id, 'BrainSuite',
                              'T1mni.svreg.map.nii.gz')
        flair_nii = os.path.join(studydir, id, 'FLAIRmni' + '.nii.gz')
        fname_lesion_w = os.path.join(studydir, id,
                                      'vae_mse.flair.mask' + '.nii.gz')
        os.system('/home/ajoshi/BrainSuite21a/svreg/bin/svreg_apply_map.sh ' +
                  fwdmap + ' ' + fname + ' ' + fname_lesion_w + ' ' +
                  flair_nii)

        fname = os.path.join(studydir, id, 'vae_mse.flair.atlas' + '.nii.gz')
        flair_nii = os.path.join(studydir, id, 'FLAIRmni' + '.nii.gz')
        fname_lesion_e = os.path.join(studydir, id,
                                      'vae_mse.flair' + '.nii.gz')
        os.system('/home/ajoshi/BrainSuite21a/svreg/bin/svreg_apply_map.sh ' +
                  fwdmap + ' ' + fname + ' ' + fname_lesion_e + ' ' +
                  flair_nii)

    for i, id in enumerate(nonepi_subids):
        fname = os.path.join(studydir, id,
                             'vae_mse.flair.atlas.mask' + '.nii.gz')
        img = ni.new_img_like(ati, nonepi_data_lesion[i, ].reshape(ati.shape))
        img.to_filename(fname)

        fwdmap = os.path.join(studydir, id, 'BrainSuite',
                              'T1mni.svreg.map.nii.gz')

        flair_nii = os.path.join(studydir, id, 'FLAIRmni' + '.nii.gz')
        fname_lesion_w = os.path.join(studydir, id,
                                      'vae_mse.flair.mask' + '.nii.gz')
        os.system('/home/ajoshi/BrainSuite21a/svreg/bin/svreg_apply_map.sh ' +
                  fwdmap + ' ' + fname + ' ' + fname_lesion_w + ' ' +
                  flair_nii)

        fname = os.path.join(studydir, id, 'vae_mse.flair.atlas' + '.nii.gz')
        flair_nii = os.path.join(studydir, id, 'FLAIRmni' + '.nii.gz')
        fname_lesion_e = os.path.join(studydir, id,
                                      'vae_mse.flair' + '.nii.gz')
        os.system('/home/ajoshi/BrainSuite21a/svreg/bin/svreg_apply_map.sh ' +
                  fwdmap + ' ' + fname + ' ' + fname_lesion_e + ' ' +
                  flair_nii)

    return epi_data_lesion, nonepi_data_lesion


def roiwise_stats_KS(epi_data, nonepi_data):

    atlas_bfc = '/ImagePTE1/ajoshi/code_farm/svreg/USCLobes/BCI-DNI_brain.bfc.nii.gz'
    ati = ni.load_img(atlas_bfc)
    atlas_labels = '/ImagePTE1/ajoshi/code_farm/svreg/USCLobes/BCI-DNI_brain.label.nii.gz'
    at_labels = ni.load_img(atlas_labels).get_data()
    vox_size = ni.load_img(atlas_labels).header.get_zooms()
    vox_vol = vox_size[0] * vox_size[1] * vox_size[2]
    #roi_list = [
    #    3, 100, 101, 184, 185, 200, 201, 300, 301, 400, 401, 500, 501, 800,
    #    850, 900, 950
    #] Removed brainstem, cerebellum, cingulate gyrus, whitematter
    roi_list = [301, 300, 401, 400, 101, 100, 201, 200, 501, 500, 900]
    epi_roi_lesion_vols = np.zeros((37, len(roi_list)))
    nonepi_roi_lesion_vols = np.zeros((37, len(roi_list)))

    roi_vols = np.zeros(len(roi_list))

    for i, roi in enumerate(roi_list):
        msk = at_labels == roi
        epi_roi_lesion_vols[:, i] = vox_vol * np.sum(epi_data[:, msk], axis=1)
        nonepi_roi_lesion_vols[:, i] = vox_vol * np.sum(nonepi_data[:, msk],
                                                        axis=1)
        roi_vols[i] = vox_vol * np.sum(at_labels.flatten() == roi)
    ''' For the whole brain comparison
    msk = at_labels > 0
    epi_roi_lesion_vols[:, len(roi_list)] = np.sum(epi_data[:, msk], axis=1)
    nonepi_roi_lesion_vols[:, len(roi_list)] = np.sum(nonepi_data[:, msk], axis=1)
    '''

    print('Doing KS test')

    rval = np.zeros(len(roi_list))
    pval = np.zeros(len(roi_list))

    for i in tqdm(range(len(roi_list))):
        rval[i], pval[i] = ks_2samp(epi_roi_lesion_vols[:,i], nonepi_roi_lesion_vols[:,i])
 
    print('significant rois in f-test are')
    print(roi_list[pval < 0.05])

    _, pval_fdr = fdrcorrection(pval)
    print('significant rois in f-test after FDR correction are')
    print(roi_list[pval_fdr < 0.05])

    w, s = shapiro(epi_roi_lesion_vols)

    print(w, s)

    for i, r in enumerate(list(roi_list)):
        print('%d \t| %0.4g(%0.4g) \t| %.2g(%0.4g) \t|' %
              (r, np.median(epi_roi_lesion_vols[:, i],
                            axis=0), np.std(epi_roi_lesion_vols[:, i], axis=0),
               np.median(nonepi_roi_lesion_vols[:, i], axis=0),
               np.std(nonepi_roi_lesion_vols[:, i], axis=0)))

    for i, r in enumerate(list(roi_list)):
        print(
            '%d \t| %0.4g(%0.4g) \t| %0.4g(%0.4g) \t|' %
            (r, np.median(100 * epi_roi_lesion_vols[:, i] / roi_vols[i],
                          axis=0),
             np.std(100 * epi_roi_lesion_vols[:, i] / roi_vols[i], axis=0),
             np.median(100 * nonepi_roi_lesion_vols[:, i] / roi_vols[i],
                       axis=0),
             np.std(100 * nonepi_roi_lesion_vols[:, i] / roi_vols[i], axis=0)))

    for i, r in enumerate(list(roi_list)):
        print('%d \t| %.4g' % (r, pval_fdr[i]))

    return


def pointwise_stats_KS(epi_data, nonepi_data):

    sm='0mm'
    atlas = '/home/ajoshi/BrainSuite21a/svreg/BCI-DNI_brain_atlas/BCI-DNI_brain.bfc.nii.gz'
    ati = ni.load_img(atlas)

    # Save mean over the epilepsy subjects
    epi = ni.new_img_like(ati, epi_data.mean(axis=0))
    epi.to_filename('epi_mean_lesion' + sm + '.nii.gz')

    # Save std-dev over the epilepsy subjects
    epi = ni.new_img_like(ati, epi_data.std(axis=0))
    epi.to_filename('epi_std_lesion' + sm + '.nii.gz')

    # Save mean over the non epilepsy subjects
    nonepi = ni.new_img_like(ati, nonepi_data.mean(axis=0))
    nonepi.to_filename('nonepi_mean_lesion' + sm + '.nii.gz')

    # Save std-dev over the non epilepsy subjects
    nonepi = ni.new_img_like(ati, nonepi_data.std(axis=0))
    nonepi.to_filename('nonepi_std_lesion' + sm + '.nii.gz')

    # Save diff of mean over the non epilepsy subjects
    nonepi = ni.new_img_like(ati,
                             epi_data.mean(axis=0) - nonepi_data.mean(axis=0))
    nonepi.to_filename('diffepi_mean_lesion' + sm + '.nii.gz')

    # Save diff of std-dev over the non epilepsy subjects
    nonepi = ni.new_img_like(ati,
                             epi_data.std(axis=0) - nonepi_data.std(axis=0))
    nonepi.to_filename('diffepi_std_lesion' + sm + '.nii.gz')

    epi_data = epi_data.reshape(epi_data.shape[0], -1)
    nonepi_data = nonepi_data.reshape(nonepi_data.shape[0], -1)

    msk = ati.get_data().flatten() > 0
    pval_vol = np.ones(ati.shape)

    #   rval_vol = sp.zeros(numV)
    #   pval_vol = sp.ones(numV)

    edat1 = epi_data[:, msk].squeeze().T
    edat2 = nonepi_data[:, msk].squeeze().T

    num_vox = np.sum(msk)

    rval = np.zeros(num_vox)
    pval = np.zeros(num_vox)

    print('Doing KS test')
    for i in tqdm(range(num_vox)):
        rval[i], pval[i] = ks_2samp(edat1[i,:], edat2[i,:])
    #    for nv in tqdm(range(numV), mininterval=30, maxinterval=90):
    #        rval[nv], pval[nv] = sp.stats.ranksums(edat1[nv, :], edat2[nv, :])

    np.savez('lesion_results_KS.npz', rval=rval, pval=pval, msk=msk)

    pval_vol = pval_vol.flatten()
    pval_vol[msk] = pval
    pval_vol = pval_vol.reshape(ati.shape)

    p = ni.new_img_like(ati, pval_vol)
    p.to_filename('pval_KS_lesion' + sm + '.nii.gz')
    '''pval_vol = 0 * pval_vol.flatten()
    pval_vol[msk] = (pval < 0.05)
    pval_vol = pval_vol.reshape(ati.shape)

    p = ni.new_img_like(ati, pval_vol)
    p.to_filename('pval_lesion.sig.mask' + sm + '.nii.gz')
    '''

    # FDR corrected p values
    _, pval_fdr = fdrcorrection(pvals=pval)

    pval_vol = pval_vol.flatten()
    pval_vol[msk] = pval_fdr
    pval_vol = pval_vol.reshape(ati.shape)

    p = ni.new_img_like(ati, pval_vol)
    p.to_filename('pval_KS_fdr_lesion' + sm + '.nii.gz')

    pval_vol = 0 * pval_vol.flatten()
    pval_vol[msk] = (pval_fdr < 0.05)
    pval_vol = pval_vol.reshape(ati.shape)

    p = ni.new_img_like(ati, pval_vol)
    p.to_filename('pval_KS_fdr_lesion.sig.mask' + sm + '.nii.gz')
    ''' Significance masks
    p1 = ni.smooth_img(p, 5)
    p1.to_filename('pval_lesion_sig_mask.smooth5' + sm + '.nii.gz')

    p1 = ni.smooth_img(p, 10)
    p1.to_filename('pval_lesion_sig_mask.smooth10' + sm + '.nii.gz')

    p1 = ni.smooth_img(p, 15)
    p1.to_filename('pval_lesion_sig_mask.smooth15' + sm + '.nii.gz')
    '''

    #edat1.img


def main():

    studydir = '/ImagePTE1/ajoshi/fitbir/preproc/maryland_rao_v1'

    epi_txt = '/ImagePTE1/ajoshi/fitbir/preproc/maryland_rao_v1_epilepsy_imgs.txt'
    nonepi_txt = '/ImagePTE1/ajoshi/fitbir/preproc/maryland_rao_v1_nonepilepsy_imgs.txt'

    with open(epi_txt) as f:
        epiIds = f.readlines()

    with open(nonepi_txt) as f:
        nonepiIds = f.readlines()

    epiIds = list(map(lambda x: x.strip(), epiIds))
    nonepiIds = list(map(lambda x: x.strip(), nonepiIds))
    

    # Use once class SVM to compute lesion volume
    #roiwise_stats_OneclassSVM(epi_data, nonepi_data)

    epi_data, epi_subids = readsubs(studydir, epiIds, read_mask=False)
    nonepi_data, nonepi_subids = readsubs(studydir, nonepiIds, read_mask=False)


    # read tiny mask

    msk = ni.load_img('stats/right_temporal_tiny.mask.nii.gz')

    msk_ind = msk.get_fdata()>0

    epi = np.mean(epi_data[:,msk_ind], axis=1)
    nonepi = np.mean(nonepi_data[:,msk_ind], axis=1)

    bins = np.linspace(0,np.max(epi),100)

    pyplot.hist(epi, bins, alpha=0.5, label='pte')
    pyplot.hist(nonepi, bins, alpha=0.5, label='nonpte')
    pyplot.legend(loc='upper right')
    #pyplot.show()
    pyplot.savefig('hist4tinyroi_right_temporal.png')
    pyplot.close()

    '''find_lesions_OneclassSVM(studydir, epi_subids, epi_data, nonepi_subids,
                             nonepi_data)
                             '''

         # Do Pointwise stats
    #pointwise_stats_KS(epi_data, nonepi_data)

    # Do ROIwise stats
    roiwise_stats_KS(epi_data, nonepi_data)

    print('done')


if __name__ == "__main__":
    main()
