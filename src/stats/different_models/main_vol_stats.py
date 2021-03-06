import os
from multiprocessing import Pool
import numpy as np
from shutil import copyfile, copy
import time
import nilearn.image as ni
from multivariate.hotelling import hotelling_t2
from tqdm import tqdm
import scipy as sp
from statsmodels.stats.multitest import fdrcorrection

ATLAS = '/home/ajoshi/BrainSuite19a/svreg/BCI-DNI_brain_atlas/BCI-DNI_brain.bfc.nii.gz'
SM = 'smooth3mm'


def check_imgs_exist(studydir, sub_ids):
    subids_imgs = list()

    for id in sub_ids:
        fname_T1 = os.path.join(studydir, id, 'T1mni.nii.gz')
        fname_T2 = os.path.join(studydir, id, 'T2mni.nii.gz')
        fname_FLAIR = os.path.join(studydir, id, 'FLAIRmni.nii.gz')
        invmap = os.path.join(studydir, id, 'BrainSuite',
                              'T1mni.svreg.inv.map.nii.gz')

        if os.path.isfile(fname_T1) and os.path.isfile(
                fname_T2) and os.path.isfile(fname_FLAIR) and os.path.isfile(
                    invmap):

            subids_imgs.append(id)

    return subids_imgs


def warpsubs(studydir, sub_ids, nsub=10000):

    print(len(sub_ids))

    sub_ids = check_imgs_exist(studydir, sub_ids)

    nsub = min(nsub, len(sub_ids))

    print('Reading Subjects')

    for n, id in enumerate(sub_ids):

        print('Processing subject %d / %d' % (n + 1, len(sub_ids)))

        invmap = os.path.join(studydir, id, 'BrainSuite',
                              'T1mni.svreg.inv.map.nii.gz')

        # Warp T1 image
        fname_T1 = os.path.join(studydir, id, 'T1mni.nii.gz')
        fname_T1_w = os.path.join(studydir, id, 'T1mni.atlas.nii.gz')

        if not os.path.isfile(fname_T1_w):
            os.system(
                '/home/ajoshi/BrainSuite19a/svreg/bin/svreg_apply_map.sh ' +
                invmap + ' ' + fname_T1 + ' ' + fname_T1_w + ' ' + ATLAS)

        fname_T2 = os.path.join(studydir, id, 'T2mni.nii.gz')
        fname_T2_w = os.path.join(studydir, id, 'T2mni.atlas.nii.gz')

        if not os.path.isfile(fname_T2_w):
            os.system(
                '/home/ajoshi/BrainSuite19a/svreg/bin/svreg_apply_map.sh ' +
                invmap + ' ' + fname_T2 + ' ' + fname_T2_w + ' ' + ATLAS)

        fname_FLAIR = os.path.join(studydir, id, 'FLAIRmni.nii.gz')
        fname_FLAIR_w = os.path.join(studydir, id, 'FLAIRmni.atlas.nii.gz')

        if not os.path.isfile(fname_FLAIR_w):
            os.system(
                '/home/ajoshi/BrainSuite19a/svreg/bin/svreg_apply_map.sh ' +
                invmap + ' ' + fname_FLAIR + ' ' + fname_FLAIR_w + ' ' + ATLAS)

        # Smooth the warped images

        fname_T1_sm = os.path.join(studydir, id,
                                   'T1mni.atlas.' + SM + '.nii.gz')
        fname_T2_sm = os.path.join(studydir, id,
                                   'T2mni.atlas.' + SM + '.nii.gz')
        fname_FLAIR_sm = os.path.join(studydir, id,
                                      'FLAIRmni.atlas.' + SM + '.nii.gz')

        # Smooth the images
        if os.path.isfile(fname_T1_sm) and os.path.isfile(
                fname_T2_sm) and os.path.isfile(fname_FLAIR_sm):
            print('File exists :' + fname_T1_sm)
            print('File exists :' + fname_T2_sm)
            print('File exists :' + fname_FLAIR_sm)
        else:
            print('Applying smoothing for:' + id)
            os.system(
                '/home/ajoshi/BrainSuite19a/svreg/bin/svreg_smooth_vol_function.sh '
                + fname_T1_w + ' 3 3 3 ' + fname_T1_sm)

            os.system(
                '/home/ajoshi/BrainSuite19a/svreg/bin/svreg_smooth_vol_function.sh '
                + fname_T2_w + ' 3 3 3 ' + fname_T2_sm)
            os.system(
                '/home/ajoshi/BrainSuite19a/svreg/bin/svreg_smooth_vol_function.sh '
                + fname_FLAIR_w + ' 3 3 3 ' + fname_FLAIR_sm)


def readsubs(studydir, sub_ids, nsub=10000):

    print(len(sub_ids))

    sub_ids = check_imgs_exist(studydir, sub_ids)

    nsub = min(nsub, len(sub_ids))

    print('Reading Subjects')

    for n, id in enumerate(sub_ids):

        fname_T1_sm = os.path.join(studydir, id,
                                   'T1mni.atlas.' + SM + '.nii.gz')
        fname_T2_sm = os.path.join(studydir, id,
                                   'T2mni.atlas.' + SM + '.nii.gz')
        fname_FLAIR_sm = os.path.join(studydir, id,
                                      'FLAIRmni.atlas.' + SM + '.nii.gz')

        print('sub:', n, 'Reading', id)
        t1 = ni.load_img(fname_T1_sm)
        t2 = ni.load_img(fname_T2_sm)
        flair = ni.load_img(fname_FLAIR_sm)

        if n == 0:
            data = np.zeros((3, min(len(sub_ids), nsub)) + t1.shape)

        data[0, n, :, :, :] = t1.get_data()
        data[1, n, :, :, :] = t2.get_data()
        data[2, n, :, :, :] = flair.get_data()

    return data, sub_ids, flair


def main():

    studydir = '/ImagePTE1/ajoshi/fitbir/preproc/maryland_rao_v1'

    epi_txt = '/ImagePTE1/ajoshi/fitbir/preproc/maryland_rao_v1_epilepsy_imgs.txt'
    nonepi_txt = '/ImagePTE1/ajoshi/fitbir/preproc/maryland_rao_v1_nonepilepsy_imgs_37.txt'

    atlas = '/home/ajoshi/BrainSuite19b/svreg/BCI-DNI_brain_atlas/BCI-DNI_brain.bfc.nii.gz'

    ati = ni.load_img(atlas)

    with open(epi_txt) as f:
        epiIds = f.readlines()

    with open(nonepi_txt) as f:
        nonepiIds = f.readlines()

    epiIds = list(map(lambda x: x.strip(), epiIds))
    nonepiIds = list(map(lambda x: x.strip(), nonepiIds))

    warpsubs(studydir, epiIds, nsub=37)
    epi_data, epi_subids, t1 = readsubs(studydir, epiIds, nsub=37)

    warpsubs(studydir, nonepiIds, nsub=37)
    nonepi_data, nonepi_subids, _ = readsubs(studydir, nonepiIds, nsub=37)

    epi_data = epi_data.reshape(epi_data.shape[0], epi_data.shape[1], -1)
    nonepi_data = nonepi_data.reshape(nonepi_data.shape[0],
                                      nonepi_data.shape[1], -1)

    msk = ati.get_data().flatten() > 0

    # Epilepsy T1, T2, FLAIR average and std-dev data

    t1_avg_vol = np.zeros(msk.shape[0])
    t2_avg_vol = np.zeros(epi_data.shape[2])
    flair_avg_vol = np.zeros(epi_data.shape[2])
    t1_std_vol = np.zeros(epi_data.shape[2])
    t2_std_vol = np.zeros(epi_data.shape[2])
    flair_std_vol = np.zeros(epi_data.shape[2])

    t1_avg_vol[msk] = np.mean(epi_data[0, :, msk], axis=1)
    t2_avg_vol[msk] = np.mean(epi_data[1, :, msk], axis=1)
    flair_avg_vol[msk] = np.mean(epi_data[2, :, msk], axis=1)
    t1_std_vol[msk] = np.std(epi_data[0, :, msk], axis=1)
    t2_std_vol[msk] = np.std(epi_data[1, :, msk], axis=1)
    flair_std_vol[msk] = np.std(epi_data[2, :, msk], axis=1)

    t1_avg = ni.new_img_like(ati, t1_avg_vol.reshape(ati.shape))
    t1_avg.to_filename('t1_epi_avg.' + SM + '.nii.gz')
    t1_std = ni.new_img_like(ati, t1_std_vol.reshape(ati.shape))
    t1_std.to_filename('t1_epi_std.' + SM + '.nii.gz')

    t2_avg = ni.new_img_like(ati, t2_avg_vol.reshape(ati.shape))
    t2_avg.to_filename('t2_epi_avg.' + SM + '.nii.gz')
    t2_std = ni.new_img_like(ati, t2_std_vol.reshape(ati.shape))
    t2_std.to_filename('t2_epi_std.' + SM + '.nii.gz')

    flair_avg = ni.new_img_like(ati, flair_avg_vol.reshape(ati.shape))
    flair_avg.to_filename('flair_epi_avg.' + SM + '.nii.gz')
    flair_std = ni.new_img_like(ati, flair_std_vol.reshape(ati.shape))
    flair_std.to_filename('flair_epi_std.' + SM + '.nii.gz')

    # Non Epilepsy T1, T2, FLAIR average data

    t1_avg_vol = np.zeros(nonepi_data.shape[2])
    t2_avg_vol = np.zeros(nonepi_data.shape[2])
    flair_avg_vol = np.zeros(nonepi_data.shape[2])
    t1_std_vol = np.zeros(nonepi_data.shape[2])
    t2_std_vol = np.zeros(nonepi_data.shape[2])
    flair_std_vol = np.zeros(nonepi_data.shape[2])

    t1_avg_vol[msk] = np.mean(nonepi_data[0, :, msk], axis=1)
    t2_avg_vol[msk] = np.mean(nonepi_data[1, :, msk], axis=1)
    flair_avg_vol[msk] = np.mean(nonepi_data[2, :, msk], axis=1)
    t1_std_vol[msk] = np.std(nonepi_data[0, :, msk], axis=1)
    t2_std_vol[msk] = np.std(nonepi_data[1, :, msk], axis=1)
    flair_std_vol[msk] = np.std(nonepi_data[2, :, msk], axis=1)

    t1_avg = ni.new_img_like(ati, t1_avg_vol.reshape(ati.shape))
    t1_avg.to_filename('t1_nonepi_avg.' + SM + '.nii.gz')
    t1_std = ni.new_img_like(ati, t1_std_vol.reshape(ati.shape))
    t1_std.to_filename('t1_nonepi_std.' + SM + '.nii.gz')

    t2_avg = ni.new_img_like(ati, t2_avg_vol.reshape(ati.shape))
    t2_avg.to_filename('t2_nonepi_avg.' + SM + '.nii.gz')
    t2_std = ni.new_img_like(ati, t2_std_vol.reshape(ati.shape))
    t2_std.to_filename('t2_nonepi_std.' + SM + '.nii.gz')

    flair_avg = ni.new_img_like(ati, flair_avg_vol.reshape(ati.shape))
    flair_avg.to_filename('flair_nonepi_avg.' + SM + '.nii.gz')
    flair_std = ni.new_img_like(ati, flair_std_vol.reshape(ati.shape))
    flair_std.to_filename('flair_nonepi_std.' + SM + '.nii.gz')

    numV = msk.sum()
    pval_vol = np.ones(ati.shape)

    #    rval = sp.zeros(numV)
    #    pval = sp.zeros(numV)

    #    edat1 = epi_data[0, :, msk].squeeze()
    #    edat2 = nonepi_data[0, :, msk].squeeze()

    #    edat1_mean = edat1.mean(axis=0)
    #    ni.new_img_like(ati, edat1_mean)

    #    for nv in tqdm(range(numV)):
    #        rval[nv], pval[nv] = sp.stats.ranksums(edat1[nv, :], edat2[nv, :])

    pval, t2 = hotelling_t2(epi_data[:, :, msk], nonepi_data[:, :, msk])

    np.savez('Hotelling_results.npz', t2=t2, pval=pval, msk=msk)

    pval_vol = pval_vol.flatten()
    pval_vol[msk] = pval
    pval_vol = pval_vol.reshape(ati.shape)

    p = ni.new_img_like(ati, pval_vol)
    p.to_filename('pval_hotelling.' + SM + '.nii.gz')

    pval_vol = 0 * pval_vol.flatten()
    pval_vol[msk] = (pval < 0.05)
    pval_vol = pval_vol.reshape(ati.shape)

    p = ni.new_img_like(ati, pval_vol)
    p.to_filename('pval_hotelling.sig.mask.' + SM + '.nii.gz')

    # Significance masks
    p1 = ni.smooth_img(p, 5)
    p1.to_filename('pval_hotelling_sig_mask.smooth5.' + SM + '.nii.gz')

    p1 = ni.smooth_img(p, 10)
    p1.to_filename('pval_hotelling_sig_mask.smooth10.' + SM + '.nii.gz')

    p1 = ni.smooth_img(p, 15)
    p1.to_filename('pval_hotelling_sig_mask.smooth15.' + SM + '.nii.gz')

    # Do FDR correction

    _, pval_fdr = fdrcorrection(pval)
    pval_vol = 0 * pval_vol.flatten()
    pval_vol[msk] = (pval_fdr)
    pval_vol = pval_vol.reshape(ati.shape)
    p = ni.new_img_like(ati, pval_vol)
    p.to_filename('pval_fdr_hotelling.' + SM + '.nii.gz')

    pval_vol = 0 * pval_vol.flatten()
    pval_vol[msk] = (pval_fdr < 0.05)
    pval_vol = pval_vol.reshape(ati.shape)

    p = ni.new_img_like(ati, pval_vol)
    p.to_filename('pval_fdr_hotelling.sig.mask.' + SM + '.nii.gz')

    # Significance masks
    p1 = ni.smooth_img(p, 5)
    p1.to_filename('pval_fdr_hotelling_sig_mask.smooth5.' + SM + '.nii.gz')

    p1 = ni.smooth_img(p, 10)
    p1.to_filename('pval_fdr_hotelling_sig_mask.smooth10.' + SM + '.nii.gz')

    p1 = ni.smooth_img(p, 15)
    p1.to_filename('pval_fdr_hotelling_sig_mask.smooth15.' + SM + '.nii.gz')

    print('done')


if __name__ == "__main__":
    main()
