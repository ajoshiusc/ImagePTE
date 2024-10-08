import os
from multiprocessing import Pool
import numpy as np
from shutil import copyfile, copy
import time
import nilearn.image as ni
# from multivariate. import TBM_t2
from tqdm import tqdm
import scipy as sp
from statsmodels.stats.multitest import fdrcorrection
from statsmodels.stats.weightstats import ttest_ind
import scipy.stats as ss
from scipy.stats import shapiro
#from statsmodels.stats import wilcoxon
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
#from sklearn.metrics import plot_roc_curve, roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
import sys
from sklearn.preprocessing import normalize
from sklearn.model_selection import cross_val_score, LeaveOneOut
from dfsio import readdfs, writedfs
from surfproc import patch_color_attrib

sm = '.smooth3mm'


def f_importances_atlas(coef, roi_ids, atlasbasename, outbase):

    v = ni.load_img(atlasbasename + '.label.nii.gz')
    vlab = v.get_fdata()
    left = readdfs(atlasbasename + '.left.mid.cortex.dfs')
    right = readdfs(atlasbasename + '.right.mid.cortex.dfs')
    left.labels = np.mod(left.labels, 1000)
    right.labels = np.mod(right.labels, 1000)
    vlab = np.mod(vlab, 1000)

    left.attributes = np.zeros(left.vertices.shape[0])
    right.attributes = np.zeros(right.vertices.shape[0])

    vimp = np.zeros(vlab.shape)
    for i, r in enumerate(roi_ids):
        vimp[vlab == r] = coef[i]
        left.attributes[left.labels == r] = coef[i]
        right.attributes[right.labels == r] = coef[i]

    
    vi = ni.new_img_like(v, np.float32(vimp))
    vi.to_filename(outbase+'feat_brainnetome.imp.nii.gz')

    patch_color_attrib(left)

    writedfs(outbase+'.left.brainnetome.imp.dfs', left)
    writedfs(outbase+'.right.brainnetome.imp.dfs', right)


def f_importances(coef, names, outbase):
    imp = coef
    imp, names = zip(*sorted(zip(imp, names)))
    plt.barh(range(len(names)), imp, align='center')
    plt.yticks(range(len(names)), names)
    # plt.show()
    plt.savefig(outbase+'feat_imp.png')


def check_imgs_exist(studydir, sub_ids):
    subids_imgs = list()

    for id in sub_ids:
        fname = os.path.join(studydir, id, 'vae_mse.flair.atlas' + '.nii.gz')

        if not os.path.isfile(fname):
            err_msg = 'the file does not exist: ' + fname
            print(err_msg)
        else:
            subids_imgs.append(id)

    return subids_imgs


def readsubs(studydir, sub_ids):

    print(len(sub_ids))

    sub_ids = check_imgs_exist(studydir, sub_ids)
    nsub = len(sub_ids)

    print('Reading Subjects')

    for n, id in enumerate(sub_ids):
        # vae_mse.flair.mask
        # vae_mse.flair.atlas.mask
        fname = os.path.join(studydir, id, 'vae_mse.flair.atlas' + '.nii.gz')
        print('sub:', n, 'Reading', id)
        im = ni.load_img(fname)

        if n == 0:
            data = np.zeros((min(len(sub_ids), nsub), ) + im.shape)

        data[n, :, :, :] = np.asanyarray(im.dataobj)

    return data, sub_ids


def roiwise_stats(epi_data, nonepi_data):

    atlas_labels = '/home/ajoshi/Software/BrainSuite23a/svreg//USCBrainMulti/Brainnetome/BCI-Brainnetome.label.nii.gz'
    at_labels = np.asanyarray(ni.load_img(atlas_labels).dataobj)
    # roi_list = [
    #    3, 100, 101, 184, 185, 200, 201, 300, 301, 400, 401, 500, 501, 800,
    #    850, 900, 950
    # ]
    #roi_list = [301, 300, 401, 400, 101, 100, 201, 200, 501, 500, 900]
    #roi_list = [300,301]
    roi_list = np.unique(at_labels.flatten())

    epi_roi_lesion_vols = np.zeros((37, len(roi_list)))
    nonepi_roi_lesion_vols = np.zeros((37, len(roi_list)))

    for i, roi in enumerate(roi_list):
        msk = at_labels == roi
        epi_roi_lesion_vols[:, i] = np.sum(epi_data[:, msk], axis=1)
        nonepi_roi_lesion_vols[:, i] = np.sum(nonepi_data[:, msk], axis=1)
    ''' For the whole brain comparison
    msk = at_labels > 0
    epi_roi_lesion_vols[:, len(roi_list)] = np.sum(epi_data[:, msk], axis=1)
    nonepi_roi_lesion_vols[:, len(roi_list)] = np.sum(nonepi_data[:, msk], axis=1)
    '''

    t, p, _ = ttest_ind(epi_roi_lesion_vols, nonepi_roi_lesion_vols)

    F = epi_roi_lesion_vols.var(axis=0) / (nonepi_roi_lesion_vols.var(axis=0) +
                                           1e-6)
    pval = 1 - ss.f.cdf(F, 37 - 1, 37 - 1)

    roi_list = np.array(roi_list)

    print('significant rois in t-test are')
    print(roi_list[p < 0.05])

    print('significant rois in f-test are')
    print(roi_list[pval < 0.05])

    _, pval_fdr = fdrcorrection(pval)
    print('significant rois in f-test after FDR correction are')
    print(roi_list[pval_fdr < 0.05])

    w, s = shapiro(epi_roi_lesion_vols)

    print(w, s)
    return epi_roi_lesion_vols, nonepi_roi_lesion_vols


def pointwise_stats(epi_data, nonepi_data):

    atlas = '/home/ajoshi/BrainSuite19a/svreg/BCI-DNI_brain_atlas/BCI-DNI_brain.bfc.nii.gz'
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

    msk = np.asanyarray(ati.dataobj).flatten() > 0
    pval_vol = np.ones(ati.shape)

    #   rval_vol = sp.zeros(numV)
    #   pval_vol = sp.ones(numV)

    edat1 = epi_data[:, msk].squeeze().T
    edat2 = nonepi_data[:, msk].squeeze().T

    rval, pval, _ = ttest_ind(edat1.T, edat2.T)
    #    for nv in tqdm(range(numV), mininterval=30, maxinterval=90):
    #        rval[nv], pval[nv] = sp.stats.ranksums(edat1[nv, :], edat2[nv, :])

    np.savez('lesion_results.npz', rval=rval, pval=pval, msk=msk)

    pval_vol = pval_vol.flatten()
    pval_vol[msk] = pval
    pval_vol = pval_vol.reshape(ati.shape)

    p = ni.new_img_like(ati, pval_vol)
    p.to_filename('pval_lesion' + sm + '.nii.gz')
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
    p.to_filename('pval_fdr_lesion' + sm + '.nii.gz')

    pval_vol = 0 * pval_vol.flatten()
    pval_vol[msk] = (pval_fdr < 0.05)
    pval_vol = pval_vol.reshape(ati.shape)

    p = ni.new_img_like(ati, pval_vol)
    p.to_filename('pval_fdr_lesion.sig.mask' + sm + '.nii.gz')
    ''' Significance masks
    p1 = ni.smooth_img(p, 5)
    p1.to_filename('pval_lesion_sig_mask.smooth5' + sm + '.nii.gz')

    p1 = ni.smooth_img(p, 10)
    p1.to_filename('pval_lesion_sig_mask.smooth10' + sm + '.nii.gz')

    p1 = ni.smooth_img(p, 15)
    p1.to_filename('pval_lesion_sig_mask.smooth15' + sm + '.nii.gz')
    '''

    # Do f test
    F = epi_data.var(axis=0) / (nonepi_data.var(axis=0) + 1e-16)

    F = F * msk
    fimg = ni.new_img_like(ati, F.reshape(ati.shape))
    fimg.to_filename('fval_lesion' + sm + '.nii.gz')

    pval = 1 - ss.f.cdf(F, 37 - 1, 37 - 1)
    fimg = ni.new_img_like(ati, pval.reshape(ati.shape))
    fimg.to_filename('pval_ftest_lesion' + sm + '.nii.gz')

    _, pval_fdr = fdrcorrection(pvals=pval)
    fimg = ni.new_img_like(ati, pval_fdr.reshape(ati.shape))
    fimg.to_filename('pval_fdr_ftest_lesion' + sm + '.nii.gz')


def main():

    studydir = '/ImagePTE1/ajoshi/fitbir/preproc/maryland_rao_v1'

    epi_txt = '/ImagePTE1/ajoshi/fitbir/preproc/maryland_rao_v1_epilepsy_imgs.txt'
    nonepi_txt = '/ImagePTE1/ajoshi/fitbir/preproc/maryland_rao_v1_nonepilepsy_imgs_37.txt'
    outbase = 'brainnetome_imp'

    with open(epi_txt) as f:
        epiIds = f.readlines()

    with open(nonepi_txt) as f:
        nonepiIds = f.readlines()

    epiIds = list(map(lambda x: x.strip(), epiIds))
    nonepiIds = list(map(lambda x: x.strip(), nonepiIds))

    epi_data, epi_subids = readsubs(studydir, epiIds)

    nonepi_data, nonepi_subids = readsubs(studydir, nonepiIds)

    # Do Pointwise stats
    #    pointwise_stats(epi_data, nonepi_data)

    # Do ROIwise stats
    epi_measures, nonepi_measures = roiwise_stats(epi_data, nonepi_data)

    X = np.vstack((epi_measures, nonepi_measures))
    y = np.hstack(
        (np.ones(epi_measures.shape[0]), np.zeros(nonepi_measures.shape[0])))

    X /= 3000
    #y = np.random.permutation(y)
    #p = np.random.permutation(len(y))
    #y = y[p]
    #X = X[p, :]

    cval = 4

    # for cval in [0.0001, 0.001, 0.01, .1, .3, .6, .9, 1, 1.5, 2, 3, 4, 5, 6, 7, 9, 10, 100]:
    #    for mygamma in [1, 0.001, 0.05, 0.075, .1, .15, 0.2, 0.3, .5, 1, 5, 10, 100]:
    clf = SVC(kernel='linear', C=cval, tol=1e-9)
    clf.fit(normalize(X), y)

    features_names = [
        '301', '300', '401', '400', '101', '100', '201', '200', '501', '500',
        '900'
    ]

    roi_ids = [301, 300, 401, 400, 101, 100, 201, 200, 501, 500, 900]

    roi_ids = ni.load_img('/home/ajoshi/Software/BrainSuite23a/svreg/USCBrainMulti/Brainnetome/BCI-Brainnetome.label.nii.gz').get_fdata()
    roi_ids = np.unique(roi_ids.flatten())

    f_importances((clf.coef_).squeeze(), features_names, outbase=outbase)

    f_importances_atlas((clf.coef_).squeeze(), roi_ids=roi_ids,
                        atlasbasename='/home/ajoshi/Software/BrainSuite23a/svreg/USCBrainMulti/Brainnetome/BCI-Brainnetome', outbase=outbase)

    my_metric = 'roc_auc'
    auc = cross_val_score(clf, X, y, cv=37, scoring=my_metric)

    print('AUC on testing data:', cval, np.mean(auc), np.std(auc))
    #print('AUC on training data:', cval, np.mean(auc_t), np.std(auc_t))

    print('done')
    #input("Press Enter to continue...")


if __name__ == "__main__":
    main()
