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
            lesion_studydir, 'MSE_T1_' + str(n + 37 * nonepi) + '.nii.gz')

        fname_lesion_w = os.path.join(studydir, id, 'lesion_vae.atlas.nii.gz')
        fname_lesion_w_sm = os.path.join(studydir, id,
                                         'lesion_vae.atlas.smooth3mm.nii.gz')

        # Warp by applying the map
        if os.path.isfile(fname_lesion_w):
            print('File exists :' + fname_lesion_w)
        else:
            print('Applying SVReg inv map for:' + id)
            os.system(
                '/home/ajoshi/BrainSuite19a/svreg/bin/svreg_apply_map.sh ' +
                invmap + ' ' + fname_lesion + ' ' + fname_lesion_w + ' ' +
                ATLAS)

        # Smooth the warped image
        if os.path.isfile(fname_lesion_w_sm):
            print('File exists :' + fname_lesion_w_sm)
        else:
            print('Applying SVReg inv map for:' + id)
            os.system(
                '/home/ajoshi/BrainSuite19a/svreg/bin/svreg_smooth_vol_function.sh '
                + fname_lesion_w + ' 3 3 3 ' + fname_lesion_w_sm)


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

    studydir = '"/big_disk/akrami/git_repos_new/ImagePTE/src/Lesion Detection/models/3D_out"'
    studydir_imgs = '/big_disk/ajoshi/fitbir/preproc/maryland_rao_v1'

    epi_txt = '/big_disk/ajoshi/fitbir/preproc/maryland_rao_v1_epilepsy_imgs.txt'
    nonepi_txt = '/big_disk/ajoshi/fitbir/preproc/maryland_rao_v1_nonepilepsy_imgs_37.txt'
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

    print('done')


if __name__ == "__main__":
    main()
