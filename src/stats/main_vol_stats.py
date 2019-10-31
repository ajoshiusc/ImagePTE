import pandas as pd
import glob
import os
import shutil
from multiprocessing import Pool
from itertools import product, repeat
import numpy as np
import numbers
from shutil import copyfile, copy
import time
import nilearn.image as ni


def check_imgs_exist(studydir, sub_ids):
    subids_imgs = list()

    for id in sub_ids:
        fname_T1 = os.path.join(studydir, id, 'T1mni.nii.gz')
        fname_T2 = os.path.join(studydir, id, 'T2mni.nii.gz')
        fname_FLAIR = os.path.join(studydir, id, 'FLAIRmni.nii.gz')

        if os.path.isfile(fname_T1) and os.path.isfile(
                fname_T2) and os.path.isfile(fname_FLAIR):

            subids_imgs.append(id)

    return subids_imgs


def readsubs(studydir, sub_ids):

    print(len(sub_ids))

    sub_ids = check_imgs_exist(studydir, sub_ids)

    print(len(sub_ids))

    for n, id in enumerate(sub_ids):
        fname_T1 = os.path.join(studydir, id, 'T1mni.nii.gz')
        fname_T2 = os.path.join(studydir, id, 'T2mni.nii.gz')
        fname_FLAIR = os.path.join(studydir, id, 'FLAIRmni.nii.gz')
        print('sub:', n, 'Reading', id)
        t1 = ni.load_img(fname_T1)
        t2 = ni.load_img(fname_T2)
        flair = ni.load_img(fname_FLAIR)

        if n == 0:
            data = np.zeros(t1.shape + (3, len(sub_ids)))

            data[:, :, :, 0, n] = t1.get_data()
            data[:, :, :, 1, n] = t2.get_data()
            data[:, :, :, 2, n] = flair.get_data()

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

    epi_data, epi_subids = readsubs(studydir, epiIds)

    nonepi_data, nonepi_subids = readsubs(studydir, nonepiIds)

    print('done')


if __name__ == "__main__":
    main()
