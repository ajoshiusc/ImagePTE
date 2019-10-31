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
        print(n, 'Reading', id)
        t1 = ni.load_img(fname_T1)
        t2 = ni.load_img(fname_T2)
        flair = ni.load_img(fname_FLAIR)

        if n == 0:
            data = np.zeros(t1.shape + (3, len(sub_ids)))

            data[:, :, :, 0, n] = t1.get_data()
            data[:, :, :, 1, n] = t2.get_data()
            data[:, :, :, 2, n] = flair.get_data()

    return data


def main():

    studydir = '/big_disk/ajoshi/fitbir/preproc/maryland_rao_v1'

    tbi_done_list = '/big_disk/ajoshi/fitbir/preproc/maryland_rao_v1_epilepsy_imgs.txt'

    with open(tbi_done_list) as f:
        epiIds = f.readlines()

    epiIds = list(map(lambda x: x.strip(), epiIds))
    print(epiIds)

    data1 = readsubs(studydir, epiIds)


if __name__ == "__main__":
    main()
