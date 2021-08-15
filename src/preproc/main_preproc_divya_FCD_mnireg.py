import pandas as pd
import glob
import os
import shutil
from fitbirpre import zip2nii, reg2mni_re, name2modality
from multiprocessing import Pool
from itertools import product, repeat
import numpy as np
import numbers
from shutil import copyfile
import time


def regparfun(subid):

    subid = str(subid)
    print(subid)
    if not isinstance(subid, str):
        return


#   Study_dir = '/big_disk/ajoshi/fitbir/tracktbi/MM_Prospective_ImagingMR_314'
#   List of subjects that maryland_rao
    subdir = subid

    t1 = os.path.join(subdir, 'T1.nii.gz')
    t1mni = os.path.join(subdir, 'T1mni.nii.gz')
    t1mnimask = os.path.join(subdir, 'T1mni.mask.nii.gz')
    #    t1bfc = os.path.join(subdir, 'T1.bfc.nii.gz')

    t1mnimat = os.path.join(subdir, 'T1mni.mat')
    print(subid)

    if not os.path.isfile(t1):
        print('T1 does not exist for ' + subid + t1)
        return

    # register T1 image to MNI space

    os.system('flirt -in ' + t1 + ' -out ' + t1mni +
              ' -ref ${FSLDIR}/data/standard/MNI152_T1_1mm_brain.nii.gz -omat '
              + t1mnimat + ' -dof 6 -cost normmi')

    t1 = os.path.join(subdir, 'T1.nii.gz')
    t1mni = os.path.join(subdir, 'T1mni.nii.gz')

    if os.path.isfile(t1):
        # Create mask
        os.system('fslmaths ' + t1mni + ' -bin ' + t1mnimask)

    return 0


def main():
    #Set subject dirs


    subIds = glob.glob('/ImagePTE1/ajoshi/FCD_divya/preproc/*')



    pool = Pool(processes=4)

    #regparfun(subIds[2])

    print('++++++++++++++')
    pool.map(regparfun, subIds)
    print('++++SUBMITTED++++++')

    pool.close()
    pool.join()


if __name__ == "__main__":
    main()
