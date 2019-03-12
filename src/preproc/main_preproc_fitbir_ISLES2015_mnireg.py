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


#    study_dir = '/big_disk/ajoshi/fitbir/tracktbi/MM_Prospective_ImagingMR_314'
# List of subjects that maryland_rao
    preproc_dir = '/big_disk/ajoshi/ISLES2015/preproc/Training'

    subdir = os.path.join(preproc_dir, subid)

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
        # Apply the same transform (T1->MNI)

        t2 = os.path.join(subdir, 'T2.nii.gz')
        t2mni = os.path.join(subdir, 'T2mni.nii.gz')
    if os.path.isfile(t2):
        # Apply the same transform (T1->MNI) to registered T2
        os.system('flirt -in ' + t2 + ' -ref ' + t1mni + ' -out ' + t2mni +
                  ' -applyxfm -init ' + t1mnimat)
        os.system('fslmaths ' + t2mni + ' -mul ' + t1mnimask + ' ' + t2mni)

    flair = os.path.join(subdir, 'FLAIR.nii.gz')
    flairmni = os.path.join(subdir, 'FLAIRmni.nii.gz')
    if os.path.isfile(flair):
        # Apply the same transform (T1->MNI) to registered FLAIR
        os.system('flirt -in ' + flair + ' -ref ' + t1mni + ' -out ' +
                  flairmni + ' -applyxfm -init ' + t1mnimat)
        os.system('fslmaths ' + flairmni + ' -mul ' + t1mnimask + ' ' +
                  flairmni)

    seg = os.path.join(subdir, 'SEGMENTATION.nii.gz')
    segmni = os.path.join(subdir, 'SEGMENTATIONmni.nii.gz')
    if os.path.isfile(seg):
        # Apply the same transform (T1->MNI) to registered FLAIR
        os.system('flirt -in ' + seg + ' -ref ' + t1mni + ' -out ' + segmni +
                  ' -applyxfm -init ' + t1mnimat)
        os.system('fslmaths ' + segmni + ' -mul ' + t1mnimask + ' ' + segmni)


def main():
    #Set subject dirs
    subIds = list(range(1, 29))

    pool = Pool(processes=12)

    #    regparfun(subsnotdone[2])

    print('++++++++++++++')
    pool.map(regparfun, subIds)

    pool.close()
    pool.join()


if __name__ == "__main__":
    main()
