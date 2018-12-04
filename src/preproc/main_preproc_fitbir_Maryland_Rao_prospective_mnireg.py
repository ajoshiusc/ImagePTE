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

    study_dir = '/big_disk/ajoshi/fitbir/maryland_rao/MM_Prospective_ImagingMR_314'
    # List of subjects that maryland_rao
    preproc_dir = '/big_disk/ajoshi/fitbir/preproc'
    study_name = 'maryland_rao_v1'

    subdir = os.path.join(preproc_dir, study_name, subid)

    t1 = os.path.join(subdir, 'T1r.nii.gz')
    t1mnir = os.path.join(subdir, 'T1mni.nii.gz')
    t1mnimask = os.path.join(subdir, 'T1mni.mask.nii.gz')
#    t1bfc = os.path.join(subdir, 'T1.bfc.nii.gz')

    t1mnimat = os.path.join(subdir, 'T1mni.mat')
    print(subid)

    if not os.path.isfile(t1):
        return

    # register T1 image to MNI space

    os.system('flirt -in ' + t1 + ' -out ' + t1mnir +
              ' -ref ${FSLDIR}/data/standard/MNI152_T1_1mm_brain.nii.gz -omat '
              + t1mnimat + ' -dof 6 -cost normmi')

    t1 = os.path.join(subdir, 'T1r.nii.gz')
    t1mni = os.path.join(subdir, 'T1mni.nii.gz')

    if os.path.isfile(t1):
        # Create mask
        os.system('fslmaths ' + t1mni + ' -bin ' + t1mnimask)
        # Apply the same transform (T1->MNI)

        t2 = os.path.join(subdir, 'T2r.nii.gz')
        t2mni = os.path.join(subdir, 'T2mni.nii.gz')
    if os.path.isfile(t2):
        # Apply the same transform (T1->MNI) to registered T2
        os.system('flirt -in ' + t2 + ' -ref ' + t1mnir + ' -out ' + t2mni +
                  ' -applyxfm -init ' + t1mnimat)
        os.system('fslmaths ' + t2mni + ' -mul ' + t1mnimask + ' ' + t2mni)

    flair = os.path.join(subdir, 'FLAIRr.nii.gz')
    flairmni = os.path.join(subdir, 'FLAIRmni.nii.gz')
    if os.path.isfile(flair):
        # Apply the same transform (T1->MNI) to registered FLAIR
        os.system('flirt -in ' + flair + ' -ref ' + t1mnir + ' -out ' +
                  flairmni + ' -applyxfm -init ' + t1mnimat)
        os.system('fslmaths ' + flairmni + ' -mul ' + t1mnimask + ' ' +
                  flairmni)


def main():
    #Set subject dirs
    med_hist_csv = '/big_disk/ajoshi/fitbir/maryland_rao/FITBIR Demographics_314/FITBIRdemographics_prospective.csv'
    subIds = pd.read_csv(med_hist_csv, index_col=1)
    pool = Pool(processes=8)
    tbi_done_list = '/big_disk/ajoshi/fitbir/preproc/maryland_rao_v1_done.txt'

    with open(tbi_done_list) as f:
        tbidoneIds = f.readlines()

    print(subIds.index)
    subsnotdone = [x for x in subIds.index if x not in tbidoneIds]

    pool.map(regparfun, subsnotdone)

    pool.close()
    pool.join()


if __name__ == "__main__":
    main()
