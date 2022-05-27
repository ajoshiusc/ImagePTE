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
    preproc_dir = '/ImagePTE1/ajoshi/camcan/cc700/mri/pipeline/release004/BIDS_20190411/anat'
    subdir = os.path.join(preproc_dir, subid)

    outdir = '/ImagePTE1/ajoshi/CamCan/'
    outdir = os.path.join(outdir, subid)
    
    if not os.path.isdir(outdir):
        os.mkdir(outdir)

    t1 = os.path.join(subdir, 'anat/' + subid + '_T1w.nii.gz')
    t2 = os.path.join(subdir, 'anat/' + subid + '_T2w.nii.gz')
    t1bse = os.path.join(outdir, 'T1.bse.nii.gz')
    t1mni = os.path.join(outdir, 'T1mni.nii.gz')
    t1mnimask = os.path.join(outdir, 'T1mni.mask.nii.gz')
    t1mnimat = os.path.join(outdir, 'T1mni.mat')
    t2mni = os.path.join(outdir, 'T2mni.nii.gz')

    if not os.path.isfile(t1):
        print('T1 does not exist for ' + subid + t1)
        return

    os.system('bet ' + t1 + ' ' + t1bse + ' -f .3')

    # register T1 image to MNI space

    os.system('flirt -in ' + t1bse + ' -out ' + t1mni +
              ' -ref ${FSLDIR}/data/standard/MNI152_T1_1mm_brain.nii.gz -omat '
              + t1mnimat + ' -dof 6 -cost normmi')

    print(subid)

    # Create mask
    os.system('fslmaths ' + t1mni + ' -bin ' + t1mnimask)
        # Apply the same transform (T1->MNI)


    if os.path.isfile(t2):
        # Apply the same transform (T1->MNI) to registered T2
        os.system('flirt -in ' + t2 + ' -ref ${FSLDIR}/data/standard/MNI152_T1_1mm_brain.nii.gz -out ' +
                  t2mni + ' -applyxfm -init ' + t1mnimat)
        os.system('fslmaths ' + t2mni + ' -mul ' + t1mnimask + ' ' + t2mni)

    return 0


def main():
    # Set subject dirs
    tbi_done_list = '/ImagePTE1/akrami/CamCan/sublist_subset.txt'

    with open(tbi_done_list) as f:
        tbidoneIds = f.readlines()

    # Get the list of subjects that are correctly registered
    tbidoneIds = [l.strip('\n\r') for l in tbidoneIds]

    pool = Pool(processes=4)

    #for j in range(10):
    #    regparfun(tbidoneIds[j])

    print('++++++++++++++')
    pool.map(regparfun, tbidoneIds)
    print('++++SUBMITTED++++++')

    # pool.close()
    # pool.join()


if __name__ == "__main__":
    main()