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

    study_dir = '/ImagePTE1/ajoshi/fitbir/maryland_rao/MM_Prospective_ImagingMR_314'
    # List of subjects that maryland_rao
    preproc_dir = '/ImagePTE1/ajoshi/fitbir/preproc'
    study_name = 'maryland_rao_v1'

    subdir = os.path.join(preproc_dir, study_name, subid)

    t1 = os.path.join(subdir, 'T1r.nii.gz')
    t1BCIr = os.path.join(subdir, 'T1BCI.nii.gz')
    t1BCImask = os.path.join(subdir, 'T1BCI.mask.nii.gz')
    #    t1bfc = os.path.join(subdir, 'T1.bfc.nii.gz')

    t1BCImat = os.path.join(subdir, 'T1BCI.mat')
    print(subid)

    if not os.path.isfile(t1):
        return

    # register T1 image to MNI space

    os.system('flirt -in ' + t1 + ' -out ' + t1BCIr +
              ' -ref /home/ajoshi/BrainSuite19a/svreg/BCI-DNI_brain_atlas/BCI-DNI_brain.bfc.nii.gz -omat '
              + t1BCImat + ' -dof 6')

    t1 = os.path.join(subdir, 'T1r.nii.gz')
    t1BCI = os.path.join(subdir, 'T1BCI.nii.gz')

    if os.path.isfile(t1):
        # Create mask
#        os.system('fslmaths ' + t1BCI + ' -bin ' + t1BCImask)
        # Apply the same transform (T1->MNI)
#        os.system('flirt -in ' + t1 + ' -ref ' + t1BCIr + ' -out ' + t1BCI +
#                  ' -applyxfm -init ' + t1BCImat)
#        

        t2 = os.path.join(subdir, 'T2r.nii.gz')
        t2BCI = os.path.join(subdir, 'T2BCI.nii.gz')
    if os.path.isfile(t2):
        # Apply the same transform (T1->MNI) to registered T2
        os.system('flirt -in ' + t2 + ' -ref ' + t1BCIr + ' -out ' + t2BCI +
                  ' -applyxfm -init ' + t1BCImat)
#        os.system('fslmaths ' + t2BCI + ' -mul ' + t1BCImask + ' ' + t2BCI)

    flair = os.path.join(subdir, 'FLAIRr.nii.gz')
    flairBCI = os.path.join(subdir, 'FLAIRBCI.nii.gz')
    if os.path.isfile(flair):
        # Apply the same transform (T1->MNI) to registered FLAIR
        os.system('flirt -in ' + flair + ' -ref ' + t1BCIr + ' -out ' +
                  flairBCI + ' -applyxfm -init ' + t1BCImat)
#        os.system('fslmaths ' + flairBCI + ' -mul ' + t1BCImask + ' ' +
#                  flairBCI)


def main():
    #Set subject dirs
    pool = Pool(processes=8)
    epi_txt = '/ImagePTE1/ajoshi/fitbir/preproc/maryland_rao_v1_epilepsy_imgs.txt'
    nonepi_txt = '/ImagePTE1/ajoshi/fitbir/preproc/maryland_rao_v1_nonepilepsy_imgs.txt'


    with open(epi_txt) as f:
        epiIds = f.readlines()

    with open(nonepi_txt) as f:
        nonepiIds = f.readlines()

    epiIds = list(map(lambda x: x.strip(), epiIds))
    nonepiIds = list(map(lambda x: x.strip(), nonepiIds))


    pool.map(regparfun, epiIds)
    pool.map(regparfun, nonepiIds)

    pool.close()
    pool.join()


if __name__ == "__main__":
    main()
