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


def regparfun(subid):

    study_dir = '/big_disk/ajoshi/fitbir/maryland_rao/MM_Prospective_ImagingMR_314'
    # List of subjects that maryland_rao
    tbi_done_list = '/big_disk/ajoshi/fitbir/preproc/maryland_rao_done.txt'
    preproc_dir = '/big_disk/ajoshi/fitbir/preproc'
    study_name = 'maryland_rao_v1'

    dirlist = glob.glob(study_dir + '*/' + subid + '*_mrtbi_*v1*.nii')
    print(dirlist)
    if len(dirlist) > 0:
        subdir = os.path.join(preproc_dir, study_name, subid)

        # Normalize all images to standard MNI space.
        imgfiles = dirlist

        t1 = os.path.join(subdir, 'T1.nii.gz')
        t1mni = os.path.join(subdir, 'T1mni.nii.gz')
        t1mnimat = os.path.join(subdir, 'T1mni.mat')
        if not os.path.isfile(t1):
            return

        # register T1 image to MNI space
        os.system('./first_flirt_rigid ' + t1 + ' ' + t1mni)

        t2 = os.path.join(subdir, 'T2.nii.gz')
        t2r = os.path.join(subdir, 'T2r.nii.gz')
        t2mni = os.path.join(subdir, 'T2mni.nii.gz')
        if os.path.isfile(t2):
            # Register T2 to T1
            os.system('flirt -in ' + t2 + ' -nosearch -out ' + t2r + ' -ref ' +
                      t1 + ' -dof 6')
            # Apply the same transform (T1->MNI) to registered T2 to take it to mni space
            os.system('flirt -in ' + t2r + ' -ref ' + t1mni +
                      ' -applyxfm -init ' + t1mnimat + ' -out ' + t2mni)

        flair = os.path.join(subdir, 'FLAIR.nii.gz')
        flairr = os.path.join(subdir, 'FLAIRr.nii.gz')
        flairmni = os.path.join(subdir, 'FLAIRmni.nii.gz')
        if os.path.isfile(flair):
            # Register FLAIR to T1
            os.system('flirt -in ' + flair + ' -nosearch -out ' + flairr +
                      ' -ref ' + t1 + ' -dof 6')
            # Apply the same transform (T1->MNI) to registered FLAIR to take it to mni space
            os.system('flirt -in ' + flairr + ' -ref ' + t1mni +
                      ' -applyxfm -init ' + t1mnimat + ' -out ' + flairmni)


def main():
    #Set subject dirs
    med_hist_csv = '/big_disk/ajoshi/fitbir/maryland_rao/FITBIR Demographics_314/FITBIRdemographics_prospective.csv'
    subIds = pd.read_csv(med_hist_csv, index_col=1)
   # pool = Pool(processes=12)
    for subid in subIds.index:
        print(subid)
        if not isinstance(subid, str):
            continue

        regparfun(subid)
        #pool.starmap(regparfun, subid)

    pool.close()
    pool.join()


if __name__ == "__main__":
    main()
