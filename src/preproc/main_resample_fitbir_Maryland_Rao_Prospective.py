import pandas as pd
import glob
import os
import shutil
from fitbirpre import zip2nii, reg2mni, name2modality
import nilearn.image as ni
from nilearn import plotting
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imrotate


def main():
    #Set subject dirs
    study_dir = '/big_disk/ajoshi/fitbir/maryland_rao/MM_Prospective_ImagingMR_314'

    preproc_dir = '/big_disk/ajoshi/fitbir/preproc'
    pngdir = '/big_disk/ajoshi/fitbir/preproc/maryland_rao_v1/pngout/'
    med_hist_csv = '/big_disk/ajoshi/fitbir/maryland_rao/FITBIR Demographics_314/FITBIRdemographics_prospective.csv'
    subIds = pd.read_csv(med_hist_csv, index_col=1)
    study_name = 'maryland_rao_v1'
    # This contains a list of TBI subjects that are done correctly
    tbi_done_list = '/big_disk/ajoshi/fitbir/preproc/tracktbi_done.txt'

    with open(tbi_done_list) as f:
        tbidoneIds = f.readlines()

    # Get the list of subjects that are correctly registered
    tbidoneIds = [l.strip('\n\r') for l in tbidoneIds]

    # print(subIds)
    ''' If fMRI data exists for some subjects, then store their cognitive scores '''
    for subid in subIds.index:

        if not isinstance(subid, str):
            continue

        if any(subid in s for s in tbidoneIds):
            print(subid + ' is already done')
            continue

        dirname = os.path.join(preproc_dir, study_name, subid)

        t1 = os.path.join(dirname, 'T1.nii')
        t2 = os.path.join(dirname, 'T2.nii')
        flair = os.path.join(dirname, 'FLAIR.nii')
        swi = os.path.join(dirname, 'SWI.nii')

        if os.path.isfile(t1) and os.path.isfile(t2):
            t2r = ni.resample_to_img(t2, t1)
            t2r.to_filename(t2[:-4] + 'r.nii.gz')

        if os.path.isfile(t1) and os.path.isfile(flair):
            flairr = ni.resample_to_img(flair, t1)
            flairr.to_filename(flair[:-4] + 'r.nii.gz')

        if os.path.isfile(t1) and os.path.isfile(swi):
            swir = ni.resample_to_img(swi, t1)
            swir.to_filename(swi[:-4] + 'r.nii.gz')


if __name__ == "__main__":
    main()
