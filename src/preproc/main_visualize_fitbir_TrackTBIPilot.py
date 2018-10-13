#||AUM||
#||Shree Ganeshaya Namaha||

import pandas as pd
import glob
import os
import shutil
from fitbirpre import zip2nii, reg2mni, name2modality
import nilearn.image as ni
from nilearn import plotting
import numpy as np

def main():
    #Set subject dirs
    study_name = 'tracktbi_pilot'
    med_hist_csv = '/big_disk/ajoshi/fitbir/tracktbi_pilot/Baseline Med History_246/TrackTBI_MedicalHx.csv'
    study_dir = '/big_disk/ajoshi/fitbir/tracktbi_pilot/TRACK TBI Pilot - MR data -'

    preproc_dir = '/big_disk/ajoshi/fitbir/preproc'
    subIds = pd.read_csv(med_hist_csv, index_col=1)
    # print(subIds)
    ''' If fMRI data exists for some subjects, then store their cognitive scores '''
    for subid in subIds.index[2:]:
        print(subid)
        if not isinstance(subid, str):
            continue

        dirname = os.path.join(preproc_dir, study_name, subid)

        fnamet1 = os.path.join(dirname, 'T1.nii.gz')
        fnamet2 = os.path.join(dirname, 'T2.nii.gz')
        fnameflair = os.path.join(dirname, 'FLAIR.nii.gz')

        print('+++++++++++' + subid + '++++++++++++++')

        if os.path.isfile(fnamet1):
            imgt1 = ni.load_img(fnamet1)
            plotting.plot_anat(
                imgt1,
                title="plot_anat",
                cut_coords=(0, 0, 0),
                output_file=subid + '_T1.png',
                vmin=0,
                vmax=np.percentile(imgt1.get_data().flatten(), 95))

        if os.path.isfile(fnamet2):
            imgt2 = ni.load_img(fnamet2)
            plotting.plot_anat(
                imgt2,
                title="plot_anat",
                cut_coords=(0, 0, 0),
                output_file=subid + '_T2.png',
                vmin=0,
                vmax=np.percentile(imgt2.get_data().flatten(), 95))

        if os.path.isfile(fnameflair):
            imgflair = ni.load_img(fnameflair)
            plotting.plot_anat(
                imgflair,
                title="plot_anat",
                cut_coords=(0, 0, 0),
                output_file=subid + '_FLAIR.png',
                vmin=0,
                vmax=np.percentile(imgflair.get_data().flatten(), 95))


if __name__ == "__main__":
    main()
