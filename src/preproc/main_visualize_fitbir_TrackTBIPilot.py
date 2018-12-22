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
import matplotlib.pyplot as plt
from scipy.misc import imrotate


def main():
    #Set subject dirs
    study_name = 'tracktbi_pilot'
    med_hist_csv = '/big_disk/ajoshi/fitbir/tracktbi_pilot/Baseline Med History_246/TrackTBI_MedicalHx.csv'
    #    study_dir = '/big_disk/ajoshi/fitbir/tracktbi_pilot/TRACK TBI Pilot - MR data -'

    preproc_dir = '/big_disk/ajoshi/fitbir/preproc'
    pngdir = '/big_disk/ajoshi/fitbir/preproc/tracktbi_pilot/pngout/'
    subIds = pd.read_csv(med_hist_csv, index_col=1)

   # This contains a list of TBI subjects that are done correctly
    tbi_done_list = '/big_disk/ajoshi/fitbir/preproc/tracktbi_done.txt'

    with open(tbi_done_list) as f:
        tbidoneIds = f.readlines()

    # Get the list of subjects that are correctly registered
    tbidoneIds = [l.strip('\n\r') for l in tbidoneIds]

    # print(subIds)
    for subid in subIds.index:

        if not isinstance(subid, str):
            continue

        if any(subid in s for s in tbidoneIds):
            print(subid + ' is already done')
            continue

        dirname = os.path.join(preproc_dir, study_name, subid)

        fnamet1 = os.path.join(dirname, 'T1mni.nii.gz')
        fnamet2 = os.path.join(dirname, 'T2mni.nii.gz')
        fnameflair = os.path.join(dirname, 'FLAIRmni.nii.gz')

        if os.path.isfile(fnamet1) and os.path.isfile(
                fnamet2) and os.path.isfile(fnameflair):

            imgt1 = ni.load_img(fnamet1)
            t1 = imgt1.get_data()
            t1img = t1[91, :, :].squeeze()

            imgt2 = ni.load_img(fnamet2)
            t2 = imgt2.get_data()
            t2img = t2[91, :, :].squeeze()

            imgflair = ni.load_img(fnameflair)
            imgflair = imgflair.get_data()
            flairimg = imgflair[91, :, :].squeeze()
            imgfull = np.hstack((imrotate(t1img, 90), imrotate(t2img, 90),
                                 imrotate(flairimg, 90)))

            vmax1 = np.percentile(imgfull.flatten(), 95)

            plt.imsave(
                pngdir+subid + '_sag.png', imgfull, cmap='gray', vmin=0, vmax=vmax1)
        else:
            print('some files do not exist for:' + subid)


if __name__ == "__main__":
    main()
