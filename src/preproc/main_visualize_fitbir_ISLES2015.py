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

    preproc_dir = '/big_disk/ajoshi/ISLES2015/preproc/Training'
    pngdir = '/big_disk/ajoshi/ISLES2015/preproc/Training/pngout/'

    # print(subIds)
    ''' If fMRI data exists for some subjects, then store their cognitive scores '''
    for subid in range(1, 29):  #subIds.index:

        dirname = os.path.join(preproc_dir, str(subid))

        fnamet1 = os.path.join(dirname, 'T1mni.nii.gz')
        fnamet2 = os.path.join(dirname, 'T2mni.nii.gz')
        fnameflair = os.path.join(dirname, 'FLAIRmni.nii.gz')

        imgt1 = ni.load_img(fnamet1)
        t1 = imgt1.get_data()
        t1img = t1[91, :, :].squeeze()

        imgt2 = ni.load_img(fnamet2)
        t2 = imgt2.get_data()
        t2img = t2[91, :, :].squeeze()

        imgflair = ni.load_img(fnameflair)
        imgflair = imgflair.get_data()
        flairimg = imgflair[91, :, :].squeeze()
        #            imgfull = np.hstack((imrotate(t1img, 90), imrotate(t2img, 90),
        #                                 imrotate(flairimg, 90)))
        imgfull = np.hstack((np.flipud(t1img.T), np.flipud(t2img.T),
                             np.flipud(flairimg.T)))

        vmax1 = np.percentile(imgfull.flatten(), 95)

        plt.imsave(
            pngdir + str(subid) + '_mni_sag.png',
            imgfull,
            cmap='gray',
            vmin=0,
            vmax=vmax1)

    print('Done')


if __name__ == "__main__":
    main()
