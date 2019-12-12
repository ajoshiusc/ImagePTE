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
from nilearn.plotting import plot_anat


def regparfun(subid):

    bcireg_dir = '/ImagePTE1/ajoshi/fitbir/maryland_rao_BCIreg/nonPTE'
    # List of subjects that maryland_rao
    preproc_dir = '/ImagePTE1/ajoshi/fitbir/preproc'
    study_name = 'maryland_rao_v1'

    subdir = os.path.join(preproc_dir, study_name, subid)

    t1BCI = os.path.join(subdir, 'T1BCI.nii.gz')
    flairBCI = os.path.join(subdir, 'FLAIRBCI.nii.gz')

    outfile = os.path.join(bcireg_dir, subid + '_t1.png')

    plot_anat(t1BCI,
              threshold=None,
              vmax=555,
              vmin=0,
              draw_cross=False,
              cut_coords=(42 * 0.8, 180 * 0.546875, 215 * 0.546875),
              display_mode="ortho",
              output_file=outfile,
              annotate=True)

    outfile = os.path.join(bcireg_dir, subid + '_flair.png')

    plot_anat(flairBCI,
              threshold=None,
              vmax=555,
              vmin=0,
              draw_cross=False,
              cut_coords=(42 * 0.8, 180 * 0.546875, 215 * 0.546875),
              display_mode="ortho",
              output_file=outfile,
              annotate=True)


def main():
    #Set subject dirs

    epi_txt = '/ImagePTE1/ajoshi/fitbir/preproc/maryland_rao_v1_epilepsy_imgs.txt'
    nonepi_txt = '/ImagePTE1/ajoshi/fitbir/preproc/maryland_rao_v1_nonepilepsy_imgs.txt'

    with open(epi_txt) as f:
        epiIds = f.readlines()

    with open(nonepi_txt) as f:
        nonepiIds = f.readlines()



    nonepiIds = list(map(lambda x: x.strip(), nonepiIds))

    for sub in nonepiIds:
        regparfun(sub)

 


if __name__ == "__main__":
    main()
