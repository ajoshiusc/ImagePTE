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
    bst_subdir = os.path.join(subdir, 'BrainSuite')
    os.makedirs(bst_subdir)

    t1mni = os.path.join(subdir, 'T1mni.nii.gz')
    t1mnimask = os.path.join(subdir, 'T1mni.mask.nii.gz')

    # If the file does not exist, then return
    if not os.path.isfile(t1mni):
        return

    # copy files to BrainSuite directory for processing
    copyfile(t1mni, bst_subdir)
    copyfile(t1mnimask, bst_subdir)

    t1mni = os.path.join(bst_subdir, 'T1mni.nii.gz')
    pial_surf = os.path.join(bst_subdir, 'T1mni.left.pial.cortex.dfs')

    print(subid)

    # Return if BrainSuite sequence has already been run
    if os.path.isfile(pial_surf):
        return

    # Run BrainSuite sequence
    os.system(
        '/big_disk/ajoshi/coding_ground/ImagePTE/src/preproc/brainsuite_fitbir.sh '
        + t1mni)


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
