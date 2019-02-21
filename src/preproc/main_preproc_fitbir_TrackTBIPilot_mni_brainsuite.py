import pandas as pd
import glob
import os
import shutil
from fitbirpre import zip2nii, reg2mni_re, name2modality
from multiprocessing import Pool
from itertools import product, repeat
import numpy as np
import numbers
from shutil import copyfile, copy
import time


def regparfun(subid):

    # List of subjects that maryland_rao
    preproc_dir = '/big_disk/ajoshi/fitbir/preproc'
    study_name = 'tracktbi_pilot'

    subdir = os.path.join(preproc_dir, study_name, subid)
    bst_subdir = os.path.join(subdir, 'BrainSuite')

    # If BrainSuite directory does not exist then create it
    if not os.path.isdir(bst_subdir):
        os.makedirs(bst_subdir)

    t1mni = os.path.join(subdir, 'T1mni.nii.gz')
    t1mnimask = os.path.join(subdir, 'T1mni.mask.nii.gz')
    t1mnimask_bst = os.path.join(bst_subdir, 'T1mni.mask.nii.gz')
    # If the file does not exist, then return
    if not os.path.isfile(t1mni):
        return

    # copy files to BrainSuite directory for processing
    copy(t1mni, bst_subdir)
    os.system('fslmaths ' + t1mnimask + ' ' + t1mnimask_bst +
              ' -odt char')  # Convert the mask to uint8

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
    med_hist_csv = '/big_disk/ajoshi/fitbir/tracktbi_pilot/Baseline Med History_246/TrackTBI_MedicalHx.csv'
    subIds = pd.read_csv(med_hist_csv, index_col=1)
    pool = Pool(processes=8)
    tbi_done_list = '/big_disk/ajoshi/fitbir/preproc/tracktbi_pilot_done.txt'

    with open(tbi_done_list) as f:
        tbidoneIds = f.readlines()

    # remove newline characters
    tbidoneIds = list(map(lambda x: x.strip(), tbidoneIds))

    print(subIds.index)
    subslist = [x for x in subIds.index if x in tbidoneIds]

    #regparfun(subsnotdone[1])
    pool.map(regparfun, subslist)

    pool.close()
    pool.join()


if __name__ == "__main__":
    main()
