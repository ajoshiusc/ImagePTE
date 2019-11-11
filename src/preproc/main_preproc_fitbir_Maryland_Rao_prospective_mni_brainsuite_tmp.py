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

    study_dir = '/big_disk/ajoshi/fitbir/maryland_rao/MM_Prospective_ImagingMR_314'
    # List of subjects that maryland_rao
    preproc_dir = '/big_disk/ajoshi/fitbir/preproc'
    study_name = 'maryland_rao_v1'

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


    # Return if BrainSuite sequence has already been run
    if not os.path.isfile(pial_surf):
        print('BSTTBD'+subid)

        # Run BrainSuite sequence
        #os.system(
        #    '/big_disk/ajoshi/coding_ground/ImagePTE/src/preproc/brainsuite_fitbir.sh '
        #    + t1mni)

    # Run SVReg sequence
    subbasename = os.path.join(bst_subdir, 'T1mni')
    csv_txt = subbasename + '.roiwise.stats.txt'
    print(csv_txt)

    #if (not os.path.isfile(csv_txt)) and os.path.isfile(pial_surf):
    #    print('SVRegTBD'+subid)

    if (os.path.isfile(csv_txt)):
        print('SVReg Done'+subid)

        #os.system(
        #    '/home/ajoshi/BrainSuite19a/svreg/bin/svreg.sh ' + subbasename +
        #    ' /home/ajoshi/BrainSuite19a/svreg/BCI-DNI_brain_atlas/BCI-DNI_brain -U'
        #)


def main():
    #Set subject dirs
    med_hist_csv = '/big_disk/ajoshi/fitbir/maryland_rao/FITBIR Demographics_314/FITBIRdemographics_prospective.csv'
    subIds = pd.read_csv(med_hist_csv, index_col=1)
    pool = Pool(processes=8)
    tbi_done_list = '/big_disk/ajoshi/fitbir/preproc/maryland_rao_v1_nonepilepsy_imgs.txt'

    with open(tbi_done_list) as f:
        tbidoneIds = f.readlines()

    tbidoneIds = list(map(lambda x: x.strip(), tbidoneIds))
    print(subIds.index)
    subsnotdone = [x for x in subIds.index if x in tbidoneIds]

    #regparfun(subsnotdone[1])
    pool.map(regparfun, subsnotdone)

    pool.close()
    pool.join()


if __name__ == "__main__":
    main()
