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

def regparfun(subdir, infile):
    modname = name2modality(infile)
    if modname is not None:
        outfname = os.path.join(subdir, modname)
    else:
        outfname = ''


    if (modname is not None) and (modname is not 'rest'):
        if not os.path.isfile(outfname + '.nii.gz'):
            reg2mni_re(infile=infile, outfile=outfname)
        
    if modname is 'rest':
        copyfile(infile, outfname + '.nii.gz')


def main():
    #Set subject dirs
    study_name = 'maryland_rao_v1'
    med_hist_csv = '/big_disk/ajoshi/fitbir/maryland_rao/FITBIR Demographics_314/FITBIRdemographics_prospective.csv'
    study_dir = '/big_disk/ajoshi/fitbir/maryland_rao/MM_Prospective_ImagingMR_314'

    # List of subjects that maryland_rao
    tbi_done_list = '/big_disk/ajoshi/fitbir/preproc/maryland_rao_done.txt'
    preproc_dir = '/big_disk/ajoshi/fitbir/preproc'
    subIds = pd.read_csv(med_hist_csv, index_col=1)

    pool = Pool(processes=4)

    for subid in subIds.index:
        print(subid)

        if not isinstance(subid, str):
            continue

        dirlist = glob.glob(study_dir + '*/' + subid + '*_mrtbi_*v1*.nii')
        print(dirlist)
        if len(dirlist) > 0:
            subdir = os.path.join(preproc_dir, study_name, subid)
            print('hi' + subdir)
            img_subdir = os.path.join(subdir, 'orig')

            # Create subject directory
            if not os.path.exists(img_subdir):
                os.makedirs(img_subdir)

            # Normalize all images to standard MNI space.
            imgfiles = dirlist #glob.glob(img_subdir + '/*.nii.gz')
            print('running proc func')
            print('done proc func')
            pool.starmap(regparfun, zip(repeat(subdir), imgfiles))

    pool.close()
    pool.join()


if __name__ == "__main__":
    main()
