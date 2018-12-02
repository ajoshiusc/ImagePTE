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
import nilearn.image as ni


def regparfun(subdir, infile):
    modname = name2modality(infile)
    if modname is not None:
        outfname = os.path.join(subdir, modname)
    else:
        outfname = ''

    copyfile(infile, outfname + '.nii')


"""     if (modname is not None) and (modname is not 'rest'):
        if not os.path.isfile(outfname + '.nii.gz'):
            os.system('flirt -in ' + infile + ' -ref ' + infile + ' -out ' +
                      outfname + '.nii.gz' + ' -applyisoxfm 1')


#            reg2mni_re(infile=infile, outfile=outfname)

    if modname is 'rest':
        copyfile(infile, outfname + '.nii.gz')
 """


def main():
    #Set subject dirs
    study_name = 'maryland_rao_v1'
    med_hist_csv = '/big_disk/ajoshi/fitbir/maryland_rao/FITBIR Demographics_314/FITBIRdemographics_prospective.csv'
    study_dir = '/big_disk/ajoshi/fitbir/maryland_rao/MM_Prospective_ImagingMR_314'

    # List of subjects that maryland_rao
    tbi_done_list = '/big_disk/ajoshi/fitbir/preproc/maryland_rao_done.txt'
    preproc_dir = '/big_disk/ajoshi/fitbir/preproc'
    subIds = pd.read_csv(med_hist_csv, index_col=1)

    pool = Pool(processes=12)

    for subid in subIds.index:
        print(subid)

        if not isinstance(subid, str):
            continue

        dirlist = glob.glob(study_dir + '*/' + subid + '*_mrtbi_*v1*.nii')
        print(dirlist)
        if len(dirlist) > 0:
            subdir = os.path.join(preproc_dir, study_name, subid)
            os.makedirs(subdir)

            # Normalize all images to 1mm res space.
            imgfiles = dirlist  #glob.glob(img_subdir + '/*.nii.gz')
            print('running proc func')
            print('done proc func')
            pool.starmap(regparfun, zip(repeat(subdir), imgfiles))

    pool.close()
    pool.join()

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

        if os.path.isfile(t1):
            t1img = ni.load_img(t1)
            t1img.to_filename(t1[:-4] + 'r.nii.gz')

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
