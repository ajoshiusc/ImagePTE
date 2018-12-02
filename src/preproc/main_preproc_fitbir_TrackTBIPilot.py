#||AUM||
#||Shree Ganeshaya Namaha||

import pandas as pd
import glob
import os
import shutil
from fitbirpre import zip2nii, reg2mni, name2modality
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

    copyfile(infile, outfname + '.nii.gz')


def main():
    #Set subject dirs
    study_name = 'tracktbi_pilot'
    med_hist_csv = '/big_disk/ajoshi/fitbir/tracktbi_pilot/Baseline Med History_246/TrackTBI_MedicalHx.csv'
    study_dir = '/big_disk/ajoshi/fitbir/tracktbi_pilot/TRACK TBI Pilot - MR data -'

    # List of subjects that are done
    #    tbi_done_list = '/big_disk/ajoshi/fitbir/preproc/tracktbi_done.txt'
    preproc_dir = '/big_disk/ajoshi/fitbir/preproc'
    subIds = pd.read_csv(med_hist_csv, index_col=1)

    # This contains a list of TBI subjects that are done correctly
    #    with open(tbi_done_list) as f:
    #        tbidoneIds = f.readlines()

    # Get the list of subjects that are correctly registered
    #   tbidoneIds = [l.strip('\n\r') for l in tbidoneIds]
    ''' If fMRI data exists for some subjects, then store their cognitive scores '''
    pool = Pool(processes=1)

    for subid in subIds.index:
        print(subid)

        if isinstance(subid, numbers.Number):
            continue

#      if any(subid in s for s in tbidoneIds):
#          print(subid + ' is already done')
#          continue

        if not isinstance(subid, str):
            continue

        os.path.join(preproc_dir, study_name, subid)
        dirlist = glob.glob(study_dir + '*/' + subid + '*.zip')
        print(dirlist)
        if len(dirlist) > 100:
            subdir = os.path.join(preproc_dir, study_name, subid)
            print('hi' + subdir)
            img_subdir = os.path.join(subdir, 'orig')

            # Create subject directory
            if not os.path.exists(img_subdir):
                os.makedirs(img_subdir)
            # copy all zip files to the subject directory
            for file_name in dirlist:
                if (os.path.isfile(file_name)) and (os.path.isdir(img_subdir)):
                    zip2nii(file_name, img_subdir)

            # Normalize all images to standard MNI space.
            imgfiles = glob.glob(img_subdir + '/*.nii.gz')
            print('running proc func')
            #            regparfun(subdir,imgfiles[0])
            print('done proc func')
            pool.starmap(regparfun, zip(repeat(subdir), imgfiles))

    pool.close()
    pool.join()

    for subid in subIds.index:

        print(subid)

        if not isinstance(subid, str):
            continue

        dirname = os.path.join(preproc_dir, study_name, subid)

        t1 = os.path.join(dirname, 'T1.nii.gz')
        t2 = os.path.join(dirname, 'T2.nii.gz')
        flair = os.path.join(dirname, 'FLAIR.nii.gz')
        fse = os.path.join(dirname, 'FSE.nii.gz')

        if os.path.isfile(t1):
            t1r = ni.load_img(t1)
            t1r.to_filename(t1[:-7] + 'r.nii.gz')
        else:
            print('T1 does not exist')

        if os.path.isfile(t1) and os.path.isfile(t2):
            t2r = ni.resample_to_img(t2, t1r)
            t2r.to_filename(t2[:-7] + 'r.nii.gz')
        else:
            print('T2 does not exist')

        if os.path.isfile(t1) and os.path.isfile(flair):
            flairr = ni.resample_to_img(flair, t1r)
            flairr.to_filename(flair[:-7] + 'r.nii.gz')
        else:
            print('FLAIR does not exist')

        if os.path.isfile(t1) and os.path.isfile(fse):
            fser = ni.resample_to_img(fse, t1r)
            fser.to_filename(fse[:-7] + 'r.nii.gz')
        else:
            print('FSE does not exist')

"""            for infile in imgfiles:
                modname = name2modality(infile)
                if modname is not None:
                    outfname = os.path.join(subdir, modname)
                    if not os.path.isfile(outfname + '.nii.gz'):
                        reg2mni(infile=infile, outfile=outfname)
"""

if __name__ == "__main__":
    main()
