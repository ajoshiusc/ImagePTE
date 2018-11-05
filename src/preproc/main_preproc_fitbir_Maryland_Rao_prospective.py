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

def regparfun(subdir, infile):
    modname = name2modality(infile)
    if (modname is not None) and (modname is not 'rest'):
        outfname = os.path.join(subdir, modname)
        if not os.path.isfile(outfname + '.nii.gz'):
            reg2mni(infile=infile, outfile=outfname)
        
    if modname is 'rest':
        copyfile(infile, outfname)


def main():
    #Set subject dirs
    study_name = 'maryland_rao_v1'
    med_hist_csv = '/big_disk/ajoshi/fitbir/maryland_rao/FITBIR Demographics_314/FITBIRdemographics_prospective.csv'
    study_dir = '/big_disk/ajoshi/fitbir/maryland_rao/MM_Prospective_ImagingMR_314'

    # List of subjects that maryland_rao
    tbi_done_list = '/big_disk/ajoshi/fitbir/preproc/maryland_rao_done.txt'
    preproc_dir = '/big_disk/ajoshi/fitbir/preproc'
    subIds = pd.read_csv(med_hist_csv, index_col=1)

    # This contains a list of TBI subjects that are done correctly
#    with open(tbi_done_list) as f:
#        tbidoneIds = f.readlines()

    # Get the list of subjects that are correctly registered
   # tbidoneIds = [l.strip('\n\r') for l in tbidoneIds]
    ''' If fMRI data exists for some subjects, then store their cognitive scores '''
    pool = Pool(processes=4)

    for subid in subIds.index:
        print(subid)
#        continue

 #       if isinstance(subid, numbers.Number):
 #           continue


#        if any(subid in s for s in tbidoneIds):
#            print(subid + ' is already done')
#            continue

        if not isinstance(subid, str):
            continue

        #os.path.join(preproc_dir, study_name, subid)
        dirlist = glob.glob(study_dir + '*/' + subid + '*v1*.nii')
        print(dirlist)
        if len(dirlist) > 0:
            subdir = os.path.join(preproc_dir, study_name, subid)
            print('hi' + subdir)
            img_subdir = os.path.join(subdir, 'orig')

            # Create subject directory
            if not os.path.exists(img_subdir):
                os.makedirs(img_subdir)
            # copy all nii files to the subject directory
 #           for file_name in dirlist:
#                if (os.path.isfile(file_name)) and (os.path.isdir(img_subdir)):
#                    copy_nii(file_name, img_subdir)

            # Normalize all images to standard MNI space.
            imgfiles = dirlist #glob.glob(img_subdir + '/*.nii.gz')
            print('running proc func')
#            regparfun(subdir,imgfiles[0])
            print('done proc func')
            pool.starmap(regparfun, zip(repeat(subdir), imgfiles))

    pool.close()
    pool.join()


"""            for infile in imgfiles:
                modname = name2modality(infile)
                if modname is not None:
                    outfname = os.path.join(subdir, modname)
                    if not os.path.isfile(outfname + '.nii.gz'):
                        reg2mni(infile=infile, outfile=outfname)
"""

if __name__ == "__main__":
    main()
