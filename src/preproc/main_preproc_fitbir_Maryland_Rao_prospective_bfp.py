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

bfp_exe = '/ImagePTE1/ajoshi/bfp_ver2p26/bfp.sh'
bfp_config = '/ImagePTE1/ajoshi/fitbir/preproc/maryland_config_bfp_preproc.ini'


def regparfun(subid):

    #    study_dir = '/ImagePTE1/ajoshi/fitbir/maryland_rao/MM_Prospective_ImagingMR_314'
    # List of subjects that maryland_rao
    preproc_dir = '/ImagePTE1/ajoshi/fitbir/preproc'
    study_name = 'maryland_rao_v1'

    subdir = os.path.join(preproc_dir, study_name, subid)
    bfp_subdir = os.path.join(subdir, 'BFP')

    # If BrainSuite directory does not exist then create it
    if not os.path.isdir(bfp_subdir):
        os.makedirs(bfp_subdir)

    t1 = os.path.join(subdir, 'T1r.nii.gz')
    fmri = os.path.join(subdir, 'rest.nii')

    # If the file does not exist, then return
    if not os.path.isfile(t1) or not os.path.isfile(fmri):
        return

    t1gz = os.path.join(bfp_subdir, 'T1r.bse.nii.gz')
    fmrigz = os.path.join(bfp_subdir, 'rest.nii.gz')

    # copy and zip the t1 and fmri files
    if not os.path.isfile(t1gz) or not os.path.isfile(fmrigz):

        # copy files to BrainSuite directory for processing
        copy(t1, bfp_subdir)
        copy(fmri, bfp_subdir)

        # zip the files
        #os.system('gzip -f ' + os.path.join(bfp_subdir, 'T1r.nii'))
        os.system('cp ' + os.path.join(bfp_subdir, 'T1r.nii.gz') + ' ' + t1gz)

        os.system('gzip -f ' + os.path.join(bfp_subdir, 'rest.nii'))

    os.system(bfp_exe + ' ' + bfp_config + ' ' + t1gz + ' ' + fmrigz + ' ' +
              bfp_subdir + ' ' + subid + ' ' + 'rest' + ' ' + '0')  # fmri

    print(subid)


def main():
    #Set subject dirs
    #    med_hist_csv = '/ImagePTE1/ajoshi/fitbir/maryland_rao/FITBIR Demographics_314/FITBIRdemographics_prospective.csv'
    #    subIds = pd.read_csv(med_hist_csv, index_col=1)

#    pool = Pool(processes=6)

    sub_list = '/ImagePTE1/ajoshi/fitbir/preproc/maryland_rao_v1_nonepilepsy_imgs.txt'

    with open(sub_list) as f:
        subIds = f.readlines()

    subIds = list(map(lambda x: x.strip(), subIds))
    print(subIds.index)

    #regparfun(subIds[1])

    print('There are %d subjects\n'% len(subIds))

    for id in subIds:
        #regparfun(id)
        fname = '/ImagePTE1/ajoshi/fitbir/preproc/maryland_rao_v1/'+id+'/BFP/'+id+'/func/'+id+'_rest_bold.32k.GOrd.mat'

        if not os.path.isfile(fname):
            print(' check : %s' % fname)
        
        

#    pool.map(regparfun, subIds)

#    pool.close()
#    pool.join()


if __name__ == "__main__":
    main()
