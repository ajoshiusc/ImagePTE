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
    outdir = '/home/ajoshi/for_paul/nonPTE'

    t1 = os.path.join(subdir, 'T1.nii')
    outfile = os.path.join(outdir,subid+'.nii')
    # copy files to BrainSuite directory for processing
    copy(t1, outfile)
    os.system('gzip -f ' + outfile)
 
    print(subid)


def main():

    sub_list = '/ImagePTE1/ajoshi/fitbir/preproc/maryland_rao_v1_nonepilepsy_imgs_37.txt'

    with open(sub_list) as f:
        subIds = f.readlines()

    subIds = list(map(lambda x: x.strip(), subIds))
    print(subIds.index)

    print('There are %d subjects\n'% len(subIds))

    for id in subIds:
        regparfun(id)

if __name__ == "__main__":
    main()
