import pandas as pd
import glob
import os
import shutil
#from fitbirpre import zip2nii, reg2mni_re, name2modality
from multiprocessing import Pool
from itertools import product, repeat
import numpy as np
import numbers
from shutil import copyfile
import time


def regparfun(subid):

    print(subid)
    if not isinstance(subid, str):
        return

#    study_dir = '/big_disk/ajoshi/fitbir/tracktbi/MM_Prospective_ImagingMR_314'
    # List of subjects that maryland_rao
    preproc_dir = '/deneb_disk/epibiosx_data/preproc'

    subdir = os.path.join(preproc_dir, subid)
    if not os.path.isdir(subdir):
        os.makedirs(subdir)

    t1_img = glob.glob(os.path.join('/deneb_disk/ifs/loni/faculty/dduncan/rgarner/shared/epibios/human/data/t1mprage/', subid + '*t1mprage.nii.gz'))
    
    if len(t1_img) == 0:
        print('T1 does not exist for ' + subid)
        return
    
    if len(t1_img) > 1:
        print('Multiple T1 images found for ' + subid)
        return

    t2_img = glob.glob(os.path.join('/deneb_disk/ifs/loni/faculty/dduncan/rgarner/shared/epibios/human/data/t2/', subid + '*t2.nii.gz'))
    
    if len(t2_img) == 0:
        print('T2 does not exist for ' + subid)
        return
    
    if len(t2_img) > 1:
        print('Multiple T2 images found for ' + subid)
        return


    t1 = os.path.join(subdir, 'T1.nii.gz')
    t2 = os.path.join(subdir, 'T2.nii.gz')
    os.system('cp ' + t1_img[0] + ' ' + t1)
    os.system('cp ' + t2_img[0] + ' ' + t2)

    t1bse = os.path.join(subdir, 'T1.bse.nii.gz')
    t1bfc = os.path.join(subdir, 'T1.bfc.nii.gz')

    if not os.path.isfile(t1bse):
        os.system('bet ' + t1 + ' ' + t1bse + ' -f .3 -A2 ' + t2)

    # apply bias field correction using bfc from BrainSuite
    if not os.path.isfile(t1bfc):
        os.system('/home/ajoshi/Software/BrainSuite23a/bin/bfc -i ' + t1bse + ' -o ' + t1bfc)


    # use the bias field corrected image for registration to mni space
    t1mnir = os.path.join(subdir, 'T1mni.nii.gz')
    t1mnimat = os.path.join(subdir, 'T1mni.mat')
    t1mnimask = os.path.join(subdir, 'T1mni.mask.nii.gz')


    if not os.path.isfile(t1mnir):
        os.system('flirt -in ' + t1bfc + ' -ref ${FSLDIR}/data/standard/MNI152_T1_1mm_brain.nii.gz -out ' + t1mnir + ' -omat ' + t1mnimat + ' -dof 6 -cost normmi')
        os.system('fslmaths ' + t1mnir + ' -bin ' + t1mnimask)





    
    if not os.path.isfile(t1bfc):
        os.system('fast -t 1 -n 3 -H 0.1 -I 4 -l 20.0 -o ' + t1bfc + ' ' + t1bse)

    '''t1mnir = os.path.join(subdir, 'T1mni.nii.gz')
    t1mnimask = os.path.join(subdir, 'T1mni.mask.nii.gz')
    #    t1bfc = os.path.join(subdir, 'T1.bfc.nii.gz')

    os.system('bet ' + t1_img + ' ' + t1bse + ' -f .3')
    os.system('fast -t 1 -n 3 -H 0.1 -I 4 -l 20.0 -o ' + t1bfc + ' ' + t1bse)
    os.system('flirt -in ' + t1bfc + ' -ref ${FSLDIR}/data/standard/MNI152_T1_1mm_brain.nii.gz -out ' + t1mnir + ' -omat ' + t1mnimat + ' -dof 6 -cost normmi')
    os.system('fslmaths ' + t1mnir + ' -bin ' + t1mnimask)

    t1mnimat = os.path.join(subdir, 'T1mni.mat')
    print(subid)

    if not os.path.isfile(t1):
        print('T1 does not exist for ' + subid + t1)
        return

    # register T1 image to MNI space

    os.system('flirt -in ' + t1 + ' -out ' + t1mnir +
              ' -ref ${FSLDIR}/data/standard/MNI152_T1_1mm_brain.nii.gz -omat '
              + t1mnimat + ' -dof 6 -cost normmi')

    
    t1mni = os.path.join(subdir, 'T1mni.nii.gz')

    if os.path.isfile(t1):
        # Create mask
        os.system('fslmaths ' + t1mni + ' -bin ' + t1mnimask)
        # Apply the same transform (T1->MNI)

        t2 = os.path.join(subdir, 'T2r.nii.gz')
        t2mni = os.path.join(subdir, 'T2mni.nii.gz')
    if os.path.isfile(t2):
        # Apply the same transform (T1->MNI) to registered T2
        os.system('flirt -in ' + t2 + ' -ref ' + t1mnir + ' -out ' + t2mni +
                  ' -applyxfm -init ' + t1mnimat)
        os.system('fslmaths ' + t2mni + ' -mul ' + t1mnimask + ' ' + t2mni)

    flair = os.path.join(subdir, 'FLAIRr.nii.gz')
    flairmni = os.path.join(subdir, 'FLAIRmni.nii.gz')
    if os.path.isfile(flair):
        # Apply the same transform (T1->MNI) to registered FLAIR
        os.system('flirt -in ' + flair + ' -ref ' + t1mnir + ' -out ' +
                  flairmni + ' -applyxfm -init ' + t1mnimat)
        os.system('fslmaths ' + flairmni + ' -mul ' + t1mnimask + ' ' +
                  flairmni)

    '''


def main():


    pte_xlsx = os.path.join('spreadsheets/short PTE.xlsx')


    # read the spreadsheet with the PTE data, and extract the subject IDs and PTE status
    pte_data = pd.read_excel(pte_xlsx)
    pte_data = pte_data[['Study ID', 'PTE', 'LS']]
    pte_data = pte_data.dropna(subset=['Study ID'])

    subIds = pte_data['Study ID'].values

    pool = Pool(processes=12)
    #done_list = '/big_disk/ajoshi/fitbir/preproc/tracktbi_pilot_done.txt'
    doneIds=[]

    #with open(tbi_done_list) as f:
    #    tbidoneIds = f.readlines()

    #tbidoneIds = list(map(lambda x: x.strip(), tbidoneIds))

    print(subIds)
    subsnotdone = [x for x in subIds if x not in doneIds]
    
    for sub in subsnotdone:
        regparfun(sub)
    
    '''print('++++++++++++++')
    pool.map(regparfun, subsnotdone)

    pool.close()
    pool.join()
    '''


if __name__ == "__main__":
    main()
