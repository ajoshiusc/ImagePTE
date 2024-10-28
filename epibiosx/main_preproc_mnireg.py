import pandas as pd
import glob
import os
import shutil

# from fitbirpre import zip2nii, reg2mni_re, name2modality
from multiprocessing import Pool
from itertools import product, repeat
import numpy as np
import numbers
from shutil import copyfile
import time


def t1_proc(subid):

    print(subid)
    if not isinstance(subid, str):
        return

    #    study_dir = '/big_disk/ajoshi/fitbir/tracktbi/MM_Prospective_ImagingMR_314'
    # List of subjects that maryland_rao
    preproc_dir = "/deneb_disk/epibiosx_data/preproc"

    t1_img = glob.glob(
        os.path.join(
            "/deneb_disk/ifs/loni/faculty/dduncan/rgarner/shared/epibios/human/data/t1mprage/",
            subid + "*t1mprage.nii.gz",
        )
    )

    if len(t1_img) == 0:
        print("T1 does not exist for " + subid)
        return

    t2_img = glob.glob(
        os.path.join(
            "/deneb_disk/ifs/loni/faculty/dduncan/rgarner/shared/epibios/human/data/t2/",
            subid + "*t2.nii.gz",
        )
    )

    if len(t2_img) == 0:
        print("T2 does not exist for " + subid)
    #    return

    if len(t1_img) > 1:
        print("Multiple T1 images found for " + subid)

    subdir = os.path.join(preproc_dir, subid)
    if not os.path.isdir(subdir):
        os.makedirs(subdir)

    t1 = os.path.join(subdir, "T1.nii.gz")
    t2 = os.path.join(subdir, "T2.nii.gz")

    # resample the T1 image to 1mm isotropic resolution using flirt

    if not os.path.isfile(t1):
        os.system(
            "flirt -in "
            + t1_img[0]
            + " -ref "
            + t1_img[0]
            + " -out "
            + t1
            + " -applyisoxfm 1 -nosearch"
        )

    if not os.path.isfile(t2) and len(t2_img) > 0:
        os.system(
            "flirt -in "
            + t2_img[0]
            + " -ref "
            + t1_img[0]
            + " -out "
            + t2
            + " -applyisoxfm 1 -nosearch"
        )

    t1bse = os.path.join(subdir, "T1.bse.nii.gz")
    t1bfc = os.path.join(subdir, "T1.bfc.nii.gz")
    t1mask = os.path.join(subdir, "T1.mask.nii.gz")

    if not os.path.isfile(t1bse):
        if os.path.isfile(t2):
            os.system("bet " + t1 + " " + t1bse + " -A2 " + t2 + " -m")
        else:
            os.system("bet " + t1 + " " + t1bse + " -m")

        os.system("fslmaths " + t1bse + " -bin " + t1mask)

    # apply bias field correction using bfc from BrainSuite
    if not os.path.isfile(t1bfc):
        os.system(
            "/home/ajoshi/Software/BrainSuite23a/bin/bfc -i " + t1bse + " -o " + t1bfc
        )

    # t12pvc = os.path.join(subdir, 'T12pvc.nii.gz')
    # use fast to segment the bias field corrected image, use both T1 and T2 images
    # os.system('fast -S 2 -o ' +t12pvc +' ' + t1bse + ' ' + t2bse)

    t1pvc = os.path.join(subdir, "T1pvc.nii.gz")
    t1pvc_file = os.path.join(subdir, "T1pvc_pveseg.nii.gz")

    if not os.path.isfile(t2) and not os.path.isfile(t1pvc_file):
        os.system('fast -o ' + t1pvc + ' ' + t1bse)

    # use the bias field corrected image for registration to mni space
    t1mnibse = os.path.join(subdir, "T1mni.bse.nii.gz")
    t1mni = os.path.join(subdir, "T1mni.nii.gz")
    t1mnimat = os.path.join(subdir, "T1mni.mat")
    t1mnimask = os.path.join(subdir, "T1mni.mask.nii.gz")

    if not os.path.isfile(t1mnibse):
        os.system(
            "flirt -in "
            + t1bfc
            + " -ref ${FSLDIR}/data/standard/MNI152_T1_1mm_brain.nii.gz -out "
            + t1mnibse
            + " -omat "
            + t1mnimat
            + " -dof 6"
        )  # -cost normmi')
        os.system("fslmaths " + t1mnibse + " -bin " + t1mnimask)

    # apply the transform to t1 image
    if not os.path.isfile(t1mni):
        os.system(
            "flirt -in "
            + t1
            + " -ref ${FSLDIR}/data/standard/MNI152_T1_1mm_brain.nii.gz -out "
            + t1mni
            + " -applyxfm -init "
            + t1mnimat
        )


def t2_proc(subid):

    print(subid)
    if not isinstance(subid, str):
        return

    #    study_dir = '/big_disk/ajoshi/fitbir/tracktbi/MM_Prospective_ImagingMR_314'
    # List of subjects that maryland_rao
    preproc_dir = "/deneb_disk/epibiosx_data/preproc"

    t1_img = glob.glob(
        os.path.join(
            "/deneb_disk/ifs/loni/faculty/dduncan/rgarner/shared/epibios/human/data/t1mprage/",
            subid + "*t1mprage.nii.gz",
        )
    )

    if len(t1_img) == 0:
        print("T1 does not exist for " + subid)
        return

    if len(t1_img) > 1:
        print("Multiple T1 images found for " + subid)
        # return

    t2_img = glob.glob(
        os.path.join(
            "/deneb_disk/ifs/loni/faculty/dduncan/rgarner/shared/epibios/human/data/t2/",
            subid + "*t2.nii.gz",
        )
    )

    if len(t2_img) == 0:
        print("T2 does not exist for " + subid)
        return

    if len(t2_img) > 1:
        print("Multiple T2 images found for " + subid)
        # return

    subdir = os.path.join(preproc_dir, subid)
    if not os.path.isdir(subdir):
        os.makedirs(subdir)

    t2 = os.path.join(subdir, "T2.nii.gz")

    # resample the T2 image to 1mm isotropic resolution using flirt
    if not os.path.isfile(t2):
        os.system(
            "flirt -in "
            + t2_img[0]
            + " -ref "
            + t1_img[0]
            + " -out "
            + t2
            + " -applyisoxfm 1 -nosearch"
        )

    # os.system('cp ' + t1_img[0] + ' ' + t1)
    # os.system('cp ' + t2_img[0] + ' ' + t2)

    t1bse = os.path.join(subdir, "T1.bse.nii.gz")
    t1mask = os.path.join(subdir, "T1.mask.nii.gz")

    t2bse = os.path.join(subdir, "T2.bse.nii.gz")

    # apply mask to T2 image
    if not os.path.isfile(t2bse):
        if not os.path.isfile(t2bse):
            os.system("fslmaths " + t2 + " -mas " + t1mask + " " + t2bse)

    t12pvc = os.path.join(subdir, "T12pvc.nii.gz")
    t12pvc_file = os.path.join(subdir, "T12pvc_pveseg.nii.gz")

    # use fast to segment the bias field corrected image, use both T1 and T2 images
    if not os.path.isfile(t12pvc_file):
        os.system("fast -S 2 -o " + t12pvc + " " + t1bse + " " + t2bse)

    # use the bias field corrected image for registration to mni space
    t1mnimat = os.path.join(subdir, "T1mni.mat")
    t1mnimask = os.path.join(subdir, "T1mni.mask.nii.gz")

    t2mni = os.path.join(subdir, "T2mni.nii.gz")
    t2mnibse = os.path.join(subdir, "T2mni.bse.nii.gz")

    # apply the transform to t2 image
    if not os.path.isfile(t2mni):
        os.system(
            "flirt -in "
            + t2
            + " -ref ${FSLDIR}/data/standard/MNI152_T1_1mm_brain.nii.gz -out "
            + t2mni
            + " -applyxfm -init "
            + t1mnimat
        )
        os.system("fslmaths " + t2mni + " -mas " + t1mnimask + " " + t2mnibse)


def process_sub(subid):
    ''' Process a single subject '''
    t1_proc(subid)
    t2_proc(subid)



def main():

    pte_xlsx = os.path.join("spreadsheets/short PTE.xlsx")

    # read the spreadsheet with the PTE data, and extract the subject IDs and PTE status
    pte_data = pd.read_excel(pte_xlsx)
    pte_data = pte_data[["Study ID", "PTE", "LS"]]
    pte_data = pte_data.dropna(subset=["Study ID"])

    subIds = pte_data["Study ID"].values

    pool = Pool(processes=8)
    # done_list = '/big_disk/ajoshi/fitbir/preproc/tracktbi_pilot_done.txt'
    doneIds = []

    # with open(tbi_done_list) as f:
    #    tbidoneIds = f.readlines()

    # tbidoneIds = list(map(lambda x: x.strip(), tbidoneIds))

    print(subIds)
    subsnotdone = [x for x in subIds if x not in doneIds]

    '''
    for sub in subsnotdone:
        t1_proc(sub)
        t2_proc(sub)
    '''

    print('++++++++++++++')
    pool.map(process_sub, subsnotdone)

    pool.close()
    pool.join()
    


if __name__ == "__main__":
    main()
