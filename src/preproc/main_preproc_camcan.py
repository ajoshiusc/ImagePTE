'''Preprocessing of CamCan data'''
import os
from multiprocessing import Pool
import glob

def regparfun(subid):
    ''' the function that does actual preprocessing '''

    subid = str(subid)
    print(subid)
    if not isinstance(subid, str):
        return 1

    #    study_dir = '/big_disk/ajoshi/fitbir/tracktbi/MM_Prospective_ImagingMR_314'
    # List of subjects that maryland_rao
    preproc_dir = '/ImagePTE1/ajoshi/data/camcan/cc700/mri/pipeline/release004/BIDS_20190411/anat'
    subdir = os.path.join(preproc_dir, subid)

    outdir = '/ImagePTE1/ajoshi/data/camcan_preproc/'
    outdir = os.path.join(outdir, subid)

    if not os.path.isdir(outdir):
        os.mkdir(outdir)

    t1_img = os.path.join(subdir, 'anat/' + subid + '_T1w.nii.gz')
    t2_img = os.path.join(subdir, 'anat/' + subid + '_T2w.nii.gz')
    t1bse = os.path.join(outdir, 'T1.bse.nii.gz')
    t1mni = os.path.join(outdir, 'T1mni.nii.gz')
    t1mnimask = os.path.join(outdir, 'T1mni.mask.nii.gz')
    t1mnimat = os.path.join(outdir, 'T1mni.mat')
    t2mni = os.path.join(outdir, 'T2mni.nii.gz')

    if not os.path.isfile(t1_img):
        print('T1 does not exist for ' + subid + t1_img)
        return 1

    os.system('bet ' + t1_img + ' ' + t1bse + ' -f .3')

    # register T1 image to MNI space

    os.system(
        'flirt -in ' + t1bse + ' -out ' + t1mni +
        ' -ref ${FSLDIR}/data/standard/MNI152_T1_1mm_brain.nii.gz -omat ' +
        t1mnimat + ' -dof 6 -cost corratio -searchrx -30 30 -searchry -30 30 -searchrz -30 30')

    print(subid)

    # Create mask
    os.system('fslmaths ' + t1mni + ' -bin ' + t1mnimask)
    # Apply the same transform (T1->MNI)

    if os.path.isfile(t2_img):
        # Apply the same transform (T1->MNI) to registered T2
        os.system(
            'flirt -in ' + t2_img +
            ' -ref ${FSLDIR}/data/standard/MNI152_T1_1mm_brain.nii.gz -out ' +
            t2mni + ' -applyxfm -init ' + t1mnimat)
        os.system('fslmaths ' + t2mni + ' -mul ' + t1mnimask + ' ' + t2mni)

    return 0


def main():
    '''Main function call'''

    # Set subject dirs
    sub_dir = '/ImagePTE1/ajoshi/data/camcan/cc700/mri/pipeline/release004/BIDS_20190411/anat'

    sub_list = [os.path.basename(x) for x in glob.glob(sub_dir+'/s*')]




    pool = Pool(processes=4)

    #for j in range(10):
    #    regparfun(sub_list[j])

    print('++++++++++++++')
    pool.map(regparfun, sub_list)
    print('++++SUBMITTED++++++')
    
    pool.close()
    pool.join()


if __name__ == "__main__":
    main()
