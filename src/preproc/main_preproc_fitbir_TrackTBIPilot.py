#||AUM||
#||Shree Ganeshaya Namaha||

import pandas as pd
import glob
import os
import shutil
from fitbirpre import zip2nii, reg2mni


def main():

    print('hi')

    study_name = 'tracktbi_pilot'
    med_hist_csv = '/big_disk/ajoshi/fitbir/tracktbi_pilot/Baseline Med History_246/TrackTBI_MedicalHx.csv'
    study_dir = '/big_disk/ajoshi/fitbir/tracktbi_pilot/TRACK TBI Pilot - MR data -'  # BR site_246'
    #    study_dir2 = '/big_disk/ajoshi/fitbir/tracktbi_pilot/TRACK TBI Pilot - MR data - PI site_246'
    #    study_dir3 = '/big_disk/ajoshi/fitbir/tracktbi_pilot/TRACK TBI Pilot - MR data - SF site__246'

    preproc_dir = '/big_disk/ajoshi/fitbir/preproc'

    dirlst = glob.glob(study_dir + '/*.zip')

    # print(med_hist_csv)
    subIds = pd.read_csv(med_hist_csv, index_col=1)
    # print(subIds)
    ''' If fMRI data exists for some subjects, then store their cognitive scores '''
    for subid in subIds.index:
        print(subid)
        if not isinstance(subid, str):
            continue

        os.path.join(preproc_dir, study_name, subid)
        dirlist = glob.glob(study_dir + '*/' + subid + '*.zip')
        print(dirlist)
        if len(dirlist) > 0:
            subdir = os.path.join(preproc_dir, study_name, subid)
            print('hi' + subdir)

            # Create subject directory
            if not os.path.exists(subdir):
                os.makedirs(subdir)
            # copy all zip files to the subject directory
            for file_name in dirlist:
                if (os.path.isfile(file_name)):
                    zip2nii(file_name, subdir)


# Normalize all images to standard MNI space.
            imgfiles = glob.glob(subdir + '/*.nii.gz')

            for infile in imgfiles:
                reg2mni(
                    infile=infile,
                    outfile=os.path.splitext(infile)[0] + '_mni')

if __name__ == "__main__":
    main()
