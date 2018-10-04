#||AUM||
#||Shree Ganeshaya Namaha||

import pandas as pd
import glob
import os
import shutil
from fitbirpre import zip2nii, reg2mni, name2modality


def main():
    #Set subject dirs
    study_name = 'tracktbi_pilot'
    med_hist_csv = '/big_disk/ajoshi/fitbir/tracktbi_pilot/Baseline Med History_246/TrackTBI_MedicalHx.csv'
    study_dir = '/big_disk/ajoshi/fitbir/tracktbi_pilot/TRACK TBI Pilot - MR data -'

    preproc_dir = '/big_disk/ajoshi/fitbir/preproc'
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
            img_subdir = os.path.join(subdir, 'orig')

            # Create subject directory
            if not os.path.exists(img_subdir):
                os.makedirs(img_subdir)
            # copy all zip files to the subject directory
            for file_name in dirlist:
                if (os.path.isfile(file_name)):
                    zip2nii(file_name, img_subdir)

            # Normalize all images to standard MNI space.
            imgfiles = glob.glob(img_subdir + '/*.nii.gz')

            for infile in imgfiles:
                outfname = os.path.join(subdir, name2modality(infile))
                if outfname is not None:
                    reg2mni(infile=infile, outfile=outfname)


if __name__ == "__main__":
    main()
