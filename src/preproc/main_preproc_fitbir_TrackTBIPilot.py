#||AUM||
#||Shree Ganeshaya Namaha||

import pandas as pd
import glob
import os.path


def main():

    print('hi')

    study_name = 'tracktbi_pilot'
    med_hist_csv = '/big_disk/ajoshi/fitbir/tracktbi_pilot/Baseline Med History_246/TrackTBI_MedicalHx.csv'
    study_dir = '/big_disk/ajoshi/fitbir/tracktbi_pilot/TRACK TBI Pilot - MR data - BR site_246'
    preproc_dir = '/big_disk/ajoshi/fitbir/preproc'

    dirlst = glob.glob(study_dir + '/*.zip')

   # print(med_hist_csv)
    subIds = pd.read_csv(med_hist_csv, index_col=1)
   # print(subIds)
    ''' If fMRI data exists for some subjects, then store their cognitive scores '''
    for subid in subIds.index:
        print(subid)
        os.path.join(preproc_dir, study_name, subid)
        dirlist = glob.glob(study_dir + '/' + subid + '*.zip')
        print(dirlist)
        if len(dirlist) > 0:
            print('hi' + subid)


if __name__ == "__main__":
    main()
