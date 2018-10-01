#||AUM||
#||Shree Ganeshaya Namaha||

import pandas as pd
import glob
import os
import shutil


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
            subdir = os.path.join(preproc_dir, study_name, subid)
            print('hi' + subdir)

            # Create subject directory
            if not os.path.exists(subdir):
                os.makedirs(subdir)
            # copy all zip files to the subject directory
            for file_name in dirlist:
                if (os.path.isfile(file_name)):
                    zip2nii(file_name, subdir)
#                    shutil.copy(file_name, subdir)


if __name__ == "__main__":
    main()

import zipfile
import tempfile
from nipype.interfaces.dcm2nii import Dcm2niix

def zip2nii(zipfname, outdir):

    with tempfile.TemporaryDirectory() as tmpdir:
        with tempfile.TemporaryDirectory() as outtmpdir:
            shutil.copy(zipfname, tmpdir)
            pth, fname = os.path.split(zipfname)
            fname = os.path.join(tmpdir, fname)

            zip_ref = zipfile.ZipFile(fname, 'r')
            zip_ref.extractall(tmpdir)
            zip_ref.close()
            os.remove(fname)
            
            converter = Dcm2niix()
            converter.inputs.source_dir = tmpdir
            converter.inputs.compression = 5
            converter.inputs.output_dir = outtmpdir
            print(converter.cmdline)
    #'dcm2niix -b y -z y -5 -x n -t n -m n -o ds005 -s n -v n tmpdir'
            converter.run() 
            dirlist = glob.glob(outtmpdir + '/' + '*.gz')
            for file_name in dirlist:
                if (os.path.isfile(file_name)):
                    shutil.copy(file_name, outdir)










