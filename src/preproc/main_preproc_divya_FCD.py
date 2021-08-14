#||AUM||
#||Shree Ganeshaya Namaha||

import glob
import os

import nilearn.image as ni


def main():
    #Set subject dirs
    study_dir = '/ImagePTE1/ajoshi/FCD_divya/divya_haleh/CCF.Jun2012/pediatric'
    outdir = '/ImagePTE1/ajoshi/FCD_divya/preproc'

    subdirs = glob.glob(study_dir + '/*.nii.gz')

    for sub in subdirs:


        _, f = os.path.split(sub)
        subname = f.split(os.extsep, 1)[0]        
        suboutdir = os.path.join(outdir, subname)

        if not os.path.isdir(suboutdir):
            os.makedirs(suboutdir)
            t1 = ni.load_img(sub)
            outfile = os.path.join(suboutdir, 'T1.nii.gz')
            t1.to_filename(outfile)

if __name__ == "__main__":
    main()
