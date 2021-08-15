from tqdm import tqdm
import glob
import os

import nilearn.image as ni


def main():
    #Set subject dirs
    #study_dir = '/ImagePTE1/ajoshi/FCD_divya/divya_haleh/CCF.Jun2012/pediatric'
    #study_dir = '/ImagePTE1/ajoshi/FCD_divya/divya_haleh/CCF.Jun2012/adult'
    study_dir = '/ImagePTE1/ajoshi/FCD_divya/divya_haleh/fcd_journal_results'

    outdir = '/ImagePTE1/ajoshi/FCD_divya/preproc'

    subdirs = glob.glob(study_dir + '/*.nii.gz')

    for sub in tqdm(subdirs):
        _, f = os.path.split(sub)
        subname = f.split(os.extsep, 1)[0]        
        suboutdir = os.path.join(outdir, subname)


        if not os.path.isdir(suboutdir):
            os.makedirs(suboutdir)
            t1 = ni.load_img(sub)
            outfile = os.path.join(suboutdir, 'T1.orig.nii.gz')
            t1.to_filename(outfile)

            if not 'bfc.nii' in f:
                cmd1 = '/home/ajoshi/BrainSuite21a/bin/bse -i ' + outfile + ' -o ' + os.path.join(suboutdir, 'T1.bse.nii.gz') + ' --auto --trim'
                cmd2 = '/home/ajoshi/BrainSuite21a/bin/bfc -i ' + os.path.join(suboutdir, 'T1.bse.nii.gz') + ' -o ' + os.path.join(suboutdir, 'T1.nii.gz')
                os.system(cmd1)
                os.system(cmd2)
            else:
                t1.to_filename(os.path.join(suboutdir, 'T1.nii.gz'))


if __name__ == "__main__":
    main()
