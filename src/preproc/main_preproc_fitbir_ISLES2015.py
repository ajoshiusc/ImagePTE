#||AUM||
#||Shree Ganeshaya Namaha||

import glob
import os

import nilearn.image as ni


def main():
    #Set subject dirs
    study_dir = '/big_disk/ajoshi/ISLES2015/orig/Training'
    outdir = '/big_disk/ajoshi/ISLES2015/preproc/Training'

    for subno in range(1, 29):

        subdir = os.path.join(study_dir, str(subno))
        subdirs = glob.glob(subdir + '/V*')
        suboutdir = os.path.join(outdir, str(subno))

        if not os.path.isdir(suboutdir):
            os.makedirs(suboutdir)

        for imgdir in subdirs:
            if '_T1' in imgdir:
                t1file = glob.glob(imgdir + '/*.nii')
                t1 = ni.load_img(t1file)
                outfile = os.path.join(outdir, str(subno), 'T1.nii.gz')
                t1.to_filename(outfile)

            if '_T2' in imgdir:
                t1file = glob.glob(imgdir + '/*.nii')
                t1 = ni.load_img(t1file)
                outfile = os.path.join(outdir, str(subno), 'T2.nii.gz')
                t1.to_filename(outfile)

            if '_Flair' in imgdir:
                t1file = glob.glob(imgdir + '/*.nii')
                t1 = ni.load_img(t1file)
                outfile = os.path.join(outdir, str(subno), 'FLAIR.nii.gz')
                t1.to_filename(outfile)

            if '.OT.' in imgdir:
                t1file = glob.glob(imgdir + '/*.nii')
                t1 = ni.load_img(t1file)
                outfile = os.path.join(outdir, str(subno),
                                       'SEGMENTATION.nii.gz')
                t1.to_filename(outfile)


if __name__ == "__main__":
    main()
