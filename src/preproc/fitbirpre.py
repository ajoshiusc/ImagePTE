import zipfile
import tempfile
from nipype.interfaces.dcm2nii import Dcm2niix
#from nipype.interfaces.fsl import first_flirt
import shutil
import glob
import os


def name2modality(fname=''):
    fname = fname.lower()

    if ('flair' in fname) and ('t1_' not in fname):
        return 'FLAIR'

    if ('flair' in fname) and ('t1_' in fname):
        return 'T1_FLAIR'

    if ('mprage' in fname) or ('adni' in fname) or ('t1' in fname):
        return 'T1'

    if ('t2_' in fname) and ('fluid' not in fname) and ('hemo' not in fname):
        return 'T2'

    if ('fse_' in fname):
        return 'fse'

    if ('_rest' in fname):
        return 'rest'

    if ('_swi' in fname):
        return 'SWI'



def reg2mni(infile, outfile):

    # Use first_flirt command from fsl to coregister to MNI space
    os.system('./first_flirt_rigid ' + infile + ' ' + outfile)


def reg2mni_re(infile, outfile):

    # Use first_flirt command from fsl to coregister to MNI space
    os.system('./first_flirt_rigid_reorient_mi ' + infile + ' ' + outfile)

#    # resample to 1mm isotropic resolution
    os.system('flirt -in ' + infile + ' -ref ' + infile + ' -out ' + outfile + ' -applyisoxfm 1')


def zip2nii(zipfname, outdir):

    with tempfile.TemporaryDirectory() as tmpdir:
        with tempfile.TemporaryDirectory() as outtmpdir:
            shutil.copy(zipfname, tmpdir)
            _, fname = os.path.split(zipfname)
            fname = os.path.join(tmpdir, fname)

            zip_ref = zipfile.ZipFile(fname, 'r')
            zip_ref.extractall(tmpdir)
            zip_ref.close()
            os.remove(fname)

            converter = Dcm2niix()
            converter.inputs.source_dir = tmpdir
            converter.inputs.compression = 5
            converter.inputs.output_dir = outtmpdir
            converter.inputs.out_filename = '%p_%t_%s'
            print(converter.cmdline)
            #'dcm2niix -b y -z y -5 -x n -t n -m n -o ds005 -s n -v n tmpdir'
            converter.run()
            dirlist = glob.glob(outtmpdir + '/' + '*.gz')
            for file_name in dirlist:
                if (os.path.isfile(file_name)):
                    shutil.copy(file_name, outdir)
