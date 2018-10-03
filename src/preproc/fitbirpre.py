import zipfile
import tempfile
from nipype.interfaces.dcm2nii import Dcm2niix
from nipype.interfaces.fsl import first_flirt
import shutil
import glob
import os


def reg2mni(infile, outfile):

    # Use first_flirt command from fsl to coregister to MNI space
    os.system('first_flirt ' + infile + ' ' + outfile)


#    # resample to 1mm isotropic resolution
#    os.system('flirt -in ' + outfile + ' -ref ' + outfile + ' -out ' + outfile + ' -applyisoxfm 1')


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
            converter.inputs.out_filename = '%p_%t_%s'
            print(converter.cmdline)
            #'dcm2niix -b y -z y -5 -x n -t n -m n -o ds005 -s n -v n tmpdir'
            converter.run()
            dirlist = glob.glob(outtmpdir + '/' + '*.gz')
            for file_name in dirlist:
                if (os.path.isfile(file_name)):
                    shutil.copy(file_name, outdir)
