import os
from shutil import copyfile, make_archive, rmtree
from tqdm import tqdm

studydir = '/ImagePTE1/ajoshi/fitbir/preproc/maryland_rao_v1'

epi_txt = '/ImagePTE1/ajoshi/fitbir/preproc/maryland_rao_v1_epilepsy_imgs.txt'
nonepi_txt = '/ImagePTE1/ajoshi/fitbir/preproc/maryland_rao_v1_nonepilepsy_imgs_37.txt'
cut_coords = (60 - 182 / 2, 122 - 212 / 2, 90 - 182 / 2)


def main():

    with open(epi_txt) as f:
        epiIds = f.readlines()

    with open(nonepi_txt) as f:
        nonepiIds = f.readlines()

    epi_subids = list(map(lambda x: x.strip(), epiIds))
    nonepi_subids = list(map(lambda x: x.strip(), nonepiIds))

    out_dir = '/ImagePTE1/ajoshi/fitbir_upload_anatomical_analysis'
    rmtree(out_dir)
    os.mkdir(out_dir)

    allids = nonepi_subids + epi_subids

    for id in tqdm(allids):

        sub_dir = os.path.join(out_dir, id)
        os.mkdir(sub_dir)
        '''
        lesion_mask = os.path.join(studydir, id, 'vae_mse.flair.mask.nii.gz')
        copyfile(lesion_mask, os.path.join(out_dir, id, 'lesion.mask.nii.gz'))
        '''

        lesion_error = os.path.join(studydir, id, 'vae_mse.flair.nii.gz')
        copyfile(lesion_error, os.path.join(out_dir, id, 'lesion.nii.gz'))

        t1 = os.path.join(studydir, id, 'T1mni.nii.gz')
        copyfile(t1, os.path.join(out_dir, id, 'T1mni.nii.gz'))

        t2 = os.path.join(studydir, id, 'T2mni.nii.gz')
        copyfile(t2, os.path.join(out_dir, id, 'T2mni.nii.gz'))

        flair = os.path.join(studydir, id, 'FLAIRmni.nii.gz')
        copyfile(t2, os.path.join(out_dir, id, 'FLAIRmni.nii.gz'))

        l = os.path.join(studydir, id, 'BrainSuite',
                         'T1mni.left.mid.cortex.svreg.dfs')
        copyfile(l, os.path.join(out_dir, id,
                                 'T1mni.left.mid.cortex.svreg.dfs'))

        r = os.path.join(studydir, id, 'BrainSuite',
                         'T1mni.right.mid.cortex.svreg.dfs')
        copyfile(r,
                 os.path.join(out_dir, id, 'T1mni.right.mid.cortex.svreg.dfs'))

        stats = os.path.join(studydir, id, 'BrainSuite',
                             'T1mni.roiwise.stats.txt')
        copyfile(stats, os.path.join(out_dir, id, 'T1mni.roiwise.stats.txt'))

        jac = os.path.join(studydir, id, 'BrainSuite',
                           'T1mni.svreg.jacobian.nii.gz')
        copyfile(jac, os.path.join(out_dir, id, 'T1mni.svreg.jacobian.nii.gz'))

        lab = os.path.join(studydir, id, 'BrainSuite',
                           'T1mni.svreg.label.nii.gz')
        copyfile(lab, os.path.join(out_dir, id, 'T1mni.svreg.label.nii.gz'))

        make_archive(os.path.join(out_dir, id + '_anat'),
                     'zip',
                     base_dir=id,
                     root_dir=out_dir)


if __name__ == "__main__":
    main()
