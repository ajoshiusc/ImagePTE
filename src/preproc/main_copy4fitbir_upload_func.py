import os
from shutil import copyfile, make_archive, rmtree
from tqdm import tqdm

studydir = '/home/ajoshi/project_ajoshi_27/ImagePTE1/fitbir/preproc/maryland_rao_v1'

epi_txt = '/home/ajoshi/project_ajoshi_27/ImagePTE1/fitbir/preproc/maryland_rao_v1_epilepsy_imgs.txt'
nonepi_txt = '/home/ajoshi/project_ajoshi_27/ImagePTE1/fitbir/preproc/maryland_rao_v1_nonepilepsy_imgs_37.txt'
cut_coords = (60 - 182 / 2, 122 - 212 / 2, 90 - 182 / 2)


def main():

    with open(epi_txt) as f:
        epiIds = f.readlines()

    with open(nonepi_txt) as f:
        nonepiIds = f.readlines()

    epi_subids = list(map(lambda x: x.strip(), epiIds))
    nonepi_subids = list(map(lambda x: x.strip(), nonepiIds))

    out_dir = '/home/ajoshi/Desktop/fitbir_upload_functional_analysis'

    if os.path.exists(out_dir):
        rmtree(out_dir)
    
    os.mkdir(out_dir)

    allids = nonepi_subids + epi_subids

    for id in tqdm(allids):

        sub_dir = os.path.join(out_dir, id)
        os.mkdir(sub_dir)


        fname = os.path.join(studydir, id,'BFP',id, 'func', id+'_rest_bold.32k.GOrd.filt.mat')        
        if os.path.isfile(fname):
            copyfile(fname, os.path.join(out_dir, id, id+'_rest_bold.32k.GOrd.filt.mat'))

        fname = os.path.join(studydir, id,'BFP',id, 'func', id+'_rest_bold.32k.GOrd.mat')
        if os.path.isfile(fname):
            copyfile(fname, os.path.join(out_dir, id, id+'_rest_bold.32k.GOrd.mat'))

        fname = os.path.join(studydir, id,'BFP',id, 'func', id+'_rest_bold.ALFF.GOrd.mat')
        if os.path.isfile(fname):
            copyfile(fname, os.path.join(out_dir, id, id+'_rest_bold.ALFF.GOrd.mat'))

        fname = os.path.join(studydir, id,'BFP',id, 'func', id+'_rest_bold.ALFF_Z.GOrd.mat')
        if os.path.isfile(fname):
            copyfile(fname, os.path.join(out_dir, id, id+'_rest_bold.ALFF_Z.GOrd.mat'))

        fname = os.path.join(studydir, id,'BFP',id, 'func', id+'_rest_bold.fALFF.GOrd.mat')
        if os.path.isfile(fname):
            copyfile(fname, os.path.join(out_dir, id, id+'_rest_bold.fALFF.GOrd.mat'))


        make_archive(os.path.join(out_dir, id + '_func'),
                     'zip',
                     base_dir=id,
                     root_dir=out_dir)


if __name__ == "__main__":
    main()
