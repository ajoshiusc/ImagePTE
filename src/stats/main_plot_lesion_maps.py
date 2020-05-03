import os

import nilearn.image as ni
import numpy as np
from matplotlib import cm
from nilearn.plotting import plot_stat_map, show, plot_anat

studydir = '/ImagePTE1/ajoshi/fitbir/preproc/maryland_rao_v1'

epi_txt = '/ImagePTE1/ajoshi/fitbir/preproc/maryland_rao_v1_epilepsy_imgs.txt'
nonepi_txt = '/ImagePTE1/ajoshi/fitbir/preproc/maryland_rao_v1_nonepilepsy_imgs_37.txt'


def check_imgs_exist(studydir, sub_ids):
    subids_imgs = list()

    for id in sub_ids:
        fname = os.path.join(studydir, id, 'lesion_vae.atlas.mask.nii.gz')

        if not os.path.isfile(fname):
            err_msg = 'the file does not exist: ' + fname
            sys.exit(err_msg)

    return subids_imgs


def readsubs(studydir, sub_ids):

    print(len(sub_ids))

    check_imgs_exist(studydir, sub_ids)
    nsub = 37

    print('Reading Subjects')

    for n, id in enumerate(sub_ids):

        fname = os.path.join(studydir, id, 'lesion_vae.atlas.mask.nii.gz')
        print('sub:', n, 'Reading', id)
        im = ni.load_img(fname)

        if n == 0:
            data = np.zeros((min(len(sub_ids), nsub), ) + im.shape)

        data[n, :, :, :] = im.get_data()

    return data, sub_ids


def main():

    with open(epi_txt) as f:
        epiIds = f.readlines()

    with open(nonepi_txt) as f:
        nonepiIds = f.readlines()

    epiIds = list(map(lambda x: x.strip(), epiIds))
    nonepiIds = list(map(lambda x: x.strip(), nonepiIds))

    epi_data, epi_subids = readsubs(studydir, epiIds)
    nonepi_data, nonepi_subids = readsubs(studydir, nonepiIds)

    for id in nonepi_subids:
        error = os.path.join(studydir, id, 'lesion_vae.atlas' + '.nii.gz')

        lesion = os.path.join(studydir, id,
                              'lesion_vae.atlas.mask' + '.nii.gz')
        anat = os.path.join(studydir, id, 'FLAIRBCI' + '.nii.gz')
        outfile1 = os.path.join('nonepi_png', id + '_error.png')
        outfile2 = os.path.join('nonepi_png', id + '_lesion.png')
        outfile3 = os.path.join('nonepi_png', id + '_anat.png')


        plot_stat_map(error,
                      anat,
                      threshold=0,
                      draw_cross=False,
                      cut_coords=(40 * 0.8, 188 * 0.546875, 240 * 0.546875),
                      display_mode="ortho",
                      output_file=outfile1,
                      annotate=True)

        plot_stat_map(lesion,
                      anat,
                      threshold=0,
                      draw_cross=False,
                      cut_coords=(40 * 0.8, 188 * 0.546875, 240 * 0.546875),
                      display_mode="ortho",
                      output_file=outfile2,
                      annotate=True)

        plot_anat(anat,
                  threshold=0,
                  draw_cross=False,
                  cut_coords=(40 * 0.8, 188 * 0.546875, 240 * 0.546875),
                  display_mode="ortho",
                  output_file=outfile3,
                  annotate=True)

    for id in epi_subids:
        lesion = os.path.join(studydir, id,
                              'lesion_vae.atlas.mask' + '.nii.gz')
        anat = os.path.join(studydir, id, 'FLAIRBCI' + '.nii.gz')
        outfile1 = os.path.join('epi_png', id + '_lesion.png')
        outfile2 = os.path.join('epi_png', id + '_anat.png')

        plot_stat_map(lesion,
                      anat,
                      threshold=0,
                      draw_cross=False,
                      cut_coords=(40 * 0.8, 188 * 0.546875, 240 * 0.546875),
                      display_mode="ortho",
                      output_file=outfile1,
                      annotate=True)

        plot_anat(anat,
                  threshold=0,
                  draw_cross=False,
                  cut_coords=(40 * 0.8, 188 * 0.546875, 240 * 0.546875),
                  display_mode="ortho",
                  output_file=outfile2,
                  annotate=True)


if __name__ == "__main__":
    main()
''''
plot_stat_map(stat_img,
              bk_img,
              threshold=0,
              draw_cross=False,
              cut_coords=(35*0.8, 144*0.546875, 210*0.546875),
              display_mode="ortho",
              output_file=outfile2,
              annotate=True)

plot_stat_map(stat_img,
              bk_img,
              threshold=0,
              draw_cross=False,
              cut_coords=(30*0.8, 100*0.546875, 180*0.546875),
              display_mode="ortho",
              output_file=outfile3,
              annotate=True)

plot_stat_map(stat_img,
              bk_img,
              threshold=0,
              draw_cross=False,
              cut_coords=(25*0.8, 66*0.546875, 150*0.546875),
              display_mode="ortho",
              output_file=outfile4,
              annotate=True)

'''
