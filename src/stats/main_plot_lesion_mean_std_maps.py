import os

import nilearn.image as ni
import numpy as np
from matplotlib import cm
from nilearn.plotting import plot_stat_map, show, plot_anat

studydir = '/ImagePTE1/ajoshi/fitbir/preproc/maryland_rao_v1'

epi_txt = '/ImagePTE1/ajoshi/fitbir/preproc/maryland_rao_v1_epilepsy_imgs.txt'
nonepi_txt = '/ImagePTE1/ajoshi/fitbir/preproc/maryland_rao_v1_nonepilepsy_imgs_37.txt'
cut_coords = (60-182, 90, 90-182)



def main():

    anat = '/home/ajoshi/BrainSuite21a/svreg/USCBrain/USCBrain.nii.gz'


    epi_mean = 'epi_mean_lesion0mm.nii.gz'

    plot_stat_map(epi_mean,
                    anat,
                    threshold=0.001,
                    draw_cross=False,
                    cut_coords=cut_coords,
                    display_mode="ortho",vmax=0.15,
                    output_file='epi_mean_lesion0mm.png',
                    annotate=True)


    nonepi_mean = 'nonepi_mean_lesion0mm.nii.gz'

    plot_stat_map(nonepi_mean,
                    anat,
                    threshold=0.001,
                    draw_cross=False,
                    cut_coords=cut_coords,
                    display_mode="ortho",vmax=0.15,
                    output_file='nonepi_mean_lesion0mm.png',
                    annotate=True)

    epi_std = 'epi_std_lesion0mm.nii.gz'

    plot_stat_map(epi_std,
                    anat,
                    threshold=0,
                    draw_cross=False,
                    cut_coords=cut_coords,vmax=0.08,
                    display_mode="ortho",
                    output_file='epi_std_lesion0mm.png',
                    annotate=True)


    nonepi_std = 'nonepi_std_lesion0mm.nii.gz'

    plot_stat_map(nonepi_std,
                    anat,
                    threshold=0,
                    draw_cross=False,
                    cut_coords=cut_coords,vmax=0.08,
                    display_mode="ortho",
                    output_file='nonepi_std_lesion0mm.png',
                    annotate=True)



if __name__ == "__main__":
    main()
