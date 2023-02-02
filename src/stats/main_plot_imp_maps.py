
from surfproc import patch_color_attrib, view_patch_vtk
import numpy as np
from dfsio import readdfs, writedfs
import nilearn.image as ni
from nilearn.plotting import plot_stat_map, show, plot_anat


cut_coords = (100/2, 212-212/2, 104)

outbase = 'lobes_pred'

left = readdfs(outbase+'.left.lobes.imp.dfs')
right = readdfs(outbase+'.right.lobes.imp.dfs')


left.attributes = np.maximum(left.attributes, 0)
patch_color_attrib(left, cmap='hot')
view_patch_vtk(left, outfile=outbase+'.left.lobes.imp_v1.png', show=0)
view_patch_vtk(left, azimuth=-90,elevation=0, roll=90, outfile=outbase+'.left.lobes.imp_v2.png', show=0)

right.attributes = np.maximum(right.attributes, 0)
patch_color_attrib(right, cmap='hot')
view_patch_vtk(right, outfile=outbase+'.right.lobes.imp_v1.png', show=0)
view_patch_vtk(right, azimuth=-90,elevation=0, roll=90, outfile=outbase+'.right.lobes.imp_v2.png', show=0)

plot_stat_map(outbase+'feat_lobes.imp0.nii.gz',
                '/ImagePTE1/ajoshi/code_farm/svreg/USCLobes/BCI-DNI_brain.nii.gz',
                threshold=0,
                draw_cross=False,
                cut_coords=cut_coords,
                display_mode="ortho",
                output_file=outbase+'feat_lobes.imp.png',
                annotate=True)