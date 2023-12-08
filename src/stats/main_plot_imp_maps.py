
from surfproc import patch_color_attrib, view_patch_vtk
import numpy as np
from dfsio import readdfs, writedfs
import nilearn.image as ni
from nilearn.plotting import plot_stat_map, show, plot_anat


cut_coords = (100/2, 212-212/2, 104)

outbase = 'uscbrain_pred'

left = readdfs(outbase+'.left.uscbrain.imp.dfs')
right = readdfs(outbase+'.right.uscbrain.imp.dfs')


left.attributes = np.maximum(left.attributes, 0)
patch_color_attrib(left, cmap='hot', zerocolor=[.25,.25,.25])
view_patch_vtk(left, outfile=outbase+'.left.uscbrain.imp_v1.png', show=0)
view_patch_vtk(left, azimuth=-90,elevation=0, roll=90, outfile=outbase+'.left.lobes.imp_v2.png', show=0)

right.attributes = np.maximum(right.attributes, 0)
patch_color_attrib(right, cmap='hot', zerocolor=[.25,.25,.25])
view_patch_vtk(right, outfile=outbase+'.right.uscbrain.imp_v1.png', show=0)

view_patch_vtk(right, azimuth=-90,elevation=0, roll=90, outfile=outbase+'.right.uscbrain.imp_v2.png', show=0)


stat_img = ni.load_img(outbase+'feat_lobes.imp.nii.gz')
stat_img = ni.new_img_like(stat_img, np.maximum(stat_img.get_fdata(),0))

plot_stat_map(stat_img,
                '/ImagePTE1/ajoshi/code_farm/svreg/USCBrain/BCI-DNI_brain.nii.gz',
                draw_cross=False,
                cut_coords=cut_coords,
                display_mode="ortho",
                symmetric_cbar=False,
                colorbar=True,
                output_file=outbase+'feat_lobes.imp.png',
                annotate=True)