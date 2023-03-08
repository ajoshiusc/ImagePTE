
from surfproc import patch_color_attrib, view_patch_vtk
import numpy as np
from dfsio import readdfs, writedfs
import nilearn.image as ni
from nilearn.plotting import plot_stat_map, show, plot_anat


cut_coords = (100/2, 212-212/2, 104)

outbase = 'usclobes_decode_lesion_conn'

left = readdfs(outbase+'.left.decode.dfs')
left_USCLobes = readdfs('/ImagePTE1/ajoshi/code_farm/svreg/USCLobes/BCI-DNI_brain.left.mid.cortex.dfs')
left.vertices = left_USCLobes.vertices

right = readdfs(outbase+'.right.decode.dfs')
right_USCLobes = readdfs('/ImagePTE1/ajoshi/code_farm/svreg/USCLobes/BCI-DNI_brain.right.mid.cortex.dfs')
right.vertices = right_USCLobes.vertices


left.attributes = np.maximum(left.attributes, 0)
patch_color_attrib(left, cmap='hot', zerocolor=[.5, .5, .5], clim=[0,.4])
view_patch_vtk(left, outfile=outbase+'.left.decode1.png', show=0)
view_patch_vtk(left, azimuth=-90, elevation=0, roll=90,
               outfile=outbase+'.left.decode2.png', show=0)

right.attributes = np.maximum(right.attributes, 0)
patch_color_attrib(right, cmap='hot', zerocolor=[.5, .5, .5], clim=[0,.4])
view_patch_vtk(right, outfile=outbase+'.right.decode1.png', show=0)
view_patch_vtk(right, azimuth=-90, elevation=0, roll=90,
               outfile=outbase+'.right.decode2.png', show=0)

stat_img = ni.load_img(outbase+'.decode.nii.gz')
stat_img = ni.new_img_like(stat_img, np.maximum(stat_img.get_fdata(),0))

plot_stat_map(stat_img,
              '/ImagePTE1/ajoshi/code_farm/svreg/USCLobes/BCI-DNI_brain.nii.gz',
              draw_cross=False,
              threshold=.05,
              cut_coords=cut_coords,
              display_mode="ortho",
              output_file=outbase+'decode.png',
              annotate=True,vmax=.4)
