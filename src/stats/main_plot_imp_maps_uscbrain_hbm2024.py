
from surfproc import patch_color_attrib, view_patch_vtk
import numpy as np
from dfsio import readdfs, writedfs
import nilearn.image as ni
from nilearn.plotting import plot_stat_map, show, plot_anat


cut_coords = (100/2, 212-212/2, 104)

outbase = 'USCBrain_imp'

left = readdfs(outbase+'.left.USCBrain.imp.dfs')
right = readdfs(outbase+'.right.USCBrain.imp.dfs')


left.attributes = np.maximum(left.attributes, 0)
patch_color_attrib(left, cmap='hot', zerocolor=[.5, .5, .5])
writedfs(outbase+'.left.uscbrain.imp.dfs', left)
view_patch_vtk(left, outfile=outbase+'.left.uscbrain.imp_v1.png', show=0)
view_patch_vtk(left, azimuth=-90, elevation=0, roll=90, outfile=outbase+'.left.uscbrain.imp_v2.png', show=0)

right.attributes = np.maximum(right.attributes, 0)
patch_color_attrib(right, cmap='hot', zerocolor=[.5, .5, .5])
view_patch_vtk(right, outfile=outbase+'.right.uscbrain.imp_v1.png', show=0)
view_patch_vtk(right, azimuth=-90, elevation=0, roll=90, outfile=outbase+'.right.uscbrain.imp_v2.png', show=0)

import matplotlib.pyplot as plt
plot_stat_map(stat_map_img=outbase+'feat_USCBrain.imp.nii.gz',
              bg_img='/home/ajoshi/Software/BrainSuite23a/svreg/USCBrain/USCBrain.nii.gz',
              threshold=1e-6,
              draw_cross=False,
              cut_coords=cut_coords,
              display_mode="ortho",
              output_file=outbase+'feat_uscbrain.imp.png',
              symmetric_cbar=False,
              annotate=True,cmap='hot',
              vmin=0,vmax=0.75,
              dim=-0.5)


plt.show()