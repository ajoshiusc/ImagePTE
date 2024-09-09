
from surfproc import patch_color_attrib, view_patch_vtk
import numpy as np
from dfsio import readdfs, writedfs
import nilearn.image as ni
from nilearn.plotting import plot_stat_map, show, plot_anat


cut_coords = (100/2, 212-212/2, 104)

outbase = 'brainnetome_imp'

left = readdfs(outbase+'.left.brainnetome.imp.dfs')
right = readdfs(outbase+'.right.brainnetome.imp.dfs')

left_uscbrain = readdfs('/home/ajoshi/Software/BrainSuite23a/svreg/USCBrain/USCBrain.left.mid.cortex.dfs')
left.vertices = left_uscbrain.vertices

right_uscbrain = readdfs('/home/ajoshi/Software/BrainSuite23a/svreg/USCBrain/USCBrain.right.mid.cortex.dfs')
right.vertices = right_uscbrain.vertices


left.attributes = np.maximum(left.attributes, 0)
patch_color_attrib(left, cmap='hot', zerocolor=[.5, .5, .5])
writedfs(outbase+'.left.brainnetome.imp.dfs', left)
view_patch_vtk(left, outfile=outbase+'.left.brainnetome.imp_v1.png', show=0)
view_patch_vtk(left, azimuth=-90, elevation=0, roll=90, outfile=outbase+'.left.brainnetome.imp_v2.png', show=0)

right.attributes = np.maximum(right.attributes, 0)
patch_color_attrib(right, cmap='hot', zerocolor=[.5, .5, .5])
view_patch_vtk(right, outfile=outbase+'.right.brainnetome.imp_v1.png', show=0)
view_patch_vtk(right, azimuth=-90, elevation=0, roll=90, outfile=outbase+'.right.brainnetome.imp_v2.png', show=0)

stat_img = ni.load_img(outbase+'feat_brainnetome.imp.nii.gz')
stat_img = ni.new_img_like(stat_img, np.maximum(stat_img.get_fdata(),0))


import matplotlib.pyplot as plt
plot_stat_map(stat_map_img=stat_img,
              bg_img='/home/ajoshi/Software/BrainSuite23a/svreg/USCBrain/USCBrain.nii.gz',
              threshold=1e-6,
              draw_cross=False,
              cut_coords=cut_coords,
              display_mode="ortho",
              output_file=outbase+'feat_brainnetome.imp.png',
              symmetric_cbar=False,
              annotate=True,cmap='hot',
              #vmin=0,vmax=0.75,
              dim=-0.5)


plt.show()