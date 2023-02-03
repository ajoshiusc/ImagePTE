
from surfproc import patch_color_attrib, view_patch_vtk
import numpy as np
from dfsio import readdfs, writedfs
import nilearn.image as ni
from nilearn.plotting import plot_stat_map, show, plot_anat


cut_coords = (100/2, 212-212/2, 104)

outbase = 'brainnetome_pred'

left = readdfs(outbase+'.left.brainnetome.imp.dfs')
left_uscbrain = readdfs('/ImagePTE1/ajoshi/code_farm/svreg/USCBrain/USCBrain.left.mid.cortex.dfs')
left.vertices = left_uscbrain.vertices

right = readdfs(outbase+'.right.brainnetome.imp.dfs')
right_uscbrain = readdfs('/ImagePTE1/ajoshi/code_farm/svreg/USCBrain/USCBrain.right.mid.cortex.dfs')
right.vertices = right_uscbrain.vertices


left.attributes = np.maximum(left.attributes, 0)
patch_color_attrib(left, cmap='hot', zerocolor=[.25, .25, .25])
view_patch_vtk(left, outfile=outbase+'.left.brainnetome.imp_v1.png', show=0)
view_patch_vtk(left, azimuth=-90, elevation=0, roll=90,
               outfile=outbase+'.left.brainnetome.imp_v2.png', show=0)

right.attributes = np.maximum(right.attributes, 0)
patch_color_attrib(right, cmap='hot', zerocolor=[.25, .25, .25])
view_patch_vtk(right, outfile=outbase+'.right.brainnetome.imp_v1.png', show=0)
view_patch_vtk(right, azimuth=-90, elevation=0, roll=90,
               outfile=outbase+'.right.brainnetome.imp_v2.png', show=0)

plot_stat_map(outbase+'feat_brainnetome.imp.nii.gz',
              '/ImagePTE1/ajoshi/code_farm/svreg/USCBrain/USCBrain.nii.gz',
              threshold=0,
              draw_cross=False,
              cut_coords=cut_coords,
              display_mode="ortho",
              output_file=outbase+'feat_brainnetome.imp.png',
              annotate=True)
