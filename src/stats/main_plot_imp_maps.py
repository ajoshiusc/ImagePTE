
from surfproc import patch_color_attrib, view_patch_vtk
import numpy as np
from dfsio import readdfs, writedfs


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
