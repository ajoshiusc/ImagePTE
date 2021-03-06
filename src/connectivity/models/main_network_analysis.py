
import sys

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.image import imsave
from grayord_utils import visdata_grayord

population = 'PTE'
f = np.load(population+'_graphs.npz')
co = f['conn_mat']
lab_ids = f['label_ids']
gordlab = f['labels']

c = np.mean(co, axis=2)
np.fill_diagonal(c, 0)


G = nx.convert_matrix.from_numpy_array(np.abs(c))
cent = nx.eigenvector_centrality(G, weight='weight')

cent = np.array(list(cent.items()))
plt.hist(cent[:, 1])
plt.show()

gord_cent = np.zeros(len(gordlab))

for i, id in enumerate(lab_ids):
    gord_cent[gordlab == id] = cent[i, 1]


visdata_grayord(data=gord_cent,
                smooth_iter=100,
                colorbar_lim=[0, .12],
                colormap='jet',
                save_png=True,
                surf_name='centrality_'+population,
                out_dir='.',
                bfp_path='/ImagePTE1/ajoshi/code_farm/bfp',
                fsl_path='/usr/share/fsl')


imsave('corr_ave_'+population+'.jpg', np.mean(co, axis=2),
       vmin=-1.0, vmax=1.0, cmap='jet')

imsave('corr_std_'+population+'.jpg', np.std(co, axis=2),
       vmin=-1.0, vmax=1.0, cmap='jet')


print(nx.info(G))

pos = nx.spring_layout(G, weight='weight', iterations=1000)
plt.subplot(211)
nx.draw(G, pos, with_labels=False, node_size=10)

plt.axis('off')
plt.show()


input('Press any key')
