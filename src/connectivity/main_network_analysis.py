#||AUM||
#||Shree Ganeshaya namaha||

import networkx as nx
import numpy as np

c = np.load('hcp_graphs.npz')
c = c['conn_mat']

c = np.median(c, axis=2)
G = nx.convert_matrix.from_numpy_array(c)
cent = nx.eigenvector_centrality(G)


print(cent)

input('Press any key')
