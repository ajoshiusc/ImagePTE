
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

c = np.load('hcp_graphs.npz')
c = c['conn_mat']

c = np.median(c, axis=2)
np.fill_diagonal(c,0)


G = nx.convert_matrix.from_numpy_array(np.abs(c))
cent = nx.eigenvector_centrality(G,weight='weight')


print(nx.info(G))

pos = nx.spring_layout(G, weight='weight',iterations=1000)
plt.subplot(211)
nx.draw(G, pos, with_labels=False, node_size=10)

plt.axis('off')
plt.show()


input('Press any key')
























