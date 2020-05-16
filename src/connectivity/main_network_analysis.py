
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

f = np.load('hcp_graphs.npz')
c = f['conn_mat']
lab_ids = f['label_ids']
gordlab = f['labels']

c = np.median(c, axis=2)
np.fill_diagonal(c,0)


G = nx.convert_matrix.from_numpy_array(np.abs(c))
cent = nx.eigenvector_centrality(G,weight='weight')

cent=np.array(list(cent.items()))
plt.hist(cent[:,1])
plt.show()

gord_cent = np.zeros(len(gordlab))

for i,id in enumerate(lab_ids):
    gord_cent[gordlab == id]=cent[i,1]


print(nx.info(G))

pos = nx.spring_layout(G, weight='weight',iterations=1000)
plt.subplot(211)
nx.draw(G, pos, with_labels=False, node_size=10)

plt.axis('off')
plt.show()


input('Press any key')
























