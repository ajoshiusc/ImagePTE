import numpy as np
# %matplotlib notebook

# from k_means import KMeans

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from skimage import io
import time
import pdb
from sklearn.cluster import KMeans

def main():   
    population = 'PTE'
    # epidata = np.load(population + '_graphs_gcn.npz')
    epidata = np.load("/home/wenhuicu/ImagePTE/pte_gcn/pygcn/PTE_graphs_gcn.npz")
    # epidata = np.load("PTE_BCI-DNI_all.npz")
    # epidata = np.load("/home/wenhuicu/ImagePTE/pte_gcn/PRGNN/ADHD_parPearson_Lobes.npz")

    # adj_epi = calc_DAD(epidata)  # n_subjects*16 *16
    features_epi = epidata['features']

    # cca = rcca.CCA(kernelcca=False, reg=0, numCC=2)
    # cca.train()

    population = 'NONPTE'
    # nonepidata = np.load(population + '_graphs_gcn.npz')
    nonepidata = np.load("/home/wenhuicu/ImagePTE/pte_gcn/pygcn/NONPTE_graphs_gcn.npz")
    # nonepidata = np.load("NONPTE_BCI-DNI_all.npz")
    # nonepidata = np.load("/home/wenhuicu/ImagePTE/pte_gcn/PRGNN/TDC_parPearson_Lobes.npz")
    # adj_non = calc_DAD(nonepidata)
    features_non = nonepidata['features']  # subjects x 16 x 171

    # print(adj_non.shape, adj_epi.shape)
    ## for now we are using the same number of epi , non epi training samples.

    features = np.concatenate((features_epi, features_non)).reshape((72, -1))
    adj = np.concatenate((epidata['conn_mat'], nonepidata['conn_mat'])).reshape((72, -1))
    # labels = torch.from_numpy(np.hstack((np.ones(adj_epi.shape[0]), np.zeros(adj_non.shape[0])))).long().to(device)
    print(adj.shape)
    k_means = KMeans(n_clusters=20, random_state=0).fit(adj)
    print(k_means.labels_[:36])
    print(k_means.labels_[36:])
    

if __name__ == '__main__':
    main()
