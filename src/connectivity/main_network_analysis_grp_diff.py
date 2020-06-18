
import sys

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.stats as ss
from matplotlib.image import imsave
#import scipy.stats
from scipy.stats import norm
from statsmodels.stats.multitest import fdrcorrection

from grayord_utils import visdata_grayord

population = 'PTE'
f = np.load(population+'_graphs.npz')
co1 = f['conn_mat']
lab_ids = f['label_ids']
gordlab = f['labels']

population = 'NONPTE'
f = np.load(population+'_graphs.npz')
co2 = f['conn_mat']
lab_ids = f['label_ids']
gordlab = f['labels']


c1 = np.mean(co1, axis=2)
c2 = np.mean(co2, axis=2)

s1 = np.std(co1, axis=2)
s2 = np.std(co2, axis=2)


z = np.abs(c1-c2)/((s1+s2)/2)

p_values = norm.sf(abs(z))

imsave('z_score_conn.png', z,
       vmin=-1.0, vmax=1.0, cmap='jet')

imsave('p_value_conn.png', p_values,
       vmin=0, vmax=0.05, cmap='jet')


F = co1.var(axis=2) / (co2.var(axis=2) + 1e-6)

nsub = co2.shape[2]
p_values = 1 - ss.f.cdf(F, nsub - 1, nsub - 1)

imsave('p_value_conn_ftest.png', p_values,
       vmin=0, vmax=0.05, cmap='jet')

_, tmp = fdrcorrection(p_values.flatten())

p_values = tmp.reshape(p_values.shape)

imsave('p_value_conn_ftest_fdr.png', p_values,
       vmin=0, vmax=0.05, cmap='jet')


input('Press any key')
