
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

imsave('p_value_conn.png', 0.05-p_values,
       vmin=0, vmax=0.05, cmap='jet')


# f-test on rois

F = co1.var(axis=2) / (co2.var(axis=2) + 1e-6)
nsub = co2.shape[2]
p_values = (1 - ss.f.cdf(F, nsub - 1, nsub - 1))

imsave('p_value_conn_ftest.png', 0.05-p_values,
       vmin=0, vmax=0.05, cmap='jet')


#_, tmp = fdrcorrection(p_values.flatten())

#p_values_fdr = tmp.reshape(p_values.shape)

# imsave('p_value_conn_ftest_fdr.png', p_values_fdr,
#       vmin=0, vmax=0.05, cmap='jet')


# Save results on ROIs


gord_pval = np.zeros(len(gordlab))

for i, id in enumerate(lab_ids):
    gord_pval[gordlab == id] = np.min(p_values[i, :])


visdata_grayord(data=0.05-gord_pval,
                smooth_iter=100,
                colorbar_lim=[0, .05],
                colormap='jet',
                save_png=True,
                surf_name='f_test_pte',
                out_dir='.',
                bfp_path='/ImagePTE1/ajoshi/code_farm/bfp',
                fsl_path='/usr/share/fsl')

# FDR correction over ROIs

pval_rois = np.min(p_values, axis=1)
_, pval_rois_fdr = fdrcorrection(pval_rois)

gord_pval = np.zeros(len(gordlab))

for i, id in enumerate(lab_ids):
    gord_pval[gordlab == id] = pval_rois_fdr[i]

visdata_grayord(data=0.05-gord_pval,
                smooth_iter=100,
                colorbar_lim=[0, .05],
                colormap='jet',
                save_png=True,
                surf_name='f_test_pte_fdr',
                out_dir='.',
                bfp_path='/ImagePTE1/ajoshi/code_farm/bfp',
                fsl_path='/usr/share/fsl')
