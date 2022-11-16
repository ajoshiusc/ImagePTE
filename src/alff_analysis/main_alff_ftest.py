import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
import scipy.stats as ss
from scipy.stats import ranksums

measure = 'ALFF'
atlas_name = 'USCLobes'
f = np.load('PTE_'+measure+'_'+atlas_name+'.npz')
conn_pte = f['roiwise_data']
lab_ids = f['label_ids']
#gordlab = f['labels']
sub_ids = f['sub_ids']
n_rois = conn_pte.shape[0]
epi_connectivity = conn_pte.T

#a = np.load('stats/PTE_lesion_vols_USCLobes.npz', allow_pickle=True)
#a = a['lesion_vols'].item()
#epi_lesion_vols = np.array([a[k] for k in sub_ids])
epi_measure = epi_connectivity


atlas_name = 'USCLobes'
f = np.load('NONPTE_'+measure+'_'+atlas_name+'.npz')
conn_pte = f['roiwise_data']
lab_ids = f['label_ids']
#gordlab = f['labels']
sub_ids = f['sub_ids']
n_rois = conn_pte.shape[0]
nonepi_connectivity = conn_pte.T

#a = np.load('stats/PTE_lesion_vols_USCLobes.npz', allow_pickle=True)
#a = a['lesion_vols'].item()
#epi_lesion_vols = np.array([a[k] for k in sub_ids])
nonepi_measure = nonepi_connectivity





F = epi_measure.var(axis=0) / (nonepi_measure.var(axis=0) + 1e-6)
pval = 1 - ss.f.cdf(F, 37*8 - 1, 37*8-1)

print(lab_ids, pval)

lab_ids = np.array(lab_ids)
rois = lab_ids[pval<0.05]

print(rois)

pval=np.zeros(epi_measure.shape[1])

for i in range(epi_measure.shape[1]):
    stat, pval[i] = ranksums(epi_measure[:,i], nonepi_measure[:,i])

print(pval)
rois = lab_ids[pval<0.05]
print(rois)


