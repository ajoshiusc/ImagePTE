import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold

f = np.load('../connectivity/PTE_graphs.npz')
conn_pte = f['conn_mat']
lab_ids = f['label_ids']
gordlab = f['labels']
sub_ids = f['sub_ids']
n_rois = conn_pte.shape[0]
ind = np.tril_indices(n_rois, k=1)
epi_connectivity = conn_pte[ind[0], ind[1], :].T

a = np.load('PTE_lesion_vols.npz', allow_pickle=True)
a = a['lesion_vols'].item()
epi_lesion_vols = np.array([a[k] for k in sub_ids])
epi_measures = np.concatenate((epi_connectivity, epi_lesion_vols), axis=1)


f = np.load('../connectivity/NONPTE_graphs.npz')
conn_nonpte = f['conn_mat']
lab_ids = f['label_ids']
gordlab = f['labels']
sub_ids = f['sub_ids']
nonepi_connectivity = conn_nonpte[ind[0], ind[1], :].T

a = np.load('NONPTE_lesion_vols.npz', allow_pickle=True)
a = a['lesion_vols'].item()
nonepi_lesion_vols = np.array([a[k] for k in sub_ids])
nonepi_measures = np.concatenate(
    (nonepi_connectivity, nonepi_lesion_vols), axis=1)


X = np.vstack((epi_measures, nonepi_measures))
y = np.hstack(
    (np.ones(epi_measures.shape[0]), np.zeros(nonepi_measures.shape[0])))

# Permute the labels to check if AUC becomes 0.5. This check is to make sure that we are not overfitting
#y = np.random.permutation(y)

n_iter = 100
auc = np.zeros(n_iter)
precision = np.zeros(n_iter)
recall = np.zeros(n_iter)
fscore = np.zeros(n_iter)
support = np.zeros(n_iter)


my_metric = 'roc_auc'

#y = np.random.permutation(y)

for gval in (0.0001, 0.001, 0.01, 0.05, 0.075, 0.1, 0.15, .2, .3, 1, 10, 100, 1000):
    #clf = SVC(kernel='rbf', gamma=gval, tol=1e-8)
    pipe = Pipeline([('pca_apply', PCA(n_components=54, whiten=True)),
                     ('svc', SVC(kernel='rbf', gamma=gval, tol=1e-8))])
    kfold = StratifiedKFold(n_splits=36, shuffle=False)
    auc = cross_val_score(pipe, X, y, cv=kfold, scoring=my_metric)
    print('AUC after CV for gamma=%g is %g' %
          (gval, np.mean(auc)))


for nf in range(1, 70):
    for gval in ('auto', 'scale'):
        pipe = Pipeline([('pca_apply', PCA(n_components=nf, whiten=True)),
                         ('svc', SVC(kernel='rbf', gamma=gval, tol=1e-8))])
        kfold = StratifiedKFold(n_splits=36, shuffle=False)
        auc = cross_val_score(pipe, X, y, cv=kfold, scoring=my_metric)

        print('AUC after CV for nf=%dgamma=%s is %g' %
              (nf, gval, np.mean(auc)))

for nf in range(1, 70):
        pipe = Pipeline([('pca_apply', PCA(n_components=nf, whiten=True)),
                         ('svc', SVC(kernel='rbf', gamma=0.075, tol=1e-8))])
        kfold = StratifiedKFold(n_splits=36, shuffle=False)
        auc = cross_val_score(pipe, X, y, cv=kfold, scoring=my_metric)

        print('AUC after CV for nf=%dgamma=%s is %g' %
              (nf, 0.075, np.mean(auc)))
