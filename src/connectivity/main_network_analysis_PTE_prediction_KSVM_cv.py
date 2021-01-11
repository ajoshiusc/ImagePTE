import sys

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.stats as ss
from matplotlib.image import imsave
from scipy.stats import norm
from sklearn.metrics import auc, plot_roc_curve, roc_auc_score, roc_curve, classification_report, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from statsmodels.stats.multitest import fdrcorrection
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from grayord_utils import visdata_grayord
from sklearn.model_selection import cross_val_score

population = 'PTE'
f = np.load(population+'_graphs.npz')
conn_pte = f['conn_mat']
lab_ids = f['label_ids']
gordlab = f['labels']

population = 'NONPTE'
f = np.load(population+'_graphs.npz')
conn_nonpte = f['conn_mat']
lab_ids = f['label_ids']
gordlab = f['labels']

n_rois = conn_pte.shape[0]
ind = np.tril_indices(n_rois, k=1)


# Do Random Forest Analysis
epi_measures = conn_pte[ind[0], ind[1], :].T
nonepi_measures = conn_nonpte[ind[0], ind[1], :].T

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

for gval in (0.0001, 0.001, 0.01, 0.05, 0.1, .2, .3, 1, 10, 100, 1000):
    clf = SVC(kernel='rbf', gamma=gval, tol=1e-8)

    auc = cross_val_score(clf, X, y, cv=36, scoring=my_metric)
    print('AUC after CV for gamma=%g is %g(%g)' %
          (gval, np.mean(auc), np.std(auc)))


for gval in ('auto', 'scale'):
    clf = SVC(kernel='rbf',  gamma=gval, tol=1e-8)

    auc = cross_val_score(clf, X, y, cv=36, scoring=my_metric)
    print('AUC after CV for gamma=%s is %g(%g)' %
          (gval, np.mean(auc), np.std(auc)))
