
import sys

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.stats as ss
from matplotlib.image import imsave
from scipy.stats import norm
from sklearn.metrics import auc, plot_roc_curve, roc_auc_score, roc_curve, classification_report, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from statsmodels.stats.multitest import fdrcorrection
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from grayord_utils import visdata_grayord

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


# Do SVM Analysis
epi_measures = conn_pte[ind[0], ind[1], :].T
nonepi_measures = conn_nonpte[ind[0], ind[1], :].T

X = np.vstack((epi_measures, nonepi_measures))
y = np.hstack(
    (np.ones(epi_measures.shape[0]), np.zeros(nonepi_measures.shape[0])))

n_iter = 100
auc = np.zeros(n_iter)
precision = np.zeros(n_iter)
recall = np.zeros(n_iter)
fscore = np.zeros(n_iter)
support = np.zeros(n_iter)

auc_t = np.zeros(n_iter)
n_features = 21
y_test_true_all = []
y_test_pred_all = []

for t in tqdm(range(n_iter)):
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.33)
    #clf = RandomForestClassifier()

    clf = LinearSVC(penalty='l1',C=0.2, tol=1e-6,dual=False)
    clf.fit(X_train, y_train)
    #ind_feat = np.argsort(-clf.feature_importances_)

    #X_train, X_test, y_train, y_test = train_test_split(X,
    #                                                    y,
    #                                                    test_size=0.33)
    #clf.fit(X_train[:, ind_feat[:n_features]], y_train)

    #svc_disp = plot_roc_curve(clf, X_test, y_test)
    #y_score = clf.predict(X_test[:, ind_feat[:n_features]])
    param = clf.coef_.ravel()
    count=param[param!=0]
    print(count.shape)
    y_score = clf.predict(X_test)
    y_test_pred_all = y_test_pred_all + list(y_score)
    y_test_true_all = y_test_true_all + list(y_test)

    precision[t], recall[t], fscore[t], support[t] = precision_recall_fscore_support(
        y_test, y_score,average='micro')

    auc[t] = roc_auc_score(y_test, y_score)
    y_score = clf.predict(X_train)
    auc_t[t] = roc_auc_score(y_train, y_score)
    #print(auc[t], auc_t[t])

print('running done')

target_names = ['class PTE', 'class nonPTE']
print(classification_report(y_test_true_all,
                            y_test_pred_all, target_names=target_names))

print('precision:',np.mean(precision), np.std(precision))
print('recall:',np.mean(recall), np.std(recall))
print('fscore:',np.mean(fscore), np.std(fscore))
print('support:',np.mean(support), np.std(support))

print(np.mean(auc), np.std(auc))
print(np.mean(auc_t), np.std(auc_t))

auc = roc_auc_score(y_test_true_all, y_test_pred_all)
print(auc)
print('done')
