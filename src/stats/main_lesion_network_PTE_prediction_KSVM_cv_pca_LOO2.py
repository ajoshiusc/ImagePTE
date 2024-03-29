import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, LeaveOneOut
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

f = np.load('../connectivity/PTE_graphs.npz')
conn_pte = f['conn_mat']
lab_ids = f['label_ids']
gordlab = f['labels']
sub_ids = f['sub_ids']
cent_mat = f['cent_mat']
n_rois = conn_pte.shape[0]
ind = np.tril_indices(n_rois, k=1)
epi_connectivity = conn_pte[ind[0], ind[1], :].T

a = np.load('../stats/PTE_lesion_vols.npz', allow_pickle=True)
a = a['lesion_vols'].item()
epi_lesion_vols = np.array([a[k] for k in sub_ids])
epi_measures = np.concatenate(
    (epi_connectivity, epi_lesion_vols), axis=1)


f = np.load('../connectivity/NONPTE_graphs.npz')
conn_nonpte = f['conn_mat']
lab_ids = f['label_ids']
gordlab = f['labels']
sub_ids = f['sub_ids']
cent_mat = f['cent_mat']

nonepi_connectivity = conn_nonpte[ind[0], ind[1], :].T

a = np.load('../stats/NONPTE_lesion_vols.npz', allow_pickle=True)
a = a['lesion_vols'].item()
nonepi_lesion_vols = np.array([a[k] for k in sub_ids])
nonepi_measures = np.concatenate(
    (nonepi_connectivity, nonepi_lesion_vols), axis=1)


X = np.vstack((epi_measures, nonepi_measures))
y = np.hstack(
    (np.ones(epi_measures.shape[0]), np.zeros(nonepi_measures.shape[0])))

# Permute the labels to check if AUC becomes 0.5. This check is to make sure that we are not overfitting
#y = np.random.permutation(y)

my_metric = 'roc_auc'


#######################selecting gamma################
# Following part of the code do a grid search to find best value of gamma using a one fold cross validation
# the metric for comparing the performance is AUC
####################################################
best_c = 0
max_AUC = 0
C_range = [0.0001, 0.001, 0.01, .1, .3, .6, .9,
           1, 1.5, 2, 3, 4, 5, 6, 7, 9, 10, 100]
'''
for mygamma in ['auto', 'scale']:
clf = SVC(kernel='rbf', gamma=mygamma, tol=1e-9)
my_metric = 'roc_auc'
kfold = StratifiedKFold(n_splits=37, shuffle=False)
auc = cross_val_score(clf, X, y, cv=kfold, scoring=my_metric)
print('AUC on testing data:gamma=%s, auc=%g' % (mygamma, np.mean(auc)))
'''
#######################selecting gamma################
# Following part of the code do a grid search to find best number of PCA component
# the metric for comparing the performance is AUC
####################################################
best_com = 53
max_AUC = 0
#######################selecting gamma################
# Random permutation of pairs of training subject for 1000 iterations
####################################################
iteration_num = 1
auc_sum = np.zeros((iteration_num))

# y = np.random.permutation(y)
#kfold = StratifiedKFold(n_splits=36, shuffle=True)
#auc = cross_val_score(pipe, X, y, cv=kfold, scoring=my_metric)
cv = LeaveOneOut()  # StratifiedKFold(n_splits=36, shuffle=True)

y_pred = np.zeros(y.shape)
for train, test in tqdm(cv.split(X, y)):

    max_AUC = 0
    best_gamma = 0
    gamma_range = [1, 0.001, 0.05, 0.075, .1, .13, .15, .17,
                   0.2, 0.3, .5, 1, 5, 10, 100]
    for current_gamma in gamma_range:
        clf = SVC(kernel='rbf', gamma=current_gamma, tol=1e-9)
        my_metric = 'roc_auc'
        #auc = cross_val_score(clf, X, y, cv=37, scoring=my_metric)
        kfold = StratifiedKFold(n_splits=35, shuffle=True, random_state=1211)
        auc = cross_val_score(
            clf, X[train], y[train], cv=kfold, scoring=my_metric)
        #print('AUC on testing data:gamma=%g, auc=%g' % (current_gamma, np.mean(auc)))
        if np.mean(auc) >= max_AUC:
            max_AUC = np.mean(auc)
            best_gamma = current_gamma

    max_AUC = 0
    best_c = 0
    for current_c in C_range:
        clf = Pipeline([('svc', SVC(kernel='rbf', C=current_c, tol=1e-9))])
        my_metric = 'roc_auc'

        cv = StratifiedKFold(n_splits=35, shuffle=True, random_state=1211)
        auc = cross_val_score(
            clf, X[train], y[train], cv=cv, scoring=my_metric)
        if np.mean(auc) >= max_AUC:
            best_c = current_c
            max_AUC = np.mean(auc)

    max_AUC = 0
    best_com = 0
    max_component = min((X.shape[0]-4), X.shape[1])

    for nf in range(5, max_component):
        pipe = Pipeline([('pca_apply', PCA(n_components=nf, whiten=True)),
                         ('svc', SVC(kernel='rbf', C=best_c, tol=1e-9))])
        kfold = StratifiedKFold(n_splits=35, shuffle=True, random_state=1211)
        auc = cross_val_score(
            pipe, X[train], y[train], cv=kfold, scoring=my_metric)

        if np.mean(auc) >= max_AUC:
            best_com = nf
            max_AUC = np.mean(auc)

    clf = Pipeline([('pca_apply', PCA(n_components=best_com, whiten=True)),
                    ('svc', SVC(kernel='rbf', C=best_c, tol=1e-9))])

    clf.fit(X[train], y[train])
    ytest = clf.predict(X[test])
    y_pred[test] = ytest

auc = roc_auc_score(y, y_pred)
auc_sum = np.mean(auc)


print('Average AUC with PCA=%g , Std AUC=%g' %
      (np.mean(auc_sum), np.std(auc_sum)))
