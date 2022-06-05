import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from numpy.random import permutation

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
num_sub_all = len(sub_ids)

nonepi_connectivity = conn_nonpte[ind[0], ind[1], :].T

a = np.load('../stats/NONPTE_lesion_vols.npz', allow_pickle=True)
a = a['lesion_vols'].item()
nonepi_lesion_vols = np.array([a[k] for k in sub_ids])
nonepi_measures = np.concatenate(
    (nonepi_connectivity, nonepi_lesion_vols), axis=1)

auc_mean_sub = []
auc_std_sub = []
n_iter = 100

auc_sub_all = np.zeros((num_sub_all, n_iter))

for num_sub in range(2, num_sub_all):


    # Permute the labels to check if AUC becomes 0.5. This check is to make sure that we are not overfitting

    auc = np.zeros(n_iter)
    precision = np.zeros(n_iter)
    recall = np.zeros(n_iter)
    fscore = np.zeros(n_iter)
    support = np.zeros(n_iter)

    my_metric = 'roc_auc'

    #y = np.random.permutation(y)

    #######################selecting gamma################
    # Following part of the code do a grid search to find best value of gamma using a one fold cross validation
    # the metric for comparing the performance is AUC
    ####################################################
    best_gamma = 0
    max_AUC = 0

    best_com = 53
    best_gamma = 0.075 # 0.075
    best_C = .1
    #######################selecting gamma################
    # Random permutation of pairs of training subject for 1000 iterations
    ####################################################
    iteration_num = 100
    auc_sum = np.zeros((iteration_num))
    for i in range(iteration_num):
        p = permutation(num_sub_all)
        X = np.vstack((epi_measures[p[:num_sub], ], nonepi_measures[p[:num_sub], ]))
        y = np.hstack(
            (np.ones(num_sub), np.zeros(num_sub)))

        # y = np.random.permutation(y)
        if 2*(num_sub-1) < 53:
            pipe = Pipeline(
                [('svc', SVC(kernel='rbf', C=best_C, gamma=best_gamma, tol=1e-9))])
        else:
            pipe = Pipeline([('pca_apply', PCA(n_components=best_com, whiten=True)),
                             ('svc', SVC(kernel='rbf', C=best_C, gamma=best_gamma, tol=1e-9))])
        kfold = StratifiedKFold(n_splits=num_sub, shuffle=True)
        auc = cross_val_score(pipe, X, y, cv=kfold, scoring=my_metric)
        auc_sum[i] = np.mean(auc)
        #print('AUC after CV for i=%dgamma=%s number of components=%d is %g' %
        #      (i, best_gamma, best_com, np.mean(auc)))

    print('Num_Sub = %d, Average AUC=%g , Std AUC=%g' %
          (num_sub, np.mean(auc_sum), np.std(auc_sum)))
    auc_mean_sub.append(np.mean(auc_sum))
    auc_std_sub.append(np.std(auc_sum))
    auc_sub_all[num_sub, :] = auc_sum

    print(auc_mean_sub)

print(auc_mean_sub)
print(auc_std_sub)

np.savez('auc_num_sub3_tmp.npz', auc_mean_sub=auc_mean_sub,
         auc_std_sub=auc_std_sub, auc_sub_all=auc_sub_all)
