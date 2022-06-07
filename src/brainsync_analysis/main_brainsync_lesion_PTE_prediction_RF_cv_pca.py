import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
import math
from tqdm import tqdm

f = np.load('PTE_fmridiff.npz')
conn_pte = f['fdiff_sub_z']
lab_ids = f['label_ids']
gordlab = f['labels']
sub_ids = f['sub_ids']
n_rois = conn_pte.shape[0]
epi_connectivity = conn_pte.T

a = np.load('./stats/PTE_lesion_vols_USCBrain.npz', allow_pickle=True)
a = a['lesion_vols'].item()
epi_lesion_vols = np.array([a[k] for k in sub_ids])
#epi_measures = epi_connectivity
epi_measures = np.concatenate((epi_connectivity, epi_lesion_vols), axis=1)


f = np.load('NONPTE_fmridiff.npz')
conn_nonpte = f['fdiff_sub_z']
lab_ids = f['label_ids']
gordlab = f['labels']
sub_ids = f['sub_ids']

nonepi_connectivity = conn_nonpte.T

a = np.load('./stats/NONPTE_lesion_vols_USCBrain.npz', allow_pickle=True)
a = a['lesion_vols'].item()
nonepi_lesion_vols = np.array([a[k] for k in sub_ids])
#nonepi_measures = nonepi_connectivity
nonepi_measures = np.concatenate((nonepi_connectivity, nonepi_lesion_vols), axis=1)


X = np.vstack((epi_measures, nonepi_measures))
y = np.hstack(
    (np.ones(epi_measures.shape[0]), np.zeros(nonepi_measures.shape[0])))

# Permute the labels to check if AUC becomes 0.5. This check is to make sure that we are not overfitting

n_iter = 1000
auc = np.zeros(n_iter)
precision = np.zeros(n_iter)
recall = np.zeros(n_iter)
fscore = np.zeros(n_iter)
support = np.zeros(n_iter)


my_metric = 'roc_auc'

#y = np.random.permutation(y)

#######################selecting gamma################
## Following part of the code do a grid search to find best value of gamma using a one fold cross validation
## the metric for comparing the performance is AUC
####################################################
best_gamma=0
max_AUC=0
gamma_range=[1, 0.001, 0.05, 0.075, .1, .13, .15, .17, 0.2, 0.3, .5, 1, 5, 10, 100]
for  m_depth in tqdm(range(int(math.sqrt(X.shape[1])))):
    clf = RandomForestClassifier(max_depth=m_depth+1)
    my_metric = 'roc_auc'
    #auc = cross_val_score(clf, X, y, cv=37, scoring=my_metric)
    kfold = StratifiedKFold(n_splits=36, shuffle=True,random_state=1211)
    auc = cross_val_score(clf, X, y, cv=kfold, scoring=my_metric)
    #print('AUC on testing data:gamma=%g, auc=%g' % (current_gamma, np.mean(auc)))
    if np.mean(auc)>= max_AUC:
        max_AUC=np.mean(auc)
        best_depth = m_depth+1

print('best gamma=%g is' %(best_gamma))

#######################selecting gamma################
## Following part of the code do a grid search to find best number of PCA component
## the metric for comparing the performance is AUC
####################################################
best_com=0
max_AUC=0
max_component=min((X.shape[0]-2 - 1),(X.shape[1]-1))
for nf in tqdm(range(1, max_component)):
    pipe = Pipeline([('pca_apply', PCA(n_components=nf, whiten=True)),
                        ('svc', RandomForestClassifier(max_depth=best_depth))])
    kfold = StratifiedKFold(n_splits=36, shuffle=True,random_state=1211)
    auc = cross_val_score(pipe, X, y, cv=kfold, scoring=my_metric)

    #print('AUC after CV for nf=%dgamma=%s is %g' %
            #(nf, best_gamma, np.mean(auc)))
    if np.mean(auc)>= max_AUC:
        max_AUC=np.mean(auc)
        best_com=nf

print('n_components=%d is' %(best_com))
#######################selecting gamma################
## Random permutation of pairs of training subject for 1000 iterations
####################################################
iteration_num=1000
auc_sum = np.zeros((iteration_num))
for i in tqdm(range(iteration_num)):
# y = np.random.permutation(y)
    pipe = Pipeline([('pca_apply', PCA(n_components=best_com, whiten=True)),
                    ('svc', RandomForestClassifier(max_depth=best_depth))])
    kfold = StratifiedKFold(n_splits=36, shuffle=True)
    auc = cross_val_score(pipe, X, y, cv=kfold, scoring=my_metric)
    auc_sum [i]= np.mean(auc)
    #print('AUC after CV for i=%dgamma=%s number of components=%d is %g' %
        #(i, best_gamma,best_com, np.mean(auc)))


print('Average AUC with PCA=%g , Std AUC=%g' % (np.mean(auc_sum),np.std(auc_sum)))

auc_sum = np.zeros((iteration_num))
for i in tqdm(range(iteration_num)):
# y = np.random.permutation(y)
    pipe = RandomForestClassifier(max_depth=best_depth)
    kfold = StratifiedKFold(n_splits=36, shuffle=True)
    auc = cross_val_score(pipe, X, y, cv=kfold, scoring=my_metric)
    auc_sum [i]= np.mean(auc)
    #print('AUC after CV for i=%dgamma=%s is %g' %
        #(i, best_gamma, np.mean(auc)))


print('Average AUC without PCA=%g , Std AUC=%g' % (np.mean(auc_sum),np.std(auc_sum)))




