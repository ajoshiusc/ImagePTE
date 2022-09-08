import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import normalize
from sklearn.ensemble import RandomForestClassifier
import math
from tqdm import tqdm

measure = 'fALFF'
atlas_name = 'USCLobes'
f = np.load('PTE_'+measure+'_'+atlas_name+'.npz')
ALFF_pte = f['roiwise_data']
ALFF_pte = ALFF_pte[1:13,:]

f = np.load('PTE_fmridiff_USCLobes.npz')
conn_pte = f['fdiff_sub']
lab_ids = f['label_ids']
gordlab = f['labels']
sub_ids = f['sub_ids']
n_rois = conn_pte.shape[0]
epi_brainsync = conn_pte.T

f = np.load('PTE_graphs_USCLobes.npz')
conn_pte = f['conn_mat']
lab_ids = f['label_ids']
gordlab = f['labels']
sub_ids = f['sub_ids']
cent_mat = f['cent_mat']
n_rois = conn_pte.shape[0]
ind = np.tril_indices(n_rois, k=1)
epi_connectivity = conn_pte[ind[0], ind[1], :].T

a = np.load('PTE_lesion_vols_USCLobes.npz', allow_pickle=True)
a = a['lesion_vols'].item()
epi_lesion_vols = np.array([a[k] for k in sub_ids])

#epi_lesion_vols=normalize(epi_lesion_vols)
#epi_connectivity=normalize(epi_connectivity)
#ALFF_pte = normalize(ALFF_pte.T).T


epi_lesion_vols = epi_lesion_vols/np.linalg.norm(epi_lesion_vols)
epi_connectivity = epi_connectivity/np.linalg.norm(epi_connectivity)
ALFF_pte = ALFF_pte/np.linalg.norm(ALFF_pte)
epi_brainsync = epi_brainsync/np.linalg.norm(epi_brainsync)


#epi_measures = np.concatenate((epi_lesion_vols,epi_connectivity,ALFF_pte.T), axis=1)
epi_measures = ALFF_pte.T



measure = 'fALFF'
atlas_name = 'USCLobes'
f = np.load('NONPTE_'+measure+'_'+atlas_name+'.npz')
ALFF_nonpte = f['roiwise_data']
ALFF_nonpte = ALFF_nonpte[1:13,:]

f = np.load('NONPTE_fmridiff_USCLobes.npz')
conn_pte = f['fdiff_sub']
lab_ids = f['label_ids']
gordlab = f['labels']
sub_ids = f['sub_ids']
n_rois = conn_pte.shape[0]
nonepi_brainsync = conn_pte.T

f = np.load('NONPTE_graphs_USCLobes.npz')
conn_nonpte = f['conn_mat']
lab_ids = f['label_ids']
gordlab = f['labels']
sub_ids = f['sub_ids']
cent_mat = f['cent_mat']

nonepi_connectivity = conn_nonpte[ind[0], ind[1], :].T

a = np.load('NONPTE_lesion_vols_USCLobes.npz', allow_pickle=True)
a = a['lesion_vols'].item()
nonepi_lesion_vols = np.array([a[k] for k in sub_ids])

#nonepi_lesion_vols=normalize(nonepi_lesion_vols)
#nonepi_connectivity=normalize(nonepi_connectivity)
#ALFF_nonpte = normalize(ALFF_nonpte.T).T

nonepi_lesion_vols=nonepi_lesion_vols/np.linalg.norm(nonepi_lesion_vols)
nonepi_connectivity=nonepi_connectivity/np.linalg.norm(nonepi_connectivity)
ALFF_nonpte = ALFF_nonpte/np.linalg.norm(ALFF_nonpte)
nonepi_brainsync = nonepi_brainsync/np.linalg.norm(nonepi_brainsync)

#nonepi_measures = np.concatenate((nonepi_lesion_vols,nonepi_connectivity,ALFF_nonpte.T), axis=1)
nonepi_measures = ALFF_nonpte.T


X = np.vstack((epi_measures, nonepi_measures))

y = np.hstack(
    (np.ones(epi_measures.shape[0]), np.zeros(nonepi_measures.shape[0])))

# Permute the labels to check if AUC becomes 0.5. This check is to make sure that we are not overfitting
#y = np.random.permutation(y)


n_iter = 1000
auc = np.zeros(n_iter)
precision = np.zeros(n_iter)
recall = np.zeros(n_iter)
fscore = np.zeros(n_iter)
support = np.zeros(n_iter)


my_metric = 'roc_auc'
best_com = np.amin((50,X.shape[1]))
best_C= .1
#y = np.random.permutation(y)

#######################selecting gamma################
## Following part of the code do a grid search to find best value of gamma using a one fold cross validation
## the metric for comparing the performance is AUC
####################################################
max_AUC=0



best_c = 0
max_AUC = 0

for m_depth in range(int(math.sqrt(X.shape[1]))):
    clf = RandomForestClassifier(max_depth=m_depth+1)
    my_metric = 'roc_auc'
    #auc = cross_val_score(clf, X, y, cv=37, scoring=my_metric)
    kfold = StratifiedKFold(n_splits=36, shuffle=True, random_state=1211)
    auc = cross_val_score(clf, X, y, cv=kfold, scoring=my_metric)
    #print('AUC on testing data:gamma=%g, auc=%g' % (current_c, np.mean(auc)))
    if np.mean(auc) >= max_AUC:
        max_AUC = np.mean(auc)
        best_depth = m_depth+1

print('best_depth is %d' % best_depth)
    

#######################selecting gamma################
## Following part of the code do a grid search to find best number of PCA component
## the metric for comparing the performance is AUC
####################################################
best_com=0
max_AUC=0
max_component=min((X.shape[0]-1),X.shape[1])
for nf in range(1, max_component):
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

#best_com=53
#best_gamma=0.075
#best_C=.1
#######################selecting gamma################
## Random permutation of pairs of training subject for 1000 iterations
####################################################
iteration_num=100
auc_sum = np.zeros((iteration_num))
for i in tqdm(range(iteration_num)):
    #y = np.random.permutation(y)
    pipe = Pipeline([('pca_apply', PCA(n_components=best_com, whiten=True)),
                        ('svc', RandomForestClassifier(max_depth=best_depth))])

    kfold = StratifiedKFold(n_splits=36, shuffle=True)
    auc = cross_val_score(pipe, X, y, cv=kfold, scoring=my_metric)
    auc_sum [i]= np.mean(auc)
    #print('AUC after CV for i=%dgamma=%s number of components=%d is %g' % (i, best_gamma,best_com, np.mean(auc)))


print('Average AUC with PCA=%g , Std AUC=%g' % (np.mean(auc_sum),np.std(auc_sum)))


auc_sum = np.zeros((iteration_num))
for i in tqdm(range(iteration_num)):
    #y = np.random.permutation(y)
    pipe = RandomForestClassifier(max_depth=best_depth)
    kfold = StratifiedKFold(n_splits=36, shuffle=True)
    auc = cross_val_score(pipe, X, y, cv=kfold, scoring=my_metric)
    auc_sum [i]= np.mean(auc)
    #print('AUC after CV for i=%dgamma=%s is %g' %
        #(i, best_gamma, np.mean(auc)))


print('Average AUC=%g , Std AUC=%g' % (np.mean(auc_sum),np.std(auc_sum)))




