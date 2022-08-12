import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold

measure = 'fALFF'
atlas_name = 'USCBrain'
f = np.load('PTE_'+measure+'_'+atlas_name+'.npz')
sub_ids = f['sub_ids']
ALFF_pte = f['roiwise_data']
ALFF_pte = dict(zip(sub_ids,ALFF_pte[1:13,:].T))


f = np.load('PTE_fmridiff_USCBrain.npz')
conn_pte = f['fdiff_sub']
lab_ids = f['label_ids']
gordlab = f['labels']
sub_ids = f['sub_ids']
n_rois = conn_pte.shape[0]
#epi_brainsync = conn_pte.T
epi_brainsync=dict(zip(sub_ids,conn_pte.T))


f = np.load('PTE_graphs_USCBrain.npz')
conn_pte = f['conn_mat']
lab_ids = f['label_ids']
gordlab = f['labels']
sub_ids = f['sub_ids']
cent_mat = f['cent_mat']
n_rois = conn_pte.shape[0]
ind = np.tril_indices(n_rois, k=1)
epi_connectivity = conn_pte[ind[0], ind[1], :].T

a = np.load('PTE_lesion_vols_USCBrain.npz', allow_pickle=True)
a = a['lesion_vols'].item()
epi_lesion_vols = np.array([a[k] for k in sub_ids])
#epi_measures = np.concatenate((ALFF_pte.T, 0*epi_lesion_vols,epi_connectivity,0*epi_brainsync), axis=1)
epi_brainsync = np.array([epi_brainsync[k] for k in sub_ids])
ALFF_pte = np.array([ALFF_pte[k] for k in sub_ids])
epi_measures = np.concatenate((ALFF_pte, epi_brainsync,epi_connectivity,epi_lesion_vols), axis=1)
#epi_measures = epi_connectivity


measure = 'fALFF'
atlas_name = 'USCBrain'
f = np.load('NONPTE_'+measure+'_'+atlas_name+'.npz')
ALFF_nonpte = f['roiwise_data']
sub_ids = f['sub_ids']
#ALFF_nonpte = ALFF_nonpte[1:13,:]
ALFF_nonpte = dict(zip(sub_ids,ALFF_nonpte[1:13,:].T))

f = np.load('NONPTE_fmridiff_USCBrain.npz')
conn_pte = f['fdiff_sub']
lab_ids = f['label_ids']
gordlab = f['labels']
sub_ids = f['sub_ids']
n_rois = conn_pte.shape[0]
#nonepi_brainsync = conn_pte.T
nonepi_brainsync=dict(zip(sub_ids,conn_pte.T))


f = np.load('NONPTE_graphs_USCBrain.npz')
conn_nonpte = f['conn_mat']
lab_ids = f['label_ids']
gordlab = f['labels']
sub_ids = f['sub_ids']
cent_mat = f['cent_mat']

nonepi_connectivity = conn_nonpte[ind[0], ind[1], :].T

a = np.load('NONPTE_lesion_vols_USCBrain.npz', allow_pickle=True)
a = a['lesion_vols'].item()
nonepi_lesion_vols = np.array([a[k] for k in sub_ids])
#nonepi_measures = np.concatenate((ALFF_nonpte.T,.0*nonepi_lesion_vols,nonepi_connectivity,.0*nonepi_brainsync), axis=1)
nonepi_brainsync = np.array([nonepi_brainsync[k] for k in sub_ids])
ALFF_nonpte = np.array([ALFF_nonpte[k] for k in sub_ids])
nonepi_measures = np.concatenate((ALFF_nonpte, nonepi_brainsync,nonepi_connectivity,nonepi_lesion_vols), axis=1)
#nonepi_measures = nonepi_connectivity


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
best_com = np.min((50,X.shape[1]-1))
best_C= .1
#y = np.random.permutation(y)

#######################selecting gamma################
## Following part of the code do a grid search to find best value of gamma using a one fold cross validation
## the metric for comparing the performance is AUC
####################################################
max_AUC=0
gamma_range=[1, 0.001, 0.05, 0.075, .1, .13, .15, .17, 0.2, 0.3, .5, 1, 5, 10, 100]
for current_gamma in gamma_range:
    pipe = Pipeline([('pca_apply', PCA(n_components=best_com, whiten=True)),
                    ('svc', SVC(kernel='rbf',C=best_C, gamma=current_gamma, tol=1e-9))])
    my_metric = 'roc_auc'
    #auc = cross_val_score(clf, X, y, cv=37, scoring=my_metric)
    kfold = StratifiedKFold(n_splits=36, shuffle=True,random_state=1211)
    auc = cross_val_score(pipe, X, y, cv=kfold, scoring=my_metric)
    #print('AUC on testing data:gamma=%g, auc=%g' % (current_gamma, np.mean(auc)))
    if np.mean(auc)>= max_AUC:
        max_AUC=np.mean(auc)
        best_gamma=current_gamma

print('best gamma=%g is' %(best_gamma))

C_range=[0.0001, 0.001, 0.01, .1, .3, .6, 0.7,0.9, 1, 1.5, 2, 3, 4, 5, 6, 7, 9, 10, 100]  
for current_C in C_range:
    pipe = Pipeline([('pca_apply', PCA(n_components=best_com, whiten=True)),
                    ('svc', SVC(kernel='rbf',C=current_C, gamma=best_gamma, tol=1e-9))])
    my_metric = 'roc_auc'
    #auc = cross_val_score(clf, X, y, cv=37, scoring=my_metric)
    kfold = StratifiedKFold(n_splits=36, shuffle=True,random_state=1211)
    auc = cross_val_score(pipe, X, y, cv=kfold, scoring=my_metric)
    #print('AUC on testing data:gamma=%g, auc=%g' % (current_gamma, np.mean(auc)))
    if np.mean(auc)>= max_AUC:
        max_AUC=np.mean(auc)
        best_C=current_C

print('best C=%g is' %(best_C))

'''
for mygamma in ['auto', 'scale']:
clf = SVC(kernel='rbf', gamma=mygamma, tol=1e-9)
my_metric = 'roc_auc'
kfold = StratifiedKFold(n_splits=37, shuffle=False)
auc = cross_val_score(clf, X, y, cv=kfold, scoring=my_metric)
print('AUC on testing data:gamma=%s, auc=%g' % (mygamma, np.mean(auc)))
'''
#######################selecting gamma################
## Following part of the code do a grid search to find best number of PCA component
## the metric for comparing the performance is AUC
####################################################
best_com=0
max_AUC=0
max_component=min((X.shape[0]-1),X.shape[1])
for nf in range(1, max_component):
    pipe = Pipeline([('pca_apply', PCA(n_components=nf, whiten=True)),
                        ('svc', SVC(kernel='rbf', C=best_C,gamma=best_gamma, tol=1e-9))])
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
for i in range(iteration_num):
    #y = np.random.permutation(y)
    pipe = Pipeline([('pca_apply', PCA(n_components=best_com, whiten=True)),
                    ('svc', SVC(kernel='rbf',C=best_C, gamma=best_gamma, tol=1e-9))])
    kfold = StratifiedKFold(n_splits=36, shuffle=True)
    auc = cross_val_score(pipe, X, y, cv=kfold, scoring=my_metric)
    auc_sum [i]= np.mean(auc)
    print('AUC after CV for i=%dgamma=%s number of components=%d is %g' % (i, best_gamma,best_com, np.mean(auc)))


print('Average AUC with PCA=%g , Std AUC=%g' % (np.mean(auc_sum),np.std(auc_sum)))


auc_sum = np.zeros((iteration_num))
for i in range(iteration_num):
    #y = np.random.permutation(y)
    pipe = SVC(kernel='rbf', C=best_C,gamma=best_gamma, tol=1e-9)
    kfold = StratifiedKFold(n_splits=36, shuffle=True)
    auc = cross_val_score(pipe, X, y, cv=kfold, scoring=my_metric)
    auc_sum [i]= np.mean(auc)
    #print('AUC after CV for i=%dgamma=%s is %g' %
        #(i, best_gamma, np.mean(auc)))


print('Average AUC=%g , Std AUC=%g' % (np.mean(auc_sum),np.std(auc_sum)))




