"""

@author: Anand, Haleh
"""

'''This code used 11 lobe lesion volumes generated with VAE to predict epileptic subjects in a TBI population
36 subject in PTE class and 36 subjects in non PTE class
'''
import numpy as np
import nilearn.image as ni
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
import math

def main():

    #######################load data################
    f = np.load('../connectivity/PTE_graphs.npz')
    conn_pte = f['conn_mat']
    lab_ids = f['label_ids']
    gordlab = f['labels']
    sub_ids = f['sub_ids']

    n_rois = conn_pte.shape[0]
    ind = np.tril_indices(n_rois, k=1)
  

    a = np.load('PTE_lesion_vols.npz', allow_pickle=True)
    a = a['lesion_vols'].item()
    epi_lesion_vols = np.array([a[k] for k in sub_ids])
    epi_measures = epi_lesion_vols


    f = np.load('../connectivity/NONPTE_graphs.npz')
    conn_nonpte = f['conn_mat']
    lab_ids = f['label_ids']
    gordlab = f['labels']
    sub_ids = f['sub_ids']







    a = np.load('NONPTE_lesion_vols.npz', allow_pickle=True)
    a = a['lesion_vols'].item()
    nonepi_lesion_vols = np.array([a[k] for k in sub_ids])
    nonepi_measures = nonepi_lesion_vols


    X = np.vstack((epi_measures, nonepi_measures))
    y = np.hstack(
        (np.ones(epi_measures.shape[0]), np.zeros(nonepi_measures.shape[0])))


#######################selecting gamma################
## Following part of the code do a grid search to find best value of gamma using a one fold cross validation
## the metric for comparing the performance is AUC
####################################################
    best_c=0
    max_AUC=0
    
    for m_depth in range(int(math.sqrt(X.shape[1]))):
        clf = RandomForestClassifier(max_depth=m_depth+1)
        my_metric = 'roc_auc'
        #auc = cross_val_score(clf, X, y, cv=37, scoring=my_metric)
        kfold = StratifiedKFold(n_splits=36, shuffle=False)
        auc = cross_val_score(clf, X, y, cv=kfold, scoring=my_metric)
        #print('AUC on testing data:gamma=%g, auc=%g' % (current_c, np.mean(auc)))
        if np.mean(auc)>= max_AUC:
            max_AUC=np.mean(auc)
            best_depth=m_depth+1

    

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
                         ('svc', RandomForestClassifier(max_depth=best_depth))])
        kfold = StratifiedKFold(n_splits=36, shuffle=False)
        auc = cross_val_score(pipe, X, y, cv=kfold, scoring=my_metric)

        #print('AUC after CV for nf=%dgamma=%s is %g' %
              #(nf, best_c, np.mean(auc)))
        if np.mean(auc)>= max_AUC:
            max_AUC=np.mean(auc)
            best_com=nf
#######################selecting gamma################
## Random permutation of pairs of training subject for 1000 iterations
####################################################
    iteration_num=100
    auc_sum = np.zeros((iteration_num))
    for i in range(iteration_num):
    # y = np.random.permutation(y)
        pipe = Pipeline([('pca_apply', PCA(n_components=best_com, whiten=True)),
                     ('svc', RandomForestClassifier(max_depth=best_depth))])
        kfold = StratifiedKFold(n_splits=36, shuffle=True)
        auc = cross_val_score(pipe, X, y, cv=kfold, scoring=my_metric)
        auc_sum [i]= np.mean(auc)
        #print('AUC after CV for i=%dgamma=%s is %g' %
            #(i, best_gamma, np.mean(auc)))


    print('Average AUC=%g , Std AUC=%g' % (np.mean(auc_sum),np.std(auc_sum)))

    for i in range(iteration_num):
    # y = np.random.permutation(y)
        pipe = RandomForestClassifier(max_depth=best_depth)
        kfold = StratifiedKFold(n_splits=36, shuffle=True)
        auc = cross_val_score(pipe, X, y, cv=kfold, scoring=my_metric)
        auc_sum [i]= np.mean(auc)
        #print('AUC after CV for i=%dgamma=%s is %g' %
            #(i, best_gamma, np.mean(auc)))


    print('Average AUC=%g , Std AUC=%g' % np.mean(auc_sum),np.std(auc_sum))


if __name__ == "__main__":
    main()
