import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import normalize
import msentropy as msen
import entropy as etp
import pdb
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--npz_name', type=str, default='_deepwalk_lobes.npz')
parser.add_argument('--nscales', type=int, default=20)
parser.add_argument('--m', type=int, default=1)
parser.add_argument('--use_rcmse', type=bool, default=True)
parser.add_argument('--use_all', type=bool, default=False)
parser.add_argument('--use_pca', type=bool, default=False)
args = parser.parse_args()

root_path = '/home/wenhuicu/ImagePTE/'

def extract_mse_features(time_series):
    n_signal = time_series.shape[1]
    nsub = time_series.shape[0]
    RCMSE = np.zeros((nsub, n_signal, args.nscales))
    for subno in range(nsub):
        for j in range(0, n_signal):
            signal = time_series[subno, j, :]

            if args.use_rcmse == False:
                RCMSE_temp = etp.multiscale_entropy(signal, args.m, tolerance=None, maxscale=args.nscales)
            else:
                RCMSE_temp = msen.rcmse(signal, args.m, 0.15 * np.std(signal), args.nscales)

            RCMSE_temp = RCMSE_temp[np.isfinite(RCMSE_temp)]
            RCMSE[subno, j, :] = RCMSE_temp #/ np.linalg.norm(RCMSE_temp)
            # for k in range(0, len(RCMSE_temp)):
            #     RCMSE[subno, k] += RCMSE_temp[k
            # print(mse, RCMSE_temp)
            # pdb.set_trace()

        # mean of the n_signal RCMSE
    return np.mean(RCMSE, axis=1), np.std(RCMSE, axis=1), RCMSE

f = np.load(root_path + 'PTE' + args.npz_name)
time_series_pte = f['features']
print(time_series_pte.shape)

mse_feat_mean, mse_feat_std, mse_feat = extract_mse_features(time_series_pte)

# print(mse_feat_mean, mse_feat_std)
# pdb.set_trace()
epi_measures = mse_feat_mean
# epi_measures = mse_feat.reshape((36, -1))
# epi_measures = normalize(np.vstack((mse_feat_mean, mse_feat_std)))
print(epi_measures)

if args.use_all:
    f = np.load(root_path + 'PTE_graphs_USCLobes.npz')
    conn_pte = f['conn_mat']
    lab_ids = f['label_ids']
    gordlab = f['labels']
    sub_ids = f['sub_ids']
    cent_mat = f['cent_mat']
    n_rois = conn_pte.shape[0]
    ind = np.tril_indices(n_rois, k=1)
    epi_connectivity = conn_pte[ind[0], ind[1], :].T

    a = np.load(root_path + 'PTE_lesion_vols_USCLobes.npz', allow_pickle=True)
    a = a['lesion_vols'].item()
    epi_lesion_vols = np.array([a[k] for k in sub_ids])
    epi_measures = np.concatenate(
        (epi_lesion_vols, .3*mse_feat), axis=1)


f = np.load(root_path + 'NONPTE' + args.npz_name)
time_series_non = f['features']

mse_feat_mean, mse_feat_std, mse_feat = extract_mse_features(time_series_non)
# nonepi_measures = mse_feat.reshape((36, -1))
# nonepi_measures = normalize(np.vstack((mse_feat_mean, mse_feat_std)))
nonepi_measures = mse_feat_mean

print(nonepi_measures)

if args.use_all:
    f = np.load(root_path + 'NONPTE_graphs_USCLobes.npz')
    conn_nonpte = f['conn_mat']
    lab_ids = f['label_ids']
    gordlab = f['labels']
    sub_ids = f['sub_ids']
    cent_mat = f['cent_mat']

    nonepi_connectivity = conn_nonpte[ind[0], ind[1], :].T

    a = np.load(root_path + 'NONPTE_lesion_vols_USCLobes.npz', allow_pickle=True)
    a = a['lesion_vols'].item()
    nonepi_lesion_vols = np.array([a[k] for k in sub_ids])
    nonepi_measures = np.concatenate(
        (nonepi_lesion_vols, .3*mse_feat), axis=1)


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
best_com = 50#
best_C= .1
#y = np.random.permutation(y)

#######################selecting gamma################
## Following part of the code do a grid search to find best value of gamma using a one fold cross validation
## the metric for comparing the performance is AUC
####################################################
if args.use_pca:
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
    # y = np.random.permutation(y)
        pipe = Pipeline([('pca_apply', PCA(n_components=best_com, whiten=True)),
                        ('svc', SVC(kernel='rbf',C=best_C, gamma=best_gamma, tol=1e-9))])
        kfold = StratifiedKFold(n_splits=36, shuffle=True)
        auc = cross_val_score(pipe, X, y, cv=kfold, scoring=my_metric)
        auc_sum [i]= np.mean(auc)
        print('AUC after CV for i=%dgamma=%s number of components=%d is %g' % (i, best_gamma,best_com, np.mean(auc)))

    print('Average AUC with PCA=%g , Std AUC=%g' % (np.mean(auc_sum),np.std(auc_sum)))

iteration_num = 100
best_gamma = 0.075
auc_sum = np.zeros((iteration_num))
for i in range(iteration_num):
# y = np.random.permutation(y)
    pipe = SVC(kernel='rbf', C=best_C, gamma=best_gamma, tol=1e-9)
    kfold = StratifiedKFold(n_splits=36, shuffle=True)
    auc = cross_val_score(pipe, X, y, cv=kfold, scoring=my_metric)
    auc_sum [i]= np.mean(auc)
    # print('AUC after CV for i=%dgamma=%s is %g' %
    #     (i, best_gamma, np.mean(auc)))


print('Average AUC=%g , Std AUC=%g' % (np.mean(auc_sum),np.std(auc_sum)))




