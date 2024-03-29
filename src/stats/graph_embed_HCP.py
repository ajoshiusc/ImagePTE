import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from scipy import io as spio
import pdb
##======================
import networkx as nx
from karateclub import DeepWalk, NetMF, GraphWave, Node2Vec, BoostNE
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/home/wenhuicu/ImagePTE/', help='')
parser.add_argument('--npz_root_path', type=str, default='/home/wenhuicu/data_npz/', help='')
parser.add_argument('--dim', type=int, default=16, help='feature dimension generated by deepwalk')
parser.add_argument('--walk_len', type=int, default=50, help='walk length')
parser.add_argument('--win_size', type=int, default=5, help='window size')
parser.add_argument('--walk_n', type=int, default=32, help='number of generated random walks')
parser.add_argument('--epochs', type=int, default=16, help='number of epochs during optimization in word2vec function')
parser.add_argument('--atlas', type=str, default="Brain", help='name of atlas to use.')
parser.add_argument('--model', type=str, default="BoostNE", help='Method used to generate node embeddings')
parser.add_argument('--use_all', type=bool, default=False, help='Whether to use all three features or not.')
parser.add_argument('--npz_name', type=str, default='hcp_1200_roi22')
parser.add_argument('--metric', type=list, default=['roc_auc', 'balanced_accuracy'], help='name of evaluation metric: f1, f1_micro, f1_weighted, balanced_accuracy')

parser.add_argument('--num_cv', type=int, default=5, help='number of epochs during optimization in word2vec function')
parser.add_argument('--rmv_reg', type=bool, default=False, help='Whether to remove some regions or not.')
args = parser.parse_args()


def deepwalk(conn):
    edge_list = []
    for i in range(conn.shape[1]):
        for j in range(conn.shape[1]):
            if abs(conn[i][j]) > 0.4:
                edge_list.append((i, j, abs(conn[i][j])))
        
    G = nx.Graph()
    G.add_weighted_edges_from(edge_list)
    # print(len(G))
    # pdb.set_trace()
    # train model and generate embedding
    if args.model == 'DeepWalk':
        model = DeepWalk(walk_number=args.walk_n, walk_length=args.walk_len, dimensions=args.dim, window_size=args.win_size, epochs=args.epochs)
    if args.model == 'GraphWave':
        model = GraphWave()
    if args.model == 'BoostNE':
        model = BoostNE(dimensions=args.dim, iterations=args.epochs)
    if args.model == 'NetMF':
        model = NetMF(dimensions=args.dim, iteration=args.epochs)
    model.fit(G)
    embedding = model.get_embedding()

    return embedding


def get_features():

    f = np.load(args.npz_root_path + args.npz_name + '.npz')
    conn_mat = abs(f['conn_mat'])
    print(conn_mat.shape)
    
    # n_rois = conn_mat.shape[0]
    # ind = np.tril_indices(n_rois, k=1)
    # connectivity = conn_mat[ind[0], ind[1], :].T
    
    ##=========Generate Graph Embedding Features====================
    nsub = conn_mat.shape[0]
    deepwalk_feat = []
    for subno in range(nsub):
        this_conn = conn_mat[subno, :, :]
        deepwalk_feat.append(deepwalk(this_conn))
    deepwalk_feat = np.array(deepwalk_feat)
    # deepwalk_feat = deepwalk_concat(fname)
    print(deepwalk_feat.shape)

    if args.use_all == True:
        measures = np.concatenate(
        (0.5*lesion_vols, deepwalk_feat.reshape((nsub, -1))), axis=1)
    else:
        measures = deepwalk_feat.reshape((nsub, -1))

    return measures, f['labels']

X, y = get_features()

# Permute the labels to check if AUC becomes 0.5. This check is to make sure that we are not overfitting

n_iter = args.num_cv
auc = np.zeros(n_iter)
precision = np.zeros(n_iter)
recall = np.zeros(n_iter)
fscore = np.zeros(n_iter)
support = np.zeros(n_iter)


my_metric = 'roc_auc'
# best_com = 120
best_com = 100#
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
    kfold = StratifiedKFold(n_splits=args.num_cv, shuffle=True,random_state=1211)
    # print(X.shape)
    # pdb.set_trace()
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
    kfold = StratifiedKFold(n_splits=args.num_cv, shuffle=True,random_state=1211)
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
# best_com=0
# max_AUC=0
# max_component=min((X.shape[0]-1),X.shape[1])
# for nf in range(1, max_component):
#     pipe = Pipeline([('pca_apply', PCA(n_components=nf, whiten=True)),
#                         ('svc', SVC(kernel='rbf', C=best_C,gamma=best_gamma, tol=1e-9))])
#     kfold = StratifiedKFold(n_splits=args.num_cv, shuffle=True,random_state=1211)
#     auc = cross_val_score(pipe, X, y, cv=kfold, scoring=my_metric)

#     # print('AUC after CV for nf=%dgamma=%s is %g' %
#     #         (nf, best_gamma, np.mean(auc)))
#     if np.mean(auc)>= max_AUC:
#         max_AUC=np.mean(auc)
#         best_com=nf

# print('n_components=%d is' %(best_com))

# #######################selecting gamma################
# ## Random permutation of pairs of training subject for 1000 iterations
# ####################################################
iteration_num=10
res_sum = np.zeros((iteration_num, len(args.metric)))
best_com=100
for i in range(iteration_num):
# y = np.random.permutation(y)
    pipe = Pipeline([('pca_apply', PCA(n_components=best_com, whiten=True)),
                    ('svc', SVC(kernel='rbf',C=best_C, gamma=best_gamma, tol=1e-9))])
    kfold = StratifiedKFold(n_splits=args.num_cv, shuffle=True)
    res = cross_validate(pipe, X, y, cv=kfold, scoring=args.metric)
    
    for j, m in enumerate(args.metric):
        res_sum[i, j]= np.mean(res['test_' + m])

    # print('AUC after CV for i=%dgamma=%s number of components=%d is' % (i, best_gamma, best_com))
    # print(res_sum)

    
print("Results with PCA:")
print(args.metric)
print((np.mean(res_sum, axis=0), np.std(res_sum, axis=0)))


res_sum = np.zeros((iteration_num, len(args.metric)))
for i in range(iteration_num):
# y = np.random.permutation(y)
    pipe = SVC(kernel='rbf', C=best_C,gamma=best_gamma, tol=1e-9)
    kfold = StratifiedKFold(n_splits=args.num_cv, shuffle=True)
    res = cross_validate(pipe, X, y, cv=kfold, scoring=args.metric)
    
    for j, m in enumerate(args.metric):
        res_sum[i, j]= np.mean(res['test_' + m])
    

print(args.metric)
print((np.mean(res_sum, axis=0), np.std(res_sum, axis=0)))

