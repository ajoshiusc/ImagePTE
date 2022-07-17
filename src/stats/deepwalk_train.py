import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
import pdb
##======================
import networkx as nx
from karateclub import DeepWalk, NetMF, GraphWave, Node2Vec, BoostNE
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/home/wenhuicu/ImagePTE-1/', help='')
parser.add_argument('--dim', type=int, default=64, help='feature dimension generated by deepwalk')
parser.add_argument('--walk_len', type=int, default=50, help='walk length')
parser.add_argument('--win_size', type=int, default=5, help='window size')
parser.add_argument('--walk_n', type=int, default=32, help='number of generated random walks')
parser.add_argument('--epochs', type=int, default=1, help='number of epochs during optimization in word2vec function')
parser.add_argument('--atlas', type=str, default="Brain", help='name of atlas to use.')
parser.add_argument('--model', type=str, default="DeepWalk", help='Method used to generate node embeddings')
parser.add_argument('--use_all', type=bool, default=False, help='Whether to use all three features or not.')
parser.add_argument('--npz_name', type=str, default='_graphs_USCBrain')
args = parser.parse_args()


def deepwalk(conn):
    edge_list = []
    for i in range(conn.shape[0]):
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


def deepwalk_concat(fname):
    f1 = np.load(args.root_path + fname + '_deepwalk1605_brain.npz')['deepwalk'].reshape((36, -1))
    f2 = np.load(args.root_path + fname + '_deepwalk1605_brain1.npz')['deepwalk'].reshape((36, -1))
    f3 = np.load(args.root_path + fname + '_deepwalk1605_brain2.npz')['deepwalk'].reshape((36, -1))
    embedding = np.zeros((f1.shape[0], 3, f1.shape[1]))
    embedding[:, 0, :] = f1
    embedding[:, 1, :] = f2
    embedding[:, 2, :] = f3
    mean_emb = np.mean(embedding, axis=1)
    std_emb = np.std(embedding, axis=1)

    return np.concatenate([mean_emb, std_emb], axis=1)


def get_features(fname='PTE'):
    f = np.load(args.root_path + fname + '_fmridiff_USCBrain.npz')
    sub_ids = f['sub_ids']
    label_ids = f['label_ids']
    brainsync = f['fdiff_sub'].T

    f = np.load(args.root_path + fname + args.npz_name + '.npz')
    conn_mat = f['conn_mat']

    n_rois = conn_mat.shape[0]
    ind = np.tril_indices(n_rois, k=1)
    connectivity = conn_mat[ind[0], ind[1], :].T
    
    a = np.load(args.root_path + fname + '_lesion_vols_USCBrain.npz', allow_pickle=True)
    a = a['lesion_vols'].item()
    lesion_vols = np.array([a[k] for k in sub_ids])
    nsub = len(sub_ids)
    ##=========Generate DeepWalk Features====================
    deepwalk_feat = []
    for subno in range(nsub):
        if 'Brain' in args.npz_name or 'Lobes' in args.npz_name:
            this_conn = conn_mat[:, :, subno]
        else:
            this_conn = conn_mat[subno, :, :]
        # deepwalk_feat.append(deepwalk(this_conn))
    # deepwalk_feat = np.array(deepwalk_feat)
    deepwalk_feat = deepwalk_concat(fname)
    print(deepwalk_feat.shape)

    if args.use_all == True:
        measures = np.concatenate(
        (0.2*lesion_vols, deepwalk_feat.reshape((nsub, -1)), 0.2*brainsync), axis=1)
    else:
        measures = deepwalk_feat.reshape((nsub, -1))

    return measures

epi_measures = get_features(fname='PTE')
nonepi_measures = get_features(fname='NONPTE')

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
max_AUC=0
gamma_range=[1, 0.001, 0.05, 0.075, .1, .13, .15, .17, 0.2, 0.3, .5, 1, 5, 10, 100]
for current_gamma in gamma_range:
    pipe = Pipeline([('pca_apply', PCA(n_components=best_com, whiten=True)),
                    ('svc', SVC(kernel='rbf',C=best_C, gamma=current_gamma, tol=1e-9))])
    my_metric = 'roc_auc'
    #auc = cross_val_score(clf, X, y, cv=37, scoring=my_metric)
    kfold = StratifiedKFold(n_splits=36, shuffle=True,random_state=1211)
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


auc_sum = np.zeros((iteration_num))
for i in range(iteration_num):
# y = np.random.permutation(y)
    pipe = SVC(kernel='rbf', C=best_C,gamma=best_gamma, tol=1e-9)
    kfold = StratifiedKFold(n_splits=36, shuffle=True)
    auc = cross_val_score(pipe, X, y, cv=kfold, scoring=my_metric)
    auc_sum [i]= np.mean(auc)
    #print('AUC after CV for i=%dgamma=%s is %g' %
        #(i, best_gamma, np.mean(auc)))


print('Average AUC=%g , Std AUC=%g' % (np.mean(auc_sum),np.std(auc_sum)))


