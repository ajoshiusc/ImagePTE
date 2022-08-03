from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from utils import accuracy
from models import GCN
from sklearn.model_selection import StratifiedKFold
import pdb
import sklearn
import scipy.stats
import networkx as nx
from karateclub import DeepWalk, NetMF, GraphWave, BoostNE

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--num_epochs', type=int, default=17,
                    help='Number of epochs to train.')
parser.add_argument('--mode', type=bool, default=False, help='select best epoch mode')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')  
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
# parser.add_argument('--hidden', type=int, default=16,
                    # help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--batch_size', type=int, default=10, help='Batch Size')
parser.add_argument('--rate', type=float, default=1.0, help='proportion of training samples')
parser.add_argument('--model_path', type=str, default="../models/", help='path to saved models.')
parser.add_argument('--iters', type=int, default=3, help='number of cross_validation iterations')
parser.add_argument('--num_cv', type=int, default=5, help='number of cross_validation')

parser.add_argument('--root_path', type=str, default='/home/wenhuicu/ImagePTE/', help='')
parser.add_argument('--dim', type=int, default=16, help='feature dimension generated by deepwalk')
parser.add_argument('--walk_len', type=int, default=50, help='walk length')
parser.add_argument('--win_size', type=int, default=5, help='window size')
parser.add_argument('--walk_n', type=int, default=32, help='number of generated random walks')
parser.add_argument('--epochs', type=int, default=16, help='number of epochs during optimization in word2vec function')
parser.add_argument('--atlas', type=str, default="Brain", help='name of atlas to use.')
parser.add_argument('--model', type=str, default="BoostNE", help='Method used to generate node embeddings')
parser.add_argument('--use_all', type=bool, default=False, help='Whether to use all three features or not.')
parser.add_argument('--npz_name', type=str, default='_graphs_USCBrain')
parser.add_argument('--weighted', type=bool, default=False, help='use weighted cross entropy or not.')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


# adj, features, labels, idx_train, idx_val, idx_test = load_data()
##======================

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


def get_features(fname='PTE'):
    f = np.load(args.root_path + fname + '_fmridiff_USCBrain.npz')
    sub_ids = f['sub_ids']
    label_ids = f['label_ids']
    brainsync = f['fdiff_sub'].T

    f = np.load(args.root_path + fname + args.npz_name + '.npz')
    conn_mat = f['conn_mat']

    # n_rois = conn_mat.shape[1]
    # ind = np.tril_indices(n_rois, k=1)
    # connectivity = conn_mat[ind[0], ind[1], :].T
    
    a = np.load(args.root_path + fname + '_lesion_vols_USCBrain.npz', allow_pickle=True)
    a = a['lesion_vols'].item()
    lesion_vols = np.array([a[k] for k in sub_ids])
    # nsub = conn_mat.shape[0]
    nsub=36
    ##=========Generate DeepWalk Features====================
    deepwalk_feat = []
    for subno in range(nsub):
        if 'Brain' in args.npz_name or 'Lobes' in args.npz_name:
            this_conn = conn_mat[:, :, subno]
        else:
            this_conn = conn_mat[subno, :, :]
        deepwalk_feat.append(deepwalk(this_conn))
    deepwalk_feat = np.array(deepwalk_feat)
    print(deepwalk_feat.shape)

    if args.use_all == True:
        measures = np.concatenate(
        (0.2*lesion_vols, deepwalk_feat, 0.2*brainsync), axis=-1)
    else:
        measures = deepwalk_feat

    return measures, np.load(args.root_path + fname + '_deepwalk.npz')

##======================================================================
def calc_DAD(data):
    thr = 0.6
    adj = abs(data['conn_mat'])
    print(adj.shape)
    adj[adj < thr] = 0 ## threshold the weakly connected edges
    adj[adj > 0] = 1

    Dl = np.sum(adj, axis=-1)

    Dl[Dl < 2] = 0
    num_node = adj.shape[1]
    Dn = np.zeros((adj.shape[0], num_node, num_node))
    for i in range(num_node):
        Dn[:, i, i] = Dl[:, i] ** (-0.5)
    Dn[Dn == np.inf] = 0

    adj_ori = abs(data['conn_mat'])
    adj_ori[adj_ori < thr] = 0
    DAD = np.matmul(np.matmul(Dn, adj_ori), Dn)

    return DAD
##=======================================================================


# if args.cuda:【
#     # model.cuda()
#     # features = features.cuda()
#     # adj = adj.cuda()
#     # labels = labels.cuda()
#     idx_train = idx_train.cuda()
#     idx_val = idx_val.cuda()
#     idx_test = idx_test.cuda()


def train(epoch, model, optimizer, scheduler, train_features, train_adj, train_labels, cls_weight=None): 
    batch_size = args.batch_size
    num_train = train_features.shape[0]
    num_batches = num_train // batch_size

    for i in range(num_batches):
        features_bc = train_features[i*batch_size:(i + 1) * batch_size, :, :]
        adj_bc = train_adj[i*batch_size:(i + 1) * batch_size, :, :]
        labels = train_labels[i*batch_size:(i + 1) * batch_size]
        
        t = time.perf_counter()
        model.train()
        optimizer.zero_grad()
        graph_output = model(features_bc, adj_bc)
        # graph_output = torch.mean(output, dim=1)
        loss_criterion = torch.nn.CrossEntropyLoss(weight=cls_weight) # this contains activation function, and calc loss
        # print(graph_output, graph_output.shape)
        loss_train = loss_criterion(graph_output, labels)
        acc_train = accuracy(graph_output, labels, num_train)
        loss_train.backward()
        optimizer.step()
        scheduler.step()

    # if not args.fastmode:
    #     # Evaluate validation set performance separately,
    #     # deactivates dropout during validation run.
    #     model.eval()
    #     output = model(features, adj)

    # loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    # acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          # 'loss_val: {:.4f}'.format(loss_val.item()),
          # 'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.perf_counter() - t))


def test(model, test_features, test_adj, test_labels):
    model.eval()
    graph_output = model(test_features, test_adj)
    # graph_output = torch.mean(output, dim=1)
    loss_criterion = torch.nn.CrossEntropyLoss()
    loss_test = loss_criterion(graph_output, test_labels)
    probabilities = F.softmax(graph_output, dim=-1)
    acc_test = accuracy(graph_output, test_labels, test_adj.shape[0])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))
    return probabilities.cpu().detach().numpy(), acc_test.item()
    # return acc_test.item()


def cross_validation():
    ##=======================Load Data================================================
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load data
    epi_measures, epidata = get_features('PTE')
    adj_epi = torch.from_numpy(calc_DAD(epidata)).float().to(device) # n_subjects*16 *16
    features_epi = torch.from_numpy(epi_measures).float().to(device) # n_subjectsx16x171

    nonepi_measures, nonepidata = get_features('NONPTE')
    adj_non = torch.from_numpy(calc_DAD(nonepidata)).float().to(device) 
    features_non = torch.from_numpy(nonepi_measures).float().to(device) #subjects x 16 x 171
    
    features = torch.cat([features_epi, features_non])
    adj = torch.cat([adj_epi, adj_non])
    labels = torch.from_numpy(np.hstack((np.ones(adj_epi.shape[0]), np.zeros(adj_non.shape[0])))).long().to(device)

    print(features.shape, adj.shape, labels.shape)

    cls_weight = None
    if args.weighted == True:
        cls_weight = torch.Tensor([1, 2.3]).to(device)
        print("weights!\n\n")

    iterations = args.iters
    acc_iter = []
    auc_iter = []
    for i in range(iterations):
        kfold = StratifiedKFold(n_splits=args.num_cv, shuffle=True)
        # the folds are made by preserving the percentage of samples for each class.
        
        acc = []
        max_epochs = []
        test_true = []
        probs_fold = []

        features_numpy = features.cpu().numpy()
        labels_numpy = labels.cpu().numpy()
        adj_numpy = adj.cpu().numpy()
        for train_ind, test_ind in kfold.split(features_numpy, labels_numpy):
            # Model and optimizer

            model = GCN(nfeat=features_epi.shape[2],
                    nhid = [200, 200, 50],
                    # nhid=[400, 400, 100],
                    nclass= 2, #labels.max().item() + 1,
                    dropout=args.dropout)
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            # optimizer = optim.RMSprop(model.parameters(), lr=args.lr, alpha=0.9)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=1e-7, last_epoch=-1)
            
            model.to(device)

            # 72 subjects in total, during CV, training has 70, testing has 2, one epi, one nonepi
            train_features = features_numpy[train_ind, :, :] 
            train_adj = adj_numpy[train_ind, :, :]
            train_labels = labels_numpy[train_ind]

            state = np.random.get_state()
            np.random.shuffle(train_features)
            np.random.set_state(state)
            np.random.shuffle(train_adj)
            np.random.set_state(state)
            np.random.shuffle(train_labels)
            
            test_features = features[test_ind, :, :]
            test_adj = adj[test_ind, :, :]
            test_labels = labels[test_ind]

            acc_test = []
            start_epoch = 13
            gap = 1
            mode_on = args.mode
            
            for epoch in range(args.num_epochs):
                train(epoch, model, optimizer, scheduler, 
                    torch.from_numpy(train_features).float().to(device), 
                    torch.from_numpy(train_adj).float().to(device), 
                    torch.from_numpy(train_labels).long().to(device), cls_weight)
                if (epoch >= start_epoch) and (epoch % gap == 0) and (mode_on == True):
                        acc_test.append(test(model, test_features, test_adj, test_labels))

            test_prob, test_accur = test(model, test_features, test_adj, test_labels)
            acc_test.append(test_accur)
            acc.append(np.max(acc_test))
            ##=============================================
            probs_fold.append(test_prob[:, -1])
            test_true.append(test_labels.cpu().numpy())


            # torch.save({'epoch': args.epochs,
            #     'model_state_dict': model.state_dict(),
            #     'optimizer_state_dict': optimizer.state_dict(),
            #     }, args.model_path+"model_cv_2002005010050" + str(len(acc)) + ".pth")

            del model
            del optimizer
            # input("any key")
        # print(acc, max_epochs)
        # with open('../results/accuracy_0.4thr_15e_5layers.txt', 'w') as f:
        #     f.write(str(acc))
        # f.close()
        probs = np.array(probs_fold).flatten()
        print(probs)
        auc = sklearn.metrics.roc_auc_score(np.array(test_true).flatten(), probs)
        print(auc)

        # print(np.mean(acc))
        acc_iter.append(np.mean(acc))
        auc_iter.append(auc)

    print("----------Mean AUC-------------")
    print(auc_iter, np.mean(auc_iter), np.std(auc_iter))
    print("----------Accuracy-------------")
    print(acc_iter, np.mean(acc_iter), np.std(acc_iter))

cross_validation()

print("Optimization Finished!")
# print("Total time elapsed: {:.4f}s".format(time.perf_counter() - t_total))

# Testing
# test()
