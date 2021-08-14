from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from utils import accuracy
from models_hcf import GCN
from sklearn.model_selection import StratifiedKFold
import sklearn
import pdb
import scipy.stats

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=15,
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

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


# adj, features, labels, idx_train, idx_val, idx_test = load_data()

##======================================================================
def calc_DAD(data):
    thr = 0.4
    adj = data['conn_mat']
    adj[adj < thr] = 0 ## threshold the weakly connected edges
    adj[adj > 0] = 1
    Dl = np.sum(adj, axis=-1)
    num_node = adj.shape[1]
    Dn = np.zeros((adj.shape[0], num_node, num_node))
    for i in range(num_node):
        Dn[:, i, i] = Dl[:, i] ** (-0.5)

    adj_ori = data['conn_mat']
    adj_ori[adj_ori < thr] = 0
    DAD = np.matmul(np.matmul(Dn, adj_ori), Dn)

    return DAD
##=======================================================================


# if args.cuda:
#     # model.cuda()
#     # features = features.cuda()
#     # adj = adj.cuda()
#     # labels = labels.cuda()
#     idx_train = idx_train.cuda()
#     idx_val = idx_val.cuda()
#     idx_test = idx_test.cuda()


def train(epoch, model, optimizer, scheduler, train_features, train_adj, train_labels): 
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
        loss_criterion = torch.nn.CrossEntropyLoss() # this contains activation function, and calc loss
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
    return probabilities.cpu().detach().numpy()
    # return acc_test.item()


def boxcox_transform(features):
    transformed_feature = np.zeros_like(features)

    for sub_id in range(features.shape[0]):
        mean_feature = features[sub_id, :, 0]
        transformed_feature[sub_id, :, 0] = (mean_feature - np.mean(mean_feature)) / np.std(mean_feature)
        for f_id in range(1, features.shape[2]):
            transformed_feature[sub_id, :, f_id], optm_lamda = scipy.stats.boxcox(features[sub_id, :, f_id])
    # print(transformed_feature, optm_lamda)
    # pdb.set_trace()
    return transformed_feature


def normalize(features):
    transformed_feature = np.zeros_like(features)
    # mean_feature = features[:, :, 0]
    # transformed_feature[:, :, 0] = (mean_feature - np.mean(mean_feature)) / np.std(mean_feature)
    for f_id in range(features.shape[2]):
        for sub_id in range(features.shape[0]):
            temp = features[sub_id, :, f_id]
            transformed_feature[sub_id, :, f_id] = (temp - np.mean(temp)) / np.std(temp)
    # print(transformed_feature)
    # pdb.set_trace()
    return transformed_feature



def engineer_features(data):
    adj_temp = data['conn_mat']
    # print(len(adj_temp[adj_temp < 0.6]), len(adj_temp[adj_temp<0.5]), len(adj_temp[adj_temp<0.4]))
    
    threshold = 0.4
    adj_temp[adj_temp < threshold] = 0
    adj_temp[adj_temp > 0] = 1
    features = np.zeros((adj_temp.shape[0], adj_temp.shape[1], 3)) # n_subjects x 16 x num_features

    features[:, :, 0] = np.mean(data['features'], axis=-1, keepdims=False)
    features[:, :, 1] = np.std(data['features'], axis=-1, keepdims=False)
    features[:, :, 2] = np.sum(adj_temp, axis=-1, keepdims=False) # degree
    # features[:, :, 3] = data['cent_coords'] # central coordinates of ROI
    # print(normalize(features))
    # pdb.set_trace()
    # return boxcox_transform(features)
    return normalize(features)
    # pdb.set_trace()
    # return features


def cross_validation():
    ##=======================Load Data================================================
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load data

    population = 'PTE'
    epidata = np.load(population+'_graphs_gcn_hcf_BCI-DNI.npz')
    adj_epi = torch.from_numpy(calc_DAD(epidata)).float().to(device) # n_subjects*16 *16
    features_epi = torch.from_numpy(engineer_features(epidata)).float().to(device) # n_subjectsx16x171

    population = 'NONPTE'
    nonepidata = np.load(population+'_graphs_gcn_hcf_BCI-DNI.npz')
    adj_non = torch.from_numpy(calc_DAD(nonepidata)).float().to(device) 
    features_non = torch.from_numpy(engineer_features(nonepidata)).float().to(device) #subjects x 16 x 4

    features = torch.cat([features_epi, features_non])
    adj = torch.cat([adj_epi, adj_non])
    labels = torch.from_numpy(np.hstack((np.ones(adj_epi.shape[0]), np.zeros(adj_non.shape[0])))).long().to(device)

    print(features.shape, adj.shape, labels.shape)
    # pdb.set_trace()
    iterations = args.iters
    acc_iter = []
    auc_iter = []
    for i in range(iterations):

        kfold = StratifiedKFold(n_splits=36, shuffle=True)
        # the folds are made by preserving the percentage of samples for each class.
        acc = []
        max_epochs = []
        test_true = []
        # epochs_choices = []
        features_numpy = features.cpu().numpy()
        labels_numpy = labels.cpu().numpy()
        adj_numpy = adj.cpu().numpy()

        for train_ind, test_ind in kfold.split(features_numpy, labels_numpy):
            # Model and optimizer

            model = GCN(nfeat=features_epi.shape[2],
                    nhid = [32, 16, 8],
                    # nhid=[200, 200, 100, 50],
                    nclass= 2, #labels.max().item() + 1,
                    dropout=args.dropout)
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            # optimizer = optim.RMSprop(model.parameters(), lr=args.lr, alpha=0.9)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-7, last_epoch=-1)
            
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
            start_epoch = 20
            gap = 1
            mode_on = args.mode
            
            for epoch in range(args.epochs):
                train(epoch, model, optimizer, scheduler, 
                    torch.from_numpy(train_features).float().to(device), 
                    torch.from_numpy(train_adj).float().to(device), 
                    torch.from_numpy(train_labels).long().to(device))
                if (epoch >= start_epoch) and (epoch % gap == 0) and (mode_on == True):
                        acc_test.append(test(model, test_features, test_adj, test_labels))

            # acc_test.append(test(model, test_features, test_adj, test_labels))

            # max_epochs.append(np.argmax(acc_test)*gap + start_epoch)
            # acc.append(np.max(acc_test))
            acc.append(test(model, test_features, test_adj, test_labels)[:, -1])
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
        probs = np.array(acc).flatten()
        print(probs)
        auc = sklearn.metrics.roc_auc_score(np.array(test_true).flatten(), probs)
        print(auc)
        # acc_iter.append(np.mean(acc))
        auc_iter.append(auc)

    # print(acc_iter, np.mean(acc_iter), np.std(acc_iter))
    print(auc_iter, np.mean(auc_iter), np.std(auc_iter))

cross_validation()

print("Optimization Finished!")
# print("Total time elapsed: {:.4f}s".format(time.perf_counter() - t_total))

# Testing
# test()
