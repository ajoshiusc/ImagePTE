from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from utils import accuracy, accuracy_dc
from models import GCN
from sklearn.model_selection import StratifiedKFold
import pdb
import sklearn
import scipy.stats

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=17,
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
parser.add_argument('--model_path', type=str, default="../../../models/", help='path to saved models.')
parser.add_argument('--iters', type=int, default=10, help='number of cross_validation iterations')
parser.add_argument('--ngroup', type=int, default=2, help='number of groups')
parser.add_argument('--ngroup_olp', type=int, default=4, help='number of groups after splitting with overlapping')
parser.add_argument('--nfold', type=int, default=3, help='number of cross validation folds')
## this is dynamic connectivity with overlaps
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# adj, features, labels, idx_train, idx_val, idx_test = load_data()

##======================================================================
def calc_DAD(data):
    thr = 0.4
    adj = data['conn_mat']
    adj[adj < thr] = 0  ## threshold the weakly connected edges
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

# def calc_DAD(data):
#     thr = 0.6
#     adj = data['conn_mat']
#     adj[adj < thr] = 0 ## threshold the weakly connected edges
#     adj[adj > 0] = 1
#
#     Dl = np.sum(adj, axis=-1)
#     Dl[Dl < 2] = 0
#     num_node = adj.shape[1]
#     Dn = np.zeros((adj.shape[0], num_node, num_node))
#     for i in range(num_node):
#         Dn[:, i, i] = Dl[:, i] ** (-0.5)
#     Dn[Dn == np.inf] = 0
#
#     adj_ori = data['conn_mat']
#     adj_ori[adj_ori < thr] = 0
#     DAD = np.matmul(np.matmul(Dn, adj_ori), Dn)
#
#     return DAD


def split(feature, adj, labels):
    total_len = feature.shape[-1]
    gl = feature.shape[-1] // args.ngroup
    slide_len = gl // 4
    args.ngroup_olp = (total_len - gl) // slide_len + 1
    features_all = torch.zeros(
        [feature.shape[0] * args.ngroup_olp, feature.shape[1], gl]).float().to(device)
    # print(features_all.shape, feature.shape)
    adj_all = torch.zeros([feature.shape[0] * args.ngroup_olp, adj.shape[1], adj.shape[-1]]).float().to(device)
    labels_all = torch.zeros([feature.shape[0] * args.ngroup_olp]).long().to(device)
    adj = adj.repeat(args.ngroup_olp, 1, 1)
    labels = labels.repeat(args.ngroup_olp)

    k = 0
    for i in range(feature.shape[0]):
        q = 0
        for j in range(0, total_len-gl, slide_len):
            features_all[k, :, :] = feature[i, :, j:j + gl]
            gap = feature.shape[0]
            adj_all[k, :, :] = adj[i + q * gap, :, :]
            labels_all[k] = labels[i + q * gap]
            q = q + 1
            k = k + 1

    return features_all, adj_all, labels_all


def train(epoch, model, optimizer, scheduler, train_features, train_adj, train_labels):
    batch_size = args.batch_size
    num_train = train_features.shape[0]
    num_batches = num_train // batch_size

    aft_sp_feat, aft_sp_adj, train_labels = split(train_features, train_adj, train_labels)
    batch_size = args.ngroup_olp * batch_size
    # print(train_labels)
    # pdb.set_trace()
    for i in range(num_batches):
        features_bc = aft_sp_feat[i * batch_size:(i + 1) * batch_size, :, :]
        adj_bc = aft_sp_adj[i * batch_size:(i + 1) * batch_size, :, :]
        labels_bc = train_labels[i * batch_size:(i + 1) * batch_size]

        t = time.perf_counter()
        model.train()
        optimizer.zero_grad()
        graph_output = model(features_bc, adj_bc)

        loss_criterion = torch.nn.CrossEntropyLoss()  # this contains activation function, and calc loss
        # print(graph_output, graph_output.shape)
        loss_train = loss_criterion(graph_output, labels_bc)
        acc_train, _ = accuracy_dc(graph_output, labels_bc, args.ngroup_olp, device)
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
    print('Epoch: {:04d}'.format(epoch + 1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          # 'loss_val: {:.4f}'.format(loss_val.item()),
          # 'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.perf_counter() - t))


def test(model, test_features, test_adj, test_labels):
    model.eval()
    test_features, test_adj, test_labels = split(test_features, test_adj, test_labels)

    graph_output = model(test_features, test_adj)
    # graph_output = torch.mean(output, dim=1)
    loss_criterion = torch.nn.CrossEntropyLoss()
    loss_test = loss_criterion(graph_output, test_labels)
    probabilities = F.softmax(graph_output, dim=-1)
    prob_max = torch.zeros([probabilities.shape[0] // args.ngroup_olp, 2]).float()
    k = 0
    for i in range(0, probabilities.shape[0], args.ngroup_olp):
        prob_max[k, -1] = torch.mean(probabilities[i: i + args.ngroup_olp, -1])
        prob_max[k, 0] = probabilities[i, 0]
        k = k + 1

    acc_test, test_labels_squeezed = accuracy_dc(graph_output, test_labels, args.ngroup_olp, device)
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))
    return prob_max.cpu().detach().numpy(), acc_test.item(), test_labels_squeezed
    # return acc_test.item()


def cross_validation():
    ##=======================Load Data================================================

    # Load data

    population = 'PTE'
    # epidata = np.load(population + '_graphs_gcn.npz')
    # epidata = np.load("/home/wenhuicu/ImagePTE/pte_gcn/PRGNN/PTE_parPearson_BCI-DNI.npz")
    epidata = np.load("/home/wenhuicu/data_npz/PTE_hcf_BCI-DNI_all.npz")


    adj_epi = torch.from_numpy(calc_DAD(epidata)).float().to(device)  # n_subjects*16 *16
    features_epi = torch.from_numpy(epidata['features']).float().to(device)  # n_subjectsx16x171

    # n_subjects = features_epi.shape[0]
    # num_train = int(n_subjects * args.rate)
    # train_adj_epi = adj_epi[:num_train, :, :]
    # train_features_epi = features_epi[:num_train, :, :]
    # test_adj_epi = adj_epi[num_train:, :, :]
    # test_features_epi = features_epi[num_train:, :, :]

    population = 'NONPTE'
    # nonepidata = np.load(population + '_graphs_gcn.npz')
    # nonepidata = np.load("/home/wenhuicu/ImagePTE/pte_gcn/PRGNN/NONPTE_parPearson_BCI-DNI.npz")
    nonepidata = np.load("/home/wenhuicu/data_npz/NONPTE_hcf_BCI-DNI_all.npz")
    adj_non = torch.from_numpy(calc_DAD(nonepidata)).float().to(device)
    features_non = torch.from_numpy(nonepidata['features']).float().to(device)  # subjects x 16 x 171

    # print("DAD shape:")
    # print(adj_non.shape, adj_epi.shape)
    ## for now we are using the same number of epi , non epi training samples.
    # n_subjects_non = features_non.shape[0]
    # num_train_non = int(n_subjects_non * args.rate)
    # train_adj_non = adj_non[:num_train_non, :, :]
    # train_features_non = features_non[:num_train_non, :, :]
    # test_adj_non = adj_non[num_train_non:, :, :]
    # test_features_non = features_non[num_train_non:, :, :]

    features = torch.cat([features_epi, features_non])
    adj = torch.cat([adj_epi, adj_non])
    labels = torch.from_numpy(np.hstack((np.ones(adj_epi.shape[0]), np.zeros(adj_non.shape[0])))).long().to(device)

    print(features.shape, adj.shape, labels.shape)

    iterations = args.iters
    acc_iter = []
    auc_iter = []
    for i in range(iterations):

        kfold = StratifiedKFold(n_splits=args.nfold, shuffle=True)
        # the folds are made by preserving the percentage of samples for each class.

        acc = []
        max_epochs = []
        test_true = []
        probs_fold = []
        # epochs_choices = []
        features_numpy = features.cpu().numpy()
        labels_numpy = labels.cpu().numpy()
        adj_numpy = adj.cpu().numpy()
        for train_ind, test_ind in kfold.split(features_numpy, labels_numpy):
            # Model and optimizer
            model = GCN(nfeat=features_epi.shape[2] // args.ngroup,
                        nhid=[200, 200, 50],
                        # nhid=[200, 200, 100, 50],
                        nclass=2,  # labels.max().item() + 1,
                        dropout=args.dropout)

            state_dict = torch.load(args.model_path + "model_ADHD_pret_dc_ovlp_split2.pth")
            # print(state_dict['model_state_dict'].keys())
            # pdb.set_trace()
            for key in list(state_dict['model_state_dict'].keys()):
                if key.startswith('gc1.') or key.startswith('mlp.2'):
                # if key.startswith('gc1.'):
                    del state_dict['model_state_dict'][key]
            model.load_state_dict(state_dict["model_state_dict"], strict=False)

            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            # optimizer = optim.RMSprop(model.parameters(), lr=args.lr, alpha=0.9)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-7,
                                                                   last_epoch=-1)

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

            for epoch in range(args.epochs):
                train(epoch, model, optimizer, scheduler,
                      torch.from_numpy(train_features).float().to(device),
                      torch.from_numpy(train_adj).float().to(device),
                      torch.from_numpy(train_labels).long().to(device))
                if (epoch >= start_epoch) and (epoch % gap == 0) and (mode_on == True):
                    acc_test.append(test(model, test_features, test_adj, test_labels))
            ##===================
            test_prob, test_accur, test_labels_squeezed = test(model, test_features, test_adj, test_labels)
            acc_test.append(test_accur)
            # max_epochs.append(np.argmax(acc_test)*gap + start_epoch)
            acc.append(np.max(acc_test))
            ##=============================================
            probs_fold.append(test_prob[:, -1])
            test_true.append(test_labels_squeezed.cpu().numpy())

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

    # print(acc_iter, np.mean(acc_iter), np.std(acc_iter))
    print("----------Mean AUC-------------")
    print(auc_iter, np.mean(auc_iter), np.std(auc_iter))
    print("Accuracy")
    print(acc_iter, np.mean(acc_iter), np.std(acc_iter))


# Train model
# t_total = time.perf_counter()
# for epoch in range(args.epochs):
#     train(epoch)
#     if (epoch > 100) and (epoch % 5 == 0):
#         test()


cross_validation()

print("Optimization Finished!")
# print("Total time elapsed: {:.4f}s".format(time.perf_counter() - t_total))

# Testing
# test()
