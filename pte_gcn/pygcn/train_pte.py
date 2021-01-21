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

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=110,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
# parser.add_argument('--hidden', type=int, default=16,
                    # help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--batch_size', type=int, default=20, help='Batch Size')
parser.add_argument('--rate', type=float, default=1.0, help='proportion of training samples')
parser.add_argument('--model_path', type=str, default="../models/", help='path to saved models.')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


# adj, features, labels, idx_train, idx_val, idx_test = load_data()

##======================================================================
def calc_DAD(data):
    adj = data['conn_mat']
    adj[adj > 0] = 1
    Dl = np.sum(adj, axis=-1)
    num_node = adj.shape[1]
    Dn = np.zeros((adj.shape[0], num_node, num_node))
    for i in range(num_node):
        Dn[:, i, i] = Dl[:, i] ** (-0.5)
    DAD = np.matmul(np.matmul(Dn, data['conn_mat']), Dn)

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
    index = torch.randperm(num_train)[:batch_size]
    # index_epi = torch.randperm(num_train)[:batch_size//2]
    # index_non = torch.randperm(num_train_non)[:batch_size//2]
    # features_epi_bc = train_features_epi[index_epi, :, :]
    # features_non_bc = train_features_non[index_non, :, :]
    # features_bc = torch.cat([features_epi_bc, features_non_bc])
    
    # adj_epi_bc = train_adj_epi[index_epi, :, :]
    # adj_non_bc = train_adj_non[index_non, :, :]
    # adj_bc = torch.cat([adj_epi_bc, adj_non_bc])
    # labels = torch.from_numpy(np.hstack((np.ones(batch_size//2), np.zeros(batch_size//2)))).long().to(device)
    features_bc = train_features[index, :, :]
    adj_bc = train_adj[index, :, :]
    labels = train_labels[index]

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
    acc_test = accuracy(graph_output, test_labels, test_adj.shape[0])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))
    return acc_test.item()


def cross_validation():
    ##=======================Load Data================================================
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load data

    population = 'PTE'
    epidata = np.load(population+'_graphs_gcn.npz')
    adj_epi = torch.from_numpy(calc_DAD(epidata)).float().to(device) # n_subjects*16 *16
    features_epi = torch.from_numpy(epidata['features']).float().to(device) # n_subjectsx16x171

    # n_subjects = features_epi.shape[0]
    # num_train = int(n_subjects * args.rate)
    # train_adj_epi = adj_epi[:num_train, :, :]
    # train_features_epi = features_epi[:num_train, :, :]
    # test_adj_epi = adj_epi[num_train:, :, :]
    # test_features_epi = features_epi[num_train:, :, :]

    population = 'NONPTE'
    nonepidata = np.load(population+'_graphs_gcn.npz')
    adj_non = torch.from_numpy(calc_DAD(nonepidata)).float().to(device) 
    features_non = torch.from_numpy(nonepidata['features']).float().to(device) #subjects x 16 x 171

    # print("DAD shape:")
    # print(adj_non.shape, adj_epi.shape)
    ## for now we are using the same number of epi , non epi training samples.
    # n_subjects_non = features_non.shape[0]
    # num_train_non = int(n_subjects_non * args.rate)
    # train_adj_non = adj_non[:num_train_non, :, :]
    # train_features_non = features_non[:num_train_non, :, :]
    # test_adj_non = adj_non[num_train_non:, :, :]
    # test_features_non = features_non[num_train_non:, :, :]

    print(features_non.shape, features_epi.shape, adj_epi.shape)

    features = torch.cat([features_epi, features_non])
    adj = torch.cat([adj_epi, adj_non])
    labels = torch.from_numpy(np.hstack((np.ones(adj_epi.shape[0]), np.zeros(adj_non.shape[0])))).long().to(device)

    kfold = StratifiedKFold(n_splits=36, shuffle=False)
    # the folds are made by preserving the percentage of samples for each class.
    
    acc = []
    for train_ind, test_ind in kfold.split(features.cpu().numpy(), labels.cpu().numpy()):
        # Model and optimizer

        model = GCN(nfeat=features_epi.shape[2],
                nhid=[200, 200, 100, 50],
                nclass= 2, #labels.max().item() + 1,
                dropout=args.dropout)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-7, last_epoch=-1)
        
        model.to(device)


        # 72 subjects in total, during CV, training has 70, testing has 2, one epi, one nonepi
        train_features = features[train_ind, :, :] 
        train_adj = adj[train_ind, :, :]
        train_labels = labels[train_ind]
        
        test_features = features[test_ind, :, :]
        test_adj = adj[test_ind, :, :]
        test_labels = labels[test_ind]

        for epoch in range(args.epochs):
            train(epoch, model, optimizer, scheduler, train_features, train_adj, train_labels)

        acc.append(test(model, test_features, test_adj, test_labels))

        torch.save({'epoch': args.epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, args.model_path+"model_cv_" + str(len(acc)) + ".pth")

        del model
        del optimizer
        # input("any key")

    with open('../results/accuracy.txt', 'w') as f:
        f.write(str(acc))
    f.close()

    print(len(acc), np.mean(acc))

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
