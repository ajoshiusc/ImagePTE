import os
import numpy as np
import argparse
import time
import copy

import torch
import torch.nn.functional as F
from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter

from imports.ABIDEDataset import ABIDEDataset
from torch_geometric.data import DataLoader
from net.braingnn import Network
from imports.utils import train_val_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
import pdb

torch.manual_seed(123)

EPS = 1e-10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=10, help='number of epochs of training')
parser.add_argument('--batchSize', type=int, default=30, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='../../../data_BG/allNONPTE1/', help='root directory of the dataset')
parser.add_argument('--fold', type=int, default=0, help='training which fold')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--stepsize', type=int, default=2, help='scheduler step size')
parser.add_argument('--gamma', type=float, default=0.5, help='scheduler shrinking rate')
parser.add_argument('--weightdecay', type=float, default=5e-3, help='regularization')
parser.add_argument('--lamb0', type=float, default=1, help='classification loss weight')
parser.add_argument('--lamb1', type=float, default=0, help='s1 unit regularization')
parser.add_argument('--lamb2', type=float, default=0, help='s2 unit regularization')
parser.add_argument('--lamb3', type=float, default=0.5, help='s1 entropy regularization')
parser.add_argument('--lamb4', type=float, default=0.5, help='s2 entropy regularization')
parser.add_argument('--lamb5', type=float, default=0, help='s1 consistence regularization')
parser.add_argument('--layer', type=int, default=2, help='number of GNN layers')
parser.add_argument('--ratio', type=float, default=0.4, help='pooling ratio')
parser.add_argument('--indim', type=int, default=93, help='feature dim')
parser.add_argument('--nroi', type=int, default=93, help='num of ROIs')
parser.add_argument('--nclass', type=int, default=2, help='num of classes')
parser.add_argument('--load_model', type=bool, default=False)
parser.add_argument('--save_model', type=bool, default=False)
parser.add_argument('--optim', type=str, default='Adam', help='optimization method: SGD, Adam')
parser.add_argument('--save_path', type=str, default='./model/', help='path to save model')
parser.add_argument('--num_folds', type=int, default=3, help='number of cross validation folds')
parser.add_argument('--iters', type=int, default=3, help='iterations of cross validation')
parser.add_argument('--rep', type=int, default=5, help='augmentation times')
parser.add_argument('--aug', type=bool, default=False, help='w/o augmentation')
parser.add_argument('--normalization', type=bool, default=True, help='normalize features')
opt = parser.parse_args()

if not os.path.exists(opt.save_path):
    os.makedirs(opt.save_path)

############################### Define Other Loss Functions ########################################
def topk_loss(s,ratio):
    if ratio > 0.5:
        ratio = 1-ratio
    s = s.sort(dim=1).values
    res = -torch.log(s[:, -int(s.size(1)*ratio):]+EPS).mean() -torch.log(1-s[:, :int(s.size(1)*ratio)]+EPS).mean()
    return res


def consist_loss(s):
    if len(s) == 0:
        return 0
    s = torch.sigmoid(s)
    W = torch.ones(s.shape[0], s.shape[0])
    D = torch.eye(s.shape[0])*torch.sum(W, dim=1)
    L = D-W
    L = L.to(device)
    res = torch.trace(torch.transpose(s, 0, 1) @ L @ s)/(s.shape[0]*s.shape[0])
    return res

###################### Network Training Function#####################################
def normal_transform_train(x):
    #xt, lamb = stats.boxcox(x - torch.min(x) + 1)
    lamb = 0
    #xt_torch = xt
    xt_mean = torch.mean(x).float()
    xt_std = torch.std(x).float()
    xt_norm = (x - xt_mean)/xt_std
    return xt_norm, lamb, xt_mean, xt_std

def normal_transform_test(x,lamb, xt_mean, xt_std):
    res = (x-xt_mean)/xt_std
    return res


def train(epoch):
    print('train...........')

    for param_group in optimizer.param_groups:
        print("LR", param_group['lr'])
    model.train()
    s1_list = []
    s2_list = []
    loss_all = 0
    step = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output, w1, w2, s1, s2 = model(data.x, data.edge_index, data.batch, data.edge_attr, data.pos)
        s1_list.append(s1.view(-1).detach().cpu().numpy())
        s2_list.append(s2.view(-1).detach().cpu().numpy())

        loss_c = F.nll_loss(output, data.y)

        loss_p1 = (torch.norm(w1, p=2)-1) ** 2
        loss_p2 = (torch.norm(w2, p=2)-1) ** 2
        loss_tpk1 = topk_loss(s1, opt.ratio)
        loss_tpk2 = topk_loss(s2, opt.ratio)
        loss_consist = 0
        for c in range(opt.nclass):
            loss_consist += consist_loss(s1[data.y == c])
        loss = opt.lamb0*loss_c + opt.lamb1 * loss_p1 + opt.lamb2 * loss_p2 \
                   + opt.lamb3 * loss_tpk1 + opt.lamb4 * loss_tpk2 + opt.lamb5 * loss_consist

        writer.add_scalar('train/classification_loss', loss_c, epoch*len(train_loader)+step)
        writer.add_scalar('train/unit_loss1', loss_p1, epoch*len(train_loader)+step)
        writer.add_scalar('train/unit_loss2', loss_p2, epoch*len(train_loader)+step)
        writer.add_scalar('train/TopK_loss1', loss_tpk1, epoch*len(train_loader)+step)
        writer.add_scalar('train/TopK_loss2', loss_tpk2, epoch*len(train_loader)+step)
        writer.add_scalar('train/GCL_loss', loss_consist, epoch*len(train_loader)+step)
        step = step + 1

        loss.backward()
        loss_all += loss.item() * data.num_graphs
        optimizer.step()

        s1_arr = np.hstack(s1_list)
        s2_arr = np.hstack(s2_list)
    scheduler.step()
    # if epoch == 15:
    #     print(s1_list, s2_list)
    #     pdb.set_trace()
    return loss_all / len(train_dataset), s1_arr, s2_arr, w1, w2


###################### Network Testing Function#####################################
def test_acc(loader):
    model.eval()
    correct = 0
    for data in loader:
        data = data.to(device)
        outputs=model(data.x, data.edge_index, data.batch, data.edge_attr, data.pos)
        pred = outputs[0].max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()

    return correct / len(loader.dataset)

def test_loss(loader,epoch):
    print('testing...........')
    model.eval()
    loss_all = 0
    for data in loader:
        data = data.to(device)
        output, w1, w2, s1, s2=model(data.x, data.edge_index, data.batch, data.edge_attr, data.pos)
        loss_c = F.nll_loss(output, data.y)

        loss_p1 = (torch.norm(w1, p=2)-1) ** 2
        loss_p2 = (torch.norm(w2, p=2)-1) ** 2
        loss_tpk1 = topk_loss(s1,opt.ratio)
        loss_tpk2 = topk_loss(s2,opt.ratio)
        loss_consist = 0
        for c in range(opt.nclass):
            loss_consist += consist_loss(s1[data.y == c])
        loss = opt.lamb0*loss_c + opt.lamb1 * loss_p1 + opt.lamb2 * loss_p2 \
                   + opt.lamb3 * loss_tpk1 + opt.lamb4 *loss_tpk2 + opt.lamb5* loss_consist

        loss_all += loss.item() * data.num_graphs
    return loss_all / len(loader.dataset)

#######################################################################################
############################   Model Training #########################################
#######################################################################################

#################### Parameter Initialization #######################

name = 'PTE_data'
path = opt.dataroot
save_model = opt.save_model
load_model = opt.load_model
opt_method = opt.optim
num_epoch = opt.n_epochs
fold = opt.fold
writer = SummaryWriter(os.path.join('./log', str(fold)))
# writer = SummaryWriter(os.path.join('./log/{}_fold{}_consis{}'.format(opt.net, opt.fold, opt.lamb5)))

############# Define Dataloader -- need costumize#####################
file_pos = "/home/wenhuicu/ImagePTE/pte_gcn/PRGNN/PTE_parPearson_BCI-DNI.npz"
file_neg = "/home/wenhuicu/ImagePTE/pte_gcn/PRGNN/NONPTE_parPearson_BCI-DNI.npz"
dataset = ABIDEDataset(path, name)
dataset.data.y = dataset.data.y.squeeze()
dataset.data.x[dataset.data.x == float('inf')] = 0

############### split train, val, and test set -- need costumize########################
# tr_index, te_index, val_index = train_val_test_split(mat_dir=opt.matroot, fold=opt.fold, rep=opt.rep)
ADHD_data = np.load(file_pos, encoding='bytes', allow_pickle=True)
TDC_data = np.load(file_neg, encoding='bytes', allow_pickle=True)
adhd_features = ADHD_data['conn_mat']
tdc_features = TDC_data['conn_mat']
x_features = np.concatenate([adhd_features, tdc_features], axis=0)
labels = np.hstack((np.ones(adhd_features.shape[0]), np.zeros(tdc_features.shape[0])))

print(x_features.shape, labels.shape)

# tr_index,val_index,te_index = train_val_test_split(fold=fold)



best_loss = 1e10
###############cross validation############################
acc_iter = []
loss_iter = []
for i in range(opt.iters):
    kfold = StratifiedKFold(n_splits=opt.num_folds, shuffle=True)
    acc_cv = []
    loss_cv = []

    for tr_index_sub, val_index_sub in kfold.split(x_features, labels):
        # for each validation, it should be a new model.
        ############### Define Graph Deep Learning Network ##########################
        model = Network(opt.indim, opt.ratio, opt.nclass, k=8, R=opt.nroi).to(device)

        if opt_method == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weightdecay)
        elif opt_method == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9, weight_decay=opt.weightdecay, nesterov = True)

        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.stepsize, gamma=opt.gamma)

        if opt.aug == True:
            tr_index = []
            val_index = []
            for k in tr_index_sub:
                tr_index += list(range(k*opt.rep, (k + 1)*opt.rep))
            for j in val_index_sub:
                val_index += list(range(j*opt.rep, (j + 1)*opt.rep))
        else:
            tr_index = tr_index_sub
            val_index = val_index_sub

        ################## Define Dataloader ##################################

        train_mask = torch.zeros(len(dataset), dtype=torch.uint8)
        val_mask = torch.zeros(len(dataset), dtype=torch.uint8)
        train_mask[tr_index] = 1
        val_mask[val_index] = 1
        train_dataset = dataset[train_mask]
        val_dataset = dataset[val_mask]

        # ######################## Data Preprocessing ########################
        # ###################### Normalize features ##########################

        if opt.normalization:
            for i in range(train_dataset.data.x.shape[1]):
                #
                # if i == 93:
                #     train_dataset.data.x[:, i] = train_dataset.data.x[:, i] * 1e15
                train_dataset.data.x[:, i], lamb, xmean, xstd = normal_transform_train(train_dataset.data.x[:, i])
                val_dataset.data.x[:, i] = normal_transform_test(val_dataset.data.x[:, i], lamb, xmean, xstd)
                #
                # train_dataset.data.x[:, i], lamb, xmean, xstd = boxcox_transform_train(train_dataset.data.x[:, i])
                # val_dataset.data.x[:, i] = boxcox_transform_test(val_dataset.data.x[:, i], lamb, xmean, xstd)

        val_loader = DataLoader(val_dataset, batch_size=opt.batchSize, shuffle=False)
        train_loader = DataLoader(train_dataset, batch_size=opt.batchSize, shuffle=False)
        print("Input feature shape" + str(train_dataset.data.x.shape))
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=1e-7,
        #                                                        last_epoch=-1)
        best_tr_loss = 2.0
        best_model_wts = copy.deepcopy(model.state_dict())
        for epoch in range(0, opt.n_epochs):
            since = time.time()
            tr_loss, s1_arr, s2_arr, le1, le2 = train(epoch)
            tr_acc = test_acc(train_loader)
            val_acc = test_acc(val_loader)
            val_loss = test_loss(val_loader, epoch)
            time_elapsed = time.time() - since
            print('*====**')
            print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            print('Epoch: {:03d}, Train Loss: {:.7f}, '
                  'Train Acc: {:.7f}, Test Loss: {:.7f}, Test Acc: {:.7f}'.format(epoch, tr_loss,
                                                               tr_acc, val_loss, val_acc))

            writer.add_scalars('Acc', {'train_acc': tr_acc, 'val_acc': val_acc}, epoch)
            writer.add_scalars('Loss', {'train_loss': tr_loss, 'val_loss': val_loss}, epoch)
            writer.add_histogram('Hist/hist_s1', s1_arr, epoch)
            writer.add_histogram('Hist/hist_s2', s2_arr, epoch)
            # if tr_loss < best_tr_loss:
            #     best_tr_loss = tr_loss
            #     best_model_wts = copy.deepcopy(model.state_dict())

            # if val_loss < best_loss and epoch > 5:
            #     print("saving best model")
            #     best_loss = val_loss
            #     best_model_wts = copy.deepcopy(model.state_dict())
            #     if not os.path.exists('models/'):
            #         os.makedirs('models/')
            #     if opt.save_model:
            #         torch.save(best_model_wts,
            #                    'models/rep{}_biopoint_{}_{}_{}.pth'.format(opt.rep, opt.fold, opt.net, opt.lamb5))

        ###############validation#######################
        model.eval()
        # model.load_state_dict(best_model_wts)
        test_accuracy = test_acc(val_loader)
        test_l = test_loss(val_loader, 0)
        print("===========================")
        print("Validation Acc: {:.7f}, Validation Loss: {:.7f} ".format(test_accuracy, test_l))
        print(opt)
        acc_cv.append(test_accuracy)
        loss_cv.append(test_l)
        del model
        del optimizer
        ############################################################
    acc_iter.append(np.mean(acc_cv))
    loss_iter.append(np.mean(loss_cv))

print("Average Validation Acc: {:.7f}, Average Validation Loss: {:.7f} ".format(np.mean(acc_iter), np.mean(loss_iter)))
print(np.std(acc_iter))


#######################################################################################
######################### Testing on testing set ######################################
#######################################################################################

# if opt.load_model:
#     model = Network(opt.indim,opt.ratio,opt.nclass).to(device)
#     model.load_state_dict(torch.load(os.path.join(opt.save_path,str(fold)+'.pth')))
#     model.eval()
#     preds = []
#     correct = 0
#     for data in val_loader:
#         data = data.to(device)
#         outputs= model(data.x, data.edge_index, data.batch, data.edge_attr,data.pos)
#         pred = outputs[0].max(1)[1]
#         preds.append(pred.cpu().detach().numpy())
#         correct += pred.eq(data.y).sum().item()
#     preds = np.concatenate(preds,axis=0)
#     trues = val_dataset.data.y.cpu().detach().numpy()
#     cm = confusion_matrix(trues,preds)
#     print("Confusion matrix")
#     print(classification_report(trues, preds))

# else:
#    model.load_state_dict(best_model_wts)
#    model.eval()
#    test_accuracy = test_acc(test_loader)
#    test_l= test_loss(test_loader,0)
#    print("===========================")
#    print("Test Acc: {:.7f}, Test Loss: {:.7f} ".format(test_accuracy, test_l))
#    print(opt)

