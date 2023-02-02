import numpy as np
from read_data_utils import load_features
import torch
import torch.nn.functional as F
from sklearn.svm import SVC
import sklearn
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
import pdb
from dl_svm import Model
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=1.0,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--penalty', type=float, default=0.5,
                    help='Initial learning rate.')
parser.add_argument('--epochs', type=int, default=30,
                    help='Number of epochs to train.')
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def hinge_loss(pred, y):
    return torch.mean(torch.square(torch.maximum(torch.zeros_like(pred), 1 - y*pred)))

def reg_loss(model):
    l2_reg = torch.tensor(0.0, requires_grad=True)
    for param in model.top_linear.parameters():
        l2_reg = l2_reg + torch.linalg.norm(param)
    return l2_reg

def train(inputs, labels_train, model, optimizer):
    model.train()
    labels_onehot = F.one_hot(labels_train, 2)
    for i in range(args.epochs):
        optimizer.zero_grad()
        output = model(inputs)
        loss = reg_loss(model) + args.penalty * hinge_loss(output, labels_onehot)
        print(i, loss.item())
        eval(output, labels_train)

        loss.backward()
        optimizer.step()

def eval(outputs, y):
    y = y.cpu().numpy()
    preds = np.argmax(outputs.cpu().detach().numpy(), axis=1)
    acc = np.mean(preds == y)
    probabilities = F.softmax(outputs, dim=1).cpu().detach().numpy()[:, 1]
    auc = sklearn.metrics.roc_auc_score(y, probabilities)
    print(auc, acc)
    return auc, acc

def test(X, y, model):
    model.eval()
    X_te = torch.from_numpy(X).float().to(device)
    y_te = torch.from_numpy(y).long().to(device)
    outputs = model(X_te)
    eval(outputs, y_te)


def cross_val():
    root_path = '/home/wenhuicu/ImagePTE/'

    features, labels = load_features(root_path, opt=0)
    # pdb.set_trace()
    kfold = StratifiedKFold(n_splits=36, shuffle=True, random_state=1211)
    for train_ind, test_ind in kfold.split(features, labels):
        idx = np.random.permutation(len(train_ind))
        model = Model(features.shape[-1], num_classes=2).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        fe_train = torch.from_numpy(features[train_ind]).float().to(device)[idx]
        labels_train = torch.from_numpy(labels[train_ind]).long().to(device)[idx]
        # pdb.set_trace()
        
        train(fe_train, labels_train, model, optimizer)
        test(features[test_ind], labels[test_ind], model)

        break

cross_val()