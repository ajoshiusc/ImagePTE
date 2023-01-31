import numpy as np
from read_data_utils import load_features
import torch
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
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
parser.add_argument('--epochs', type=int, default=17,
                    help='Number of epochs to train.')
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

root_path = '/home/wenhuicu/ImagePTE/'

features, labels = load_features(root_path)
model = Model(features.shape[-1], num_classes=2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
# pdb.set_trace()

def hinge_loss(pred, y):
    return torch.mean(torch.square(torch.maximum(torch.zeros_like(pred), 1 - y*pred)))

def reg_loss(model):
    l2_reg = torch.tensor(0.0, requires_grad=True)
    for param in model.top_linear.parameters():
        l2_reg = l2_reg + torch.linalg.norm(param)

def train(model, inputs, labels_train):
    inputs = torch.from_numpy(inputs).float().to(device)
    labels_train = torch.from_numpy(labels_train).float().to(device)
    model.train()

    for i in range(args.epochs):
        optimizer.zero_grad()
        output = model(inputs)
        loss = reg_loss(model) + args.penalty * hinge_loss(output, labels_train)
        loss.backward()
        optimizer.step()
        print(loss.item())


train(model, features, labels)

