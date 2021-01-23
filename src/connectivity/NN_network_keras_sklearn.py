
import sys
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.stats as ss
from matplotlib.image import imsave
from scipy.stats import norm
from sklearn.metrics import auc, plot_roc_curve, roc_auc_score, roc_curve, classification_report, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from statsmodels.stats.multitest import fdrcorrection
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from grayord_utils import visdata_grayord
from sklearn.metrics import confusion_matrix, classification_report
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

population = 'PTE'
f = np.load(population+'_graphs.npz')
conn_pte = f['conn_mat']
lab_ids = f['label_ids']
gordlab = f['labels']

population = 'NONPTE'
f = np.load(population+'_graphs.npz')
conn_nonpte = f['conn_mat']
lab_ids = f['label_ids']
gordlab = f['labels']

n_rois = conn_pte.shape[0]
ind = np.tril_indices(n_rois, k=1)


# Do SVM Analysis
epi_measures = conn_pte[ind[0], ind[1], :].T
nonepi_measures = conn_nonpte[ind[0], ind[1], :].T

X = np.vstack((epi_measures, nonepi_measures))
y = np.hstack(
    (np.ones(epi_measures.shape[0]), np.zeros(nonepi_measures.shape[0])))

n_iter = 100
auc = np.zeros(n_iter)
precision = np.zeros(n_iter)
recall = np.zeros(n_iter)
fscore = np.zeros(n_iter)
support = np.zeros(n_iter)

auc_t = np.zeros(n_iter)
n_features = 21
y_test_true_all = []
y_test_pred_all = []

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
num_sub=int(X.shape[0]/2)
cr=[]

def create_model():
	# create model
    model = Sequential()
    model.add(Dense(32, activation="tanh", kernel_initializer="random_normal", input_shape=X.shape[1:]))
    model.add(Dense(16, activation="relu", kernel_initializer="random_normal"))
    model.add(Dense(16, activation="relu", kernel_initializer="random_normal"))
    model.add(Dense(1, activation="sigmoid", kernel_initializer="random_normal"))
    # Compile model
    model.compile(optimizer = Adam(lr =.0001),loss='binary_crossentropy', metrics =['accuracy'])
    return model


model = KerasClassifier(build_fn=create_model, epochs=100, batch_size=20,verbose=0)   
my_metric = 'roc_auc'
kfold = StratifiedKFold(n_splits=36, shuffle=False)
auc = cross_val_score(model, X, y, cv=kfold,scoring=my_metric)
print('AUC after CV is %g(%g)'%( np.mean(auc), np.std(auc)))
