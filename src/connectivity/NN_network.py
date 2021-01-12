
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
for i in range (num_sub):
    X_train= np.delete(X, [i,i+num_sub], 0)
    y_train=np.delete(y, [i,i+num_sub], 0)
    y_test=y[[i,i+num_sub]]
    X_test=X[[i,i+num_sub]]


    classifier = Sequential()
    #First Hidden Layer
    classifier.add(Dense(32, activation="tanh", kernel_initializer="random_normal", input_shape=X.shape[1:]))
    #Second Hidden Layer
    classifier.add(Dense(16, activation="relu", kernel_initializer="random_normal"))
    #Third Hidden Layer
    classifier.add(Dense(16, activation="relu", kernel_initializer="random_normal"))
    #Output Layer
    classifier.add(Dense(1, activation="sigmoid", kernel_initializer="random_normal"))
    #Compiling the model
    classifier.compile(optimizer = Adam(lr =.0001),loss='binary_crossentropy', metrics =['accuracy'])
    #Fitting the model
    classifier.fit(np.array(X_train),np.array(y_train), batch_size=8, epochs=100)



    y_pred=classifier.predict(X_test,batch_size=1)
    y_pred =(y_pred>0.5)
    cr.append(classifier.evaluate(np.array(X_test), np.array(y_test))[1])

print(sum(cr) / len(cr) )


