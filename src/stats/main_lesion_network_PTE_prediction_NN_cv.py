"""

@author: Anand, Haleh
"""

'''This code used 11 lobe lesion volumes generated with VAE to predict epileptic subjects in a TBI population
36 subject in PTE class and 36 subjects in non PTE class
'''
import numpy as np
import nilearn.image as ni
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.optimizers import Adam
from keras import backend as K

import os
import importlib

def set_keras_backend(backend):

    if K.backend() != backend:
        os.environ['KERAS_BACKEND'] = backend
        importlib.reload(K)
        assert K.backend() == backend

set_keras_backend("theano")


def main():

    #######################load data################
    a = np.load('PTE_lesion_vols.npz', allow_pickle=True)
    a = a['lesion_vols'].item()
    epi_measures = np.array([a[k] for k in a])

    a = np.load('NONPTE_lesion_vols.npz', allow_pickle=True)
    a = a['lesion_vols'].item()
    nonepi_measures = np.array([a[k] for k in a])

    X = np.vstack((epi_measures, nonepi_measures))
    y = np.hstack(
        (np.ones(epi_measures.shape[0]), np.zeros(nonepi_measures.shape[0])))





#######################Random permutation################
## Random permutation of pairs of training subject for 1000 iterations
####################################################

    def create_model():
	# create model
        
        model = Sequential()
        model.add(Dense(32, activation="tanh", kernel_initializer="random_normal", input_shape=X.shape[1:]))
        if drop:
            model.add(Dropout(0.2))
        model.add(Dense(5, activation="relu", kernel_initializer="random_normal"))
        model.add(Dense(5, activation="relu", kernel_initializer="random_normal"))
        model.add(Dense(1, activation="sigmoid", kernel_initializer="random_normal"))
        # Compile model
        model.compile(optimizer = Adam(lr =.0001),loss='binary_crossentropy', metrics =['accuracy'])
        return model


     
    my_metric = 'roc_auc'
    kfold = StratifiedKFold(n_splits=36, shuffle=False)


    iteration_num=100
    auc_sum = np.zeros((iteration_num))
    for i in range(iteration_num):
    # y = np.random.permutation(y)
        drop=1
        model = KerasClassifier(build_fn=create_model, epochs=100, batch_size=8,verbose=0)  
        kfold = StratifiedKFold(n_splits=36, shuffle=True)
        auc = cross_val_score(model, X, y, cv=kfold, scoring=my_metric)
        auc_sum [i]= np.mean(auc)

        print('iter=%d, auc=%g'%(i,auc_sum[i]))
        #print('AUC after CV for i=%dgamma=%s is %g' %
            #(i, best_gamma, np.mean(auc)))


    print('Average AUC with drop=%g , Std AUC=%g' % (np.mean(auc_sum), np.std(auc_sum)))

      
    for i in range(iteration_num):
    # y = np.random.permutation(y)
        drop=0
        model = KerasClassifier(build_fn=create_model, epochs=100, batch_size=8,verbose=0)
        kfold = StratifiedKFold(n_splits=36, shuffle=True)
        auc = cross_val_score(model, X, y, cv=kfold, scoring=my_metric)
        auc_sum [i]= np.mean(auc)
        print('iter=%d, auc=%g'%(i,auc_sum[i]))


    print('Average AUC=%g , Std AUC=%g' % (np.mean(auc_sum), np.std(auc_sum)))


if __name__ == "__main__":
    main()
