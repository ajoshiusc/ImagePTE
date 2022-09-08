import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import normalize
from sklearn.ensemble import RandomForestClassifier
import math
from tqdm import tqdm
#from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.neural_network import MLPClassifier

measure = 'fALFF'
atlas_name = 'USCLobes'
f = np.load('PTE_'+measure+'_'+atlas_name+'.npz')
ALFF_pte = f['roiwise_data']
ALFF_pte = ALFF_pte[1:13,:]

f = np.load('PTE_fmridiff_USCLobes.npz')
conn_pte = f['fdiff_sub']
lab_ids = f['label_ids']
gordlab = f['labels']
sub_ids = f['sub_ids']
n_rois = conn_pte.shape[0]
epi_brainsync = conn_pte.T

f = np.load('PTE_graphs_USCLobes.npz')
conn_pte = f['conn_mat']
lab_ids = f['label_ids']
gordlab = f['labels']
sub_ids = f['sub_ids']
cent_mat = f['cent_mat']
n_rois = conn_pte.shape[0]
ind = np.tril_indices(n_rois, k=1)
epi_connectivity = conn_pte[ind[0], ind[1], :].T

a = np.load('PTE_lesion_vols_USCLobes.npz', allow_pickle=True)
a = a['lesion_vols'].item()
epi_lesion_vols = np.array([a[k] for k in sub_ids])

#epi_lesion_vols=normalize(epi_lesion_vols)
#epi_connectivity=normalize(epi_connectivity)
#ALFF_pte = normalize(ALFF_pte.T).T


epi_lesion_vols = epi_lesion_vols/np.linalg.norm(epi_lesion_vols)
epi_connectivity = epi_connectivity/np.linalg.norm(epi_connectivity)
ALFF_pte = ALFF_pte/np.linalg.norm(ALFF_pte)
epi_brainsync = epi_brainsync/np.linalg.norm(epi_brainsync)


#epi_measures = np.concatenate((epi_lesion_vols,epi_connectivity,ALFF_pte.T), axis=1)
epi_measures = epi_connectivity



measure = 'fALFF'
atlas_name = 'USCLobes'
f = np.load('NONPTE_'+measure+'_'+atlas_name+'.npz')
ALFF_nonpte = f['roiwise_data']
ALFF_nonpte = ALFF_nonpte[1:13,:]

f = np.load('NONPTE_fmridiff_USCLobes.npz')
conn_pte = f['fdiff_sub']
lab_ids = f['label_ids']
gordlab = f['labels']
sub_ids = f['sub_ids']
n_rois = conn_pte.shape[0]
nonepi_brainsync = conn_pte.T

f = np.load('NONPTE_graphs_USCLobes.npz')
conn_nonpte = f['conn_mat']
lab_ids = f['label_ids']
gordlab = f['labels']
sub_ids = f['sub_ids']
cent_mat = f['cent_mat']

nonepi_connectivity = conn_nonpte[ind[0], ind[1], :].T

a = np.load('NONPTE_lesion_vols_USCLobes.npz', allow_pickle=True)
a = a['lesion_vols'].item()
nonepi_lesion_vols = np.array([a[k] for k in sub_ids])

#nonepi_lesion_vols=normalize(nonepi_lesion_vols)
#nonepi_connectivity=normalize(nonepi_connectivity)
#ALFF_nonpte = normalize(ALFF_nonpte.T).T

nonepi_lesion_vols=nonepi_lesion_vols/np.linalg.norm(nonepi_lesion_vols)
nonepi_connectivity=nonepi_connectivity/np.linalg.norm(nonepi_connectivity)
ALFF_nonpte = ALFF_nonpte/np.linalg.norm(ALFF_nonpte)
nonepi_brainsync = nonepi_brainsync/np.linalg.norm(nonepi_brainsync)

#nonepi_measures = np.concatenate((nonepi_lesion_vols,nonepi_connectivity,ALFF_nonpte.T), axis=1)
nonepi_measures = nonepi_connectivity


X = np.vstack((epi_measures, nonepi_measures))

y = np.hstack(
    (np.ones(epi_measures.shape[0]), np.zeros(nonepi_measures.shape[0])))

# Permute the labels to check if AUC becomes 0.5. This check is to make sure that we are not overfitting
#y = np.random.permutation(y)





    
my_metric = 'roc_auc'
kfold = StratifiedKFold(n_splits=36, shuffle=False)


iteration_num=10
auc_sum = np.zeros((iteration_num))
for i in range(iteration_num):
# y = np.random.permutation(y)
    drop=1

    #model = KerasClassifier(build_fn=create_model, epochs=100, batch_size=8,verbose=0)  
    model = MLPClassifier(solver='adam', alpha=1e-6, hidden_layer_sizes=(8), random_state=1, max_iter=1000)
   
    kfold = StratifiedKFold(n_splits=36, shuffle=True)
    auc = cross_val_score(model, X, y, cv=kfold, scoring=my_metric)
    auc_sum [i]= np.mean(auc)

    print('iter=%d, auc=%g'%(i,auc_sum[i]))
    


print('Average AUC with drop=%g , Std AUC=%g' % (np.mean(auc_sum), np.std(auc_sum)))
