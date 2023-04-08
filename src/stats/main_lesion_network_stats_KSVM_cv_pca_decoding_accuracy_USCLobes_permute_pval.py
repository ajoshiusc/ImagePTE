import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from dfsio import readdfs, writedfs
import nilearn.image as ni
from tqdm import tqdm

def decode_accuracy_atlas(coef, roi_ids, atlasbasename, outbase):

    coef = 2.0*(coef-0.5)
    v = ni.load_img(atlasbasename + '.label.nii.gz')
    vlab = v.get_fdata()
    left = readdfs(atlasbasename + '.left.mid.cortex.dfs')
    right = readdfs(atlasbasename + '.right.mid.cortex.dfs')
    left.labels = np.mod(left.labels, 1000)
    right.labels = np.mod(right.labels, 1000)
    vlab = np.mod(vlab, 1000)

    left.attributes = np.zeros(left.vertices.shape[0])
    right.attributes = np.zeros(right.vertices.shape[0])

    vimp = np.zeros(vlab.shape)
    for i, r in enumerate(roi_ids):
        vimp[vlab == r] = coef[i]
        left.attributes[left.labels == r] = coef[i]
        right.attributes[right.labels == r] = coef[i]

    
    vi = ni.new_img_like(v, np.float32(vimp))
    vi.to_filename(outbase+'.decode.nii.gz')

    writedfs(outbase+'.left.decode.dfs', left)
    writedfs(outbase+'.right.decode.dfs', right)



f = np.load('./PTE_graphs_USCLobes_selected.npz')
conn_pte = f['conn_mat']
lab_ids = f['label_ids']
gordlab = f['labels']
sub_ids = f['sub_ids']
cent_mat = f['cent_mat']
n_rois = conn_pte.shape[0]
epi_connectivity = np.moveaxis(conn_pte,2,0)

a = np.load('./stats/PTE_lesion_vols.npz', allow_pickle=True)
a = a['lesion_vols'].item()
epi_lesion_vols = np.array([a[k] for k in sub_ids])
epi_measures = np.concatenate(
    (epi_connectivity, epi_lesion_vols[:,:,None]), axis=2)


f = np.load('./NONPTE_graphs_USCLobes_selected.npz')
conn_nonpte = f['conn_mat']
lab_ids = f['label_ids']
gordlab = f['labels']
sub_ids = f['sub_ids']
cent_mat = f['cent_mat']

nonepi_connectivity = np.moveaxis(conn_nonpte,2,0)

a = np.load('./stats/NONPTE_lesion_vols.npz', allow_pickle=True)
a = a['lesion_vols'].item()
nonepi_lesion_vols = np.array([a[k] for k in sub_ids])
nonepi_measures = np.concatenate(
    (nonepi_connectivity, nonepi_lesion_vols[:,:,None]), axis=2)

X = np.vstack((epi_measures, nonepi_measures))
y = np.hstack(
    (np.ones(epi_measures.shape[0]), np.zeros(nonepi_measures.shape[0])))

# Permute the labels to check if AUC becomes 0.5. This check is to make sure that we are not overfitting

n_iter = 1000
auc = np.zeros(n_iter)
precision = np.zeros(n_iter)
recall = np.zeros(n_iter)
fscore = np.zeros(n_iter)
support = np.zeros(n_iter)


my_metric = 'roc_auc'

#y = np.random.permutation(y)

#######################selecting gamma################
## Following part of the code do a grid search to find best value of gamma using a one fold cross validation
## the metric for comparing the performance is AUC

roi_auc = np.zeros(X.shape[1])


for f in range(X.shape[1]):

    #    for mygamma in [1, 0.001, 0.05, 0.075, .1, .15, 0.2, 0.3, .5, 1, 5, 10, 100]:
    clf = SVC(kernel='rbf', tol=1e-9)
    my_metric = 'roc_auc'
    kfold = StratifiedKFold(n_splits=36, shuffle=True)
    auc = cross_val_score(clf, X[:,f].squeeze(), y, cv=kfold, scoring=my_metric)

    roi_auc[f] = np.mean(auc)

print('done')

atlas='/ImagePTE1/ajoshi/code_farm/svreg/USCLobes/BCI-DNI_brain'

outbase = 'usclobes_decode_lesion_conn'

roi_list = [301, 300, 401, 400, 101, 100, 201, 200, 501, 500, 900]

decode_accuracy_atlas(roi_auc, roi_ids=roi_list, atlasbasename=atlas, outbase=outbase)


# Generate null distribution and p value

nperm = 100
roi_auc_null = np.zeros((1000,X.shape[1]))

for i in tqdm(range(nperm)):
    for f in range(X.shape[1]):

        #    for mygamma in [1, 0.001, 0.05, 0.075, .1, .15, 0.2, 0.3, .5, 1, 5, 10, 100]:
        clf = SVC(kernel='rbf', tol=1e-9)
        my_metric = 'roc_auc'
        kfold = StratifiedKFold(n_splits=36, shuffle=True)
        auc = cross_val_score(clf, X[:,f].squeeze(), y, cv=kfold, scoring=my_metric)

        roi_auc_null[i,f] = np.mean(auc)

print('done')

atlas='/ImagePTE1/ajoshi/code_farm/svreg/USCLobes/BCI-DNI_brain'

outbase = 'usclobes_decode_lesion_conn_pval'

roi_list = [301, 300, 401, 400, 101, 100, 201, 200, 501, 500, 900]

roi_auc_pval = np.sum(roi_auc_null > roi_auc,axis=0)/nperm
decode_accuracy_atlas(roi_auc_pval, roi_ids=roi_list, atlasbasename=atlas, outbase=outbase)





