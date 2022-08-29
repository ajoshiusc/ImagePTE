import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import normalize
from sklearn.model_selection import GridSearchCV, ParameterGrid, RandomizedSearchCV
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier

a = np.load('../PTE_nonPTE_features_USCLobes.npz')

epi_measures = a['epi_measures']
nonepi_measures = a['nonepi_measures']

'''epi_measures = np.concatenate(
    (a['epi_lesion_vols'], a['epi_connectivity']), axis=1)  # , a['ALFF_pte']
nonepi_measures = np.concatenate(
    (a['nonepi_lesion_vols'], a['nonepi_connectivity']), axis=1)
'''


X = np.vstack((epi_measures, nonepi_measures))
y = np.hstack(
    (np.ones(epi_measures.shape[0]), np.zeros(nonepi_measures.shape[0])))

############## Features are created now ##############
# Permute the labels to check if AUC becomes 0.5. This check is to make sure that we are not overfitting
#y = np.random.permutation(y)
max_component = min((X.shape[0]-1), X.shape[1])
# for all features
param_grid = {"pca__n_components": [55, 56, 57, 58, 59], "svc__max_depth": [1, 1, 10, 100]}

#for lesion features and ALFF features
param_grid = {"pca__n_components": [3,5,7,9], "svc__max_depth": [3, 5, 7, 11]}

# best gamma=0.075 is
# best C=100 is
# n_components=54 is


pipe = Pipeline([('pca', PCA(whiten=True)),
                ('svc', RandomForestClassifier())])



#grid_search = GridSearchCV(pipe, param_grid=param_grid)

NUM_TRIALS = 10
auc = np.zeros(NUM_TRIALS)

for i in tqdm(range(NUM_TRIALS)):

    inner_cv = StratifiedKFold(n_splits=35)  
    outer_cv = StratifiedKFold(n_splits=36)

    clf = RandomizedSearchCV(
        estimator=pipe, param_distributions=param_grid, cv=inner_cv, scoring='roc_auc')
    nested_score = cross_val_score(
        clf, X=X, y=y, cv=outer_cv, scoring='roc_auc')
    auc[i] = nested_score.mean()
    print(auc[i])

print('Average AUC=%g , Std AUC=%g' % (np.mean(auc), np.std(auc)))
