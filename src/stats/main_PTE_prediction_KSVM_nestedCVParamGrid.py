import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import normalize
from sklearn.model_selection import GridSearchCV,ParameterGrid
from tqdm import tqdm
a = np.load('PTE_nonPTE_features_USCLobes.npz')

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
# param_grid = {"pca__n_components": range(1, max_component), "svc__C": [0.0001, 0.001, 0.01, .1, .3, .6, 0.7, 0.9, 1, 1.5, 2, 3, 4, 5, 6, 7, 9, 10, 100], "svc__gamma": [
#    1, 0.001, 0.05, 0.075, .1, .13, .15, .17, 0.2, 0.3, .5, 1, 5, 10, 100]}

grid = [{"pca__n_components": [54], "svc__C": [0.01, .1,  10, 100]}, {"svc__gamma": [0.001, 0.075, 0.1]}]
#param_grid = {"pca__n_components": [54],
#              "svc__C": [100], "svc__gamma": [0.075]}

param_grid = ParameterGrid(grid)
#param_grid = {"pca__n_components": [65], "svc__C": [0.01, .1,  10,100], "svc__gamma": [1, .1, 10]}

# best gamma=0.075 is
# best C=100 is
# n_components=54 is


pipe = Pipeline([('pca', PCA(whiten=True)),
                ('svc', SVC(kernel='rbf', tol=1e-9))])


#grid_search = GridSearchCV(pipe, param_grid=param_grid)

NUM_TRIALS = 100
auc = np.zeros(NUM_TRIALS)

for i in tqdm(range(NUM_TRIALS)):

    inner_cv = StratifiedKFold(n_splits=35, shuffle=True)#, random_state=1211)
    outer_cv = StratifiedKFold(n_splits=36, shuffle=True, random_state=1211)

    clf = GridSearchCV(estimator=pipe, param_grid=param_grid, cv=inner_cv)
    nested_score = cross_val_score(
        clf, X=X, y=y, cv=outer_cv, error_score='raise')
    auc[i] = nested_score.mean()
    print(auc[i])

print('Average AUC=%g , Std AUC=%g' % (np.mean(auc), np.std(auc)))
