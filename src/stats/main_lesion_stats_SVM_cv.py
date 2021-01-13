import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

def main():

    a = np.load('PTE_lesion_vols.npz', allow_pickle=True)
    a = a['lesion_vols'].item()
    epi_measures = np.array([a[k] for k in a])

    a = np.load('NONPTE_lesion_vols.npz', allow_pickle=True)
    a = a['lesion_vols'].item()
    nonepi_measures = np.array([a[k] for k in a])

    X = np.vstack((epi_measures, nonepi_measures))
    y = np.hstack(
        (np.ones(epi_measures.shape[0]), np.zeros(nonepi_measures.shape[0])))

    for cval in [0.0001, 0.001, 0.01, .1, .3, .6, .9, 1, 1.5, 2, 3, 4, 5, 6, 7, 9, 10, 100]:
        #    for mygamma in [1, 0.001, 0.05, 0.075, .1, .15, 0.2, 0.3, .5, 1, 5, 10, 100]:
        clf = SVC(kernel='linear', C=cval, tol=1e-4)
        my_metric = 'roc_auc'
        kfold = StratifiedKFold(n_splits=37, shuffle=False)
        auc = cross_val_score(clf, X, y, cv=kfold, scoring=my_metric)

        print('AUC on testing data:', cval, np.mean(auc))

    print('done')


if __name__ == "__main__":
    main()
