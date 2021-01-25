import numpy as np
import nilearn.image as ni
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

    for mygamma in [1, 0.001, 0.05, 0.075, .1, .13, .15, .17, 0.2, 0.3, .5, 1, 5, 10, 100]:
        clf = SVC(kernel='rbf', gamma=mygamma, tol=1e-9)
        my_metric = 'roc_auc'
        #auc = cross_val_score(clf, X, y, cv=37, scoring=my_metric)
        kfold = StratifiedKFold(n_splits=37, shuffle=False)
        auc = cross_val_score(clf, X, y, cv=kfold, scoring=my_metric)
        print('AUC on testing data:gamma=%g, auc=%g' % (mygamma, np.mean(auc)))

    for mygamma in ['auto', 'scale']:
        clf = SVC(kernel='rbf', gamma=mygamma, tol=1e-9)
        my_metric = 'roc_auc'
        kfold = StratifiedKFold(n_splits=37, shuffle=False)
        auc = cross_val_score(clf, X, y, cv=kfold, scoring=my_metric)
        print('AUC on testing data:gamma=%s, auc=%g' % (mygamma, np.mean(auc)))

    print('done')


if __name__ == "__main__":
    main()
