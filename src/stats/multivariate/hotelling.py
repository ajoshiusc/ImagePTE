import numpy as np
from scipy.stats import f as f_distrib


def hotelling_t2(X, Y):

    # X and Y are 3D arrays
    # dim 0: number of features
    # dim 1: number of subjects
    # dim 2: number of mesh nodes or voxels

    nx = X.shape[1]
    ny = Y.shape[1]
    p = X.shape[0]
    Xbar = X.mean(1)
    Ybar = Y.mean(1)
    Xbar = Xbar.reshape(Xbar.shape[0], 1, Xbar.shape[1])
    Ybar = Ybar.reshape(Ybar.shape[0], 1, Ybar.shape[1])

    X_Xbar = X - Xbar
    Y_Ybar = Y - Ybar
    Wx = np.einsum('ijk,ljk->ilk', X_Xbar, X_Xbar)
    Wy = np.einsum('ijk,ljk->ilk', Y_Ybar, Y_Ybar)
    W = (Wx + Wy) / float(nx + ny - 2)
    Xbar_minus_Ybar = Xbar - Ybar
    x = np.linalg.solve(W.transpose(2, 0, 1),
                        Xbar_minus_Ybar.transpose(2, 0, 1))
    x = x.transpose(1, 2, 0)

    t2 = np.sum(Xbar_minus_Ybar * x, 0)
    t2 = t2 * float(nx * ny) / float(nx + ny)
    stat = (t2 * float(nx + ny - 1 - p) / (float(nx + ny - 2) * p))

    pval = 1 - np.squeeze(f_distrib.cdf(stat, p, nx + ny - 1 - p))
    return pval, t2
