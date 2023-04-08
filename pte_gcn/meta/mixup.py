import numpy as np
import torch

def mixup_same_cls(x, alpha=1.0):
    """
    Apply mixup augmentation to inputs x and labels y.
    :param x: Input data, shape (batch_size, num_features)
    :param y: Input labels, shape (batch_size,)
    :param alpha: Mixup alpha value
    :return: Mixed inputs and labels, shape (batch_size, num_features) and (batch_size,)
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.shape[0]
    index = np.random.permutation(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]

    return mixed_x
