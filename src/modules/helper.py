import numpy as np


def dense_transform(X):
    return X.todense()


def log_transform(x):
    return np.log(x + 1)

