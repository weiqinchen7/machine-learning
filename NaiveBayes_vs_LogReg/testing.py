from math import ceil
import numpy as np
from numpy.random import randint


def train_test_index(label, train_percent):
    num_obs = label.size
    label_index = np.arange(num_obs)
    label_val = np.unique(label)
    train_index = np.array([], dtype=int)
    for k in range(0, label_val.size):
        y_sub_index = label_index[label == label_val[k]]
        num_train_k = ceil(y_sub_index.size * train_percent)
        train_index_k = np.random.choice(y_sub_index, num_train_k, replace=False)
        train_index = np.concatenate((train_index, train_index_k))
    test_index = np.setdiff1d(label_index, train_index, assume_unique=True)
    return (train_index, test_index)


def test_error_func(X, y, Classifier_, n_rep, train_percent, args=[]):
    print("Testing for training percent", train_percent)
    print("with", n_rep, "random 80-20 train-test splits", "for", Classifier_.__name__)
    n_obs = y.size
    test_error_mat = np.zeros((n_rep, train_percent.size))
    for rep in range(n_rep):
        print("Split--", rep, end="\r")
        # split as 80% train 20% test
        train_index, test_index = train_test_index(y, .8)
        X_train = X[train_index, :]
        y_train = y[train_index]
        X_test = X[test_index, :]
        y_test = y[test_index]
        for p in range(train_percent.size):
            percent = train_percent[p] / 100
            train_sub_index = train_test_index(y_train, percent)[0]
            X_train_sub = X_train[train_sub_index, :]
            y_train_sub = y_train[train_sub_index]
            model = Classifier_(X_train_sub, y_train_sub, *args)
            test_error_mat[rep, p] = model.validate(X_test, y_test)
    print("Testing is complete!")
    return test_error_mat