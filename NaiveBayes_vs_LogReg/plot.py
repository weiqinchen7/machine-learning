import numpy as np
import matplotlib.pyplot as plt
from random import seed
from testing import test_error_func
from logregAgent import Logistic_Regression
from NBagent import Naive_Bayes


def logReg(dataset, num_splits=100,
           train_percent=np.array([5, 10, 15, 20, 25, 30])):
    feature = dataset[1:, 0:-1]
    label = dataset[1:, -1].astype(int)
    test_error = test_error_func(feature, label, Logistic_Regression,
                                 num_splits, train_percent, [1])
    test_error_mean = np.mean(test_error, axis=0)
    test_error_std = np.std(test_error, axis=0, ddof=1)
    return (test_error_mean, test_error_std)


def NBGaussian(dataset, num_splits=100,
               train_percent=np.array([5, 10, 15, 20, 25, 30])):
    feature = dataset[1:, 0:-1]
    label = dataset[1:, -1].astype(int)
    test_error = test_error_func(feature, label, Naive_Bayes,
                                 num_splits, train_percent)
    test_error_mean = np.mean(test_error, axis=0)
    test_error_std = np.std(test_error, axis=0, ddof=1)
    return (test_error_mean, test_error_std)

seed(10)
np.random.seed(10)
dataset = np.genfromtxt('./spam.csv', delimiter=",", skip_header=0)
logreg_error_mean, logreg_error_std = logReg(dataset)
nb_error_mean, nb_error_std = NBGaussian(dataset)
train_percent = np.array([5, 10, 15, 20, 25, 30])

plt.xlim([4, 31])
plt.errorbar(train_percent, logreg_error_mean, yerr=logreg_error_std * 1.96,
             label='Logistic Regression', fmt='-D', color='r', capthick=1.5)
plt.errorbar(train_percent, nb_error_mean, yerr=nb_error_std * 1.96,
             label='Naive Bayes', fmt='-s', color='g', capthick=1.5)
plt.legend()
plt.ylabel('Testing Error')
plt.xlabel('Percent of Training')
plt.title('Logistic Regression vs Naive Bayes', fontsize=15)
plt.savefig('./NB_logReg.pdf', dpi=700)