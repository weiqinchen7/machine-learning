import numpy as np
import sys
import os.path
from testing import test_error_func
from logregAgent import Logistic_Regression


def read_file(filename):
    assert os.path.isfile(filename)
    data = np.genfromtxt(filename, delimiter=",", skip_header=0)
    print("Successfully Loaded", filename)
    return data


def logisticRegression(filename, num_splits,
                       train_percent=np.array([5, 10, 15, 20, 25, 30])):
    num_splits = int(num_splits)
    data = read_file(filename)
    feature = data[1:, 0:-1]
    label = data[1:, -1].astype(int)
    test_error = test_error_func(feature, label, Logistic_Regression,
                                 num_splits, train_percent, [1])
    test_error_mean = np.mean(test_error, axis=0)
    test_error_std = np.std(test_error, axis=0, ddof=1)
    print("Testing Error:")
    print(test_error)
    print("Mean of Testing Error:")
    print(test_error_mean)
    print("Standard Deviation of Testing Error:")
    print(test_error_std)


def main(argv=sys.argv):
    if len(argv) == 4:
        train_percent_str = argv[3]
        train_percent = np.array(train_percent_str.split(), dtype=int)
        logisticRegression(argv[1], argv[2], train_percent)
    else:
        print('Usage: ./logisticRegression '
              '/path/to/dataset.csv num_splits train_percent', file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()