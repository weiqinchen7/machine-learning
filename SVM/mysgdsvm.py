import numpy as np
import time
from agent import SVM
import sys


def read_data(x):
    y = np.genfromtxt(x, delimiter=",", skip_header=0)
    return y

def mysgdsvm(file, k, repeat):
    repeat, k = int(repeat), int(k)
    assert(repeat > 0 and k > 0)
    data = read_data(file)
    X = data[:, 1:]  # the rest of entries are the feature
    y = data[:, 0]  # the first entry is the label
    regu_lam = 1
    time_ = np.zeros(repeat)
    log_list = []
    for i in range(repeat):
        b = time.time()  # begin time
        svm_model = SVM(X, y, regu_lam, k)
        log_list.append(svm_model.loglist)
        e = time.time()  # end time
        time_[i] = e - b

    avg = np.mean(time_)
    std = np.std(time_, ddof=1)
    with open("./tmp.txt", "w") as f:
        for li in log_list:
            for val in li:
                print(val, end=", ", file=f)
            print(file=f)

    print()
    print("Average of ", repeat, " runs and ", k, " batch size",
          ":\t", avg, " sec.")
    print("Std of ", repeat, " runs and", k, "batch size",
          ":\t", std, " sec.")
    return

def main(argv=sys.argv):
    if len(argv) == 4:
        mysgdsvm(*argv[1:])
    else:
        print(
            'Usage: python3 ./mysgdsvm.py /path/to/dataset.csv k repeat', file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()