from agent import SVM
import numpy as np
import matplotlib.pyplot as plt


gen_data = np.genfromtxt('MNIST-13.csv', delimiter=',')
label = gen_data[:, 0]  # the first entry is the label
feature = gen_data[:, 1:]  # the rest of entries are the feature
np.random.seed(10)

plt.figure()
plt.grid()
for i in range(5):
    result = SVM(feature, label, 1, 1)
    plt.plot(result.loglist)
plt.xlabel('# of iteration')
plt.ylabel('Primal Objective')
plt.title('k = 1')
plt.savefig('k_1.png', dpi=600)

plt.figure()
plt.grid()
for i in range(5):
    result = SVM(feature, label, 1, 20)
    plt.plot(result.loglist)
plt.xlabel('# of iteration')
plt.ylabel('Primal Objective')
plt.title('k = 20')
plt.savefig('k_20.png', dpi=600)

plt.figure()
plt.grid()
for i in range(5):
    result = SVM(feature, label, 1, 100)
    plt.plot(result.loglist)
plt.xlabel('# of iteration')
plt.ylabel('Primal Objective')
plt.title('k = 100')
plt.savefig('k_100.png', dpi=600)

plt.figure()
plt.grid()
for i in range(5):
    result = SVM(feature, label, 1, 200)
    plt.plot(result.loglist)
plt.xlabel('# of iteration')
plt.ylabel('Primal Objective')
plt.title('k = 200')
plt.savefig('k_200.png', dpi=600)

plt.figure()
plt.grid()
for i in range(5):
    result = SVM(feature, label, 1, 2000)
    plt.plot(result.loglist)
plt.xlabel('# of iteration')
plt.ylabel('Primal Objective')
plt.title('k = 2000')
plt.savefig('k_2000.png', dpi=600)