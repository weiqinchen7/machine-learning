import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.sparse.linalg.eigen.arpack import eigsh as largest_eigsh  # largest eigenvalue
import seaborn as sns


# hyper-parameters
num_iteration = 800
l2_reg = 0.0001
worker = 6


# load data
trainingDataWorker = []
trainingLabelWorker = []
number = 0
for order in range(10):
	trainingDataWorker.append(np.load('./mnist/train_%d_data.npy'%order).reshape((-1,28*28))/255.)
	trainingLabelWorker.append(([-1 if i==0 else 1 for i in np.load('./mnist/train_%d_label.npy'%order)]))
	number += len(trainingLabelWorker[order])

train_data_all = trainingDataWorker[0]
train_label_all = trainingLabelWorker[0]

for num_i in range(1, 10):
    train_data_all = np.concatenate((train_data_all, trainingDataWorker[num_i]))
    train_label_all = np.concatenate((train_label_all, trainingLabelWorker[num_i]))

N = train_data_all.shape[0]
num_sample = N // worker
trainingDataWorker = []
trainingLabelWorker = []
for i in range(5):
    trainingDataWorker.append(train_data_all[num_sample * i:num_sample * (i + 1)])
    trainingLabelWorker.append(train_label_all[num_sample * i:num_sample * (i + 1)])
trainingDataWorker.append(train_data_all[num_sample * 5:])
trainingLabelWorker.append(train_label_all[num_sample * 5:])


# Lipschitz constant
Lips = 0
for i in range(6):
    large_sparse, _ = largest_eigsh(np.matmul(trainingDataWorker[i].T, trainingDataWorker[i]), 1, which='LM')
    Lips = np.max(large_sparse / 4 / trainingDataWorker[i].shape[0])
alpha = 1 / Lips
dimen = 28 * 28


def Decentralized_GD(worker, method, model, trainingDataWorker, trainingLabelWorker,
        l2_reg, num_iteration, stepsize):
    assert (len(trainingDataWorker) == len(trainingLabelWorker))
    assert (isinstance(num_iteration, int) & (num_iteration > 0))
    assert (isinstance(stepsize, float) & (stepsize > 0))

    # Initialization
    theta0 = np.zeros(dimen)
    theta = np.zeros((worker, dimen))
    for i in range(6):
        theta[i] = theta0

    # mixing matrix W
    if model == 'ring_network':
        W = np.zeros((worker, worker))
        for i in range(worker):
            W[i, (i - 1) % worker] = 1 / 3
            W[i, i] = 1 / 3
            W[i, (i + 1) % worker] = 1 / 3
        stepsize = stepsize * (1 + np.min(np.linalg.eig(W)[0])) / 3
    elif model == 'complete_network':
        W = np.ones((worker, worker)) / worker
        stepsize = stepsize * (1 + np.min(np.linalg.eig(W)[0])) / 3

    grad_hi = np.zeros(num_iteration)
    all_loss_hi = np.zeros((worker, num_iteration))
    mean_loss_hi = np.zeros((num_iteration))
    local_loss_hi = np.zeros((worker, num_iteration))
    GradientTemp = np.zeros((worker, dimen))
    objFuncTemp = np.zeros((worker, worker))
    objFuncMean = np.zeros((worker))
    objFuncTempLocal = np.zeros((worker))
    GradientTemp0 = np.zeros((worker, dimen))

    thetaTemp = np.zeros((worker, dimen))
    for i in range(num_iteration):
        thetaTemp = np.matmul(W, theta)

        for work in range(worker):
            Xdata = trainingDataWorker[work]
            Ydata = np.array(trainingLabelWorker[work])
            if method == 'DGD':
                temp = np.exp(-Ydata * np.matmul(Xdata, theta[work]))
                GradientTemp[work] = np.mean((-Ydata * Xdata.T * temp) / (1 + temp), axis=1) + \
                                     l2_reg * theta[work]
            elif method == 'EXTRA':
                idx = random.randint(0, Xdata.shape[0] - 1)
                Xdata = Xdata[idx]
                Ydata = Ydata[idx]
                temp = np.exp(-Ydata * np.matmul(Xdata, theta[work]))
                GradientTemp[work] = (-Ydata * Xdata.T * temp) / (1 + temp) + l2_reg * theta[work]

            Xdata = trainingDataWorker[work]
            Ydata = np.array(trainingLabelWorker[work])

            temp = np.exp(-Ydata * np.matmul(Xdata, theta[work]))
            objFuncTempLocal[work] = np.mean(np.log(1 + temp)) + l2_reg / 2 * np.linalg.norm(theta[work]) ** 2

            for estwork in range(worker):
                Xdata = trainingDataWorker[estwork]
                Ydata = np.array(trainingLabelWorker[estwork])

                temp = np.exp(-Ydata * np.matmul(Xdata, theta[work]))
                objFuncTemp[work][estwork] = np.mean(np.log(1 + temp)) + l2_reg / 2 * np.linalg.norm(theta[work]) ** 2
        # Update
        theta = thetaTemp - stepsize * GradientTemp
        thetamean = np.mean(theta, axis=0)

        for work in range(worker):
            Xdata = trainingDataWorker[work]
            Ydata = np.array(trainingLabelWorker[work])

            temp = np.exp(-Ydata * np.matmul(Xdata, thetamean))
            objFuncMean[work] = np.mean(np.log(1 + temp)) + l2_reg / 2 * np.linalg.norm(thetamean) ** 2

        mean_loss_hi[i] = np.sum(objFuncMean)
        all_loss_hi[:, i] = np.sum(objFuncTemp, axis=1)
        local_loss_hi[:, i] = objFuncTempLocal
        Gradient = np.sum(GradientTemp, axis=0)
        grad_hi[i] = np.linalg.norm(Gradient)

    return grad_hi, all_loss_hi, mean_loss_hi, local_loss_hi


#%%%%%
'''plot of ring_network'''
grad_hi = np.zeros((2, num_iteration))
all_loss_hi = np.zeros((2, worker, num_iteration))
mean_loss_hi = np.zeros((2, num_iteration))
local_loss_hi = np.zeros((2, worker, num_iteration))

# stepsize
stepsize = alpha
grad_hi[0], all_loss_hi[0], mean_loss_hi[0], local_loss_hi[0] = \
    Decentralized_GD(worker, 'DGD', 'ring_network', trainingDataWorker, trainingLabelWorker, l2_reg, num_iteration, stepsize)
grad_hi[1], all_loss_hi[1], mean_loss_hi[1], local_loss_hi[1] = \
    Decentralized_GD(worker, 'EXTRA', 'ring_network', trainingDataWorker, trainingLabelWorker, l2_reg, num_iteration, stepsize)

legend_worker = ['Worker_1', 'Worker_2', 'Worker_3', 'Worker_4', 'Worker_5', 'Worker_6']
legend_method = ['DGD_ring_network', 'EXTRA_ring_network']

plt.style.use('seaborn')
plt.rc('axes', labelsize=19)
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)
plt.rc('legend', fontsize=17)
ax = plt.gca()
ax.xaxis.get_major_formatter().set_powerlimits((0, 1))
for i in range(2):
    plt.plot(mean_loss_hi[i])
plt.xlabel('# of iteration')
plt.ylabel('Objective on average theta')
plt.legend(legend_method, fontsize='large')
plt.savefig('Obj_iter_ringNetwork.png', dpi=200)
plt.close()

plt.style.use('seaborn')
plt.rc('axes', labelsize=19)
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)
plt.rc('legend', fontsize=17)
ax = plt.gca()
ax.xaxis.get_major_formatter().set_powerlimits((0, 1))
for i in range(2):
    plt.plot(mean_loss_hi[i])
plt.xlabel('# of iteration')
plt.ylabel('Averaged consensus error')
plt.legend(legend_method, fontsize='large')
plt.savefig('Average consensus error_ringNetwork.png', dpi=200)
plt.close()

for i in range(6):
    plt.plot(all_loss_hi[0, i, :])
plt.style.use('seaborn')
plt.rc('axes', labelsize=19)
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)
plt.rc('legend', fontsize=17)
plt.legend(legend_worker, fontsize='large')
plt.xlabel('# of iteration')
plt.ylabel('Objective of workers')
ax = plt.gca()
ax.xaxis.get_major_formatter().set_powerlimits((0, 1))
plt.savefig('Obj_iter_ringNetwork_DGD.png', dpi=200)
plt.close()

for i in range(6):
    plt.plot(all_loss_hi[1, i, :])
plt.style.use('seaborn')
plt.rc('axes', labelsize=19)
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)
plt.rc('legend', fontsize=17)
plt.legend(legend_worker, fontsize='large')
plt.xlabel('# of iteration')
plt.ylabel('Objective of workers')
ax = plt.gca()
ax.xaxis.get_major_formatter().set_powerlimits((0, 1))
plt.savefig('Obj_iter_ringNetwork_EXTRA.png', dpi=200)
plt.close()

plt.plot(range(0, num_iteration * N, N), mean_loss_hi[0])
plt.plot(range(0, num_iteration * worker, worker), mean_loss_hi[1])
plt.legend(legend_method, fontsize='large')
plt.xlabel('Index of sample')
plt.ylabel('Objective of workers')
ax = plt.gca()
ax.xaxis.get_major_formatter().set_powerlimits((0, 1))
plt.savefig('Obj_sampIdx_ringNetwork.png', dpi=200)
plt.close()


#%%%%%
'''plot of complete_network'''
grad_hi = np.zeros((2, num_iteration))
all_loss_hi = np.zeros((2, worker, num_iteration))
mean_loss_hi = np.zeros((2, num_iteration))
local_loss_hi = np.zeros((2, worker, num_iteration))

stepsize = alpha
grad_hi[0], all_loss_hi[0], mean_loss_hi[0], local_loss_hi[0] = \
    Decentralized_GD(worker, 'DGD', 'complete_network', trainingDataWorker, trainingLabelWorker, l2_reg, num_iteration, stepsize)
grad_hi[1], all_loss_hi[1], mean_loss_hi[1], local_loss_hi[1] = \
    Decentralized_GD(worker, 'EXTRA', 'complete_network', trainingDataWorker, trainingLabelWorker, l2_reg, num_iteration, stepsize)

legend_worker = ['Worker_1', 'Worker_2', 'Worker_3', 'Worker_4', 'Worker_5', 'Worker_6']
legend_method = ['DGD_complete_network', 'EXTRA_complete_network']

for i in range(2):
    plt.plot(mean_loss_hi[i])
plt.style.use('seaborn')
plt.rc('axes', labelsize=19)
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)
plt.rc('legend', fontsize=17)
plt.legend(legend_method, fontsize='large')
plt.xlabel('# of iteration')
plt.ylabel('Objective on average theta')
ax = plt.gca()
ax.xaxis.get_major_formatter().set_powerlimits((0, 1))
plt.savefig('Obj_iter_completeNetwork.png', dpi=200)

for i in range(2):
    plt.plot(mean_loss_hi[i])
plt.style.use('seaborn')
plt.rc('axes', labelsize=19)
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)
plt.rc('legend', fontsize=17)
plt.legend(legend_method, fontsize='large')
plt.xlabel('# of iteration')
plt.ylabel('Average consensus error')
ax = plt.gca()
ax.xaxis.get_major_formatter().set_powerlimits((0, 1))
plt.savefig('Average consensus error_complete.png', dpi=200)


for i in range(6):
    plt.plot(all_loss_hi[0, i, :])
plt.style.use('seaborn')
plt.rc('axes', labelsize=19)
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)
plt.rc('legend', fontsize=17)
plt.legend(legend_worker, fontsize='large')
plt.xlabel('# of iteration')
plt.ylabel('Objective of workers')
ax = plt.gca()
ax.xaxis.get_major_formatter().set_powerlimits((0, 1))
plt.savefig('Obj_iter_completeNetwork_DGD.png', dpi=200)
plt.close()

for i in range(6):
    plt.plot(all_loss_hi[1, i, :])
plt.style.use('seaborn')
plt.rc('axes', labelsize=19)
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)
plt.rc('legend', fontsize=17)
plt.legend(legend_worker, fontsize='large')
plt.xlabel('# of iteration')
plt.ylabel('Objective of workers')
ax = plt.gca()
ax.xaxis.get_major_formatter().set_powerlimits((0, 1))
plt.savefig('Obj_iter_completeNetwork_EXTRA.png', dpi=200)
plt.close()

plt.plot(range(0, num_iteration * N, N), mean_loss_hi[0])
plt.plot(range(0, num_iteration * worker, worker), mean_loss_hi[1])
plt.legend(legend_method, fontsize='large')
plt.xlabel('Index of sample')
plt.ylabel('Objective of workers')
ax = plt.gca()
ax.xaxis.get_major_formatter().set_powerlimits((0, 1))
plt.savefig('Obj_sampIdx_completeNetwork.png', dpi=200)
plt.close()