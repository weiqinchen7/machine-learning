import random
import time
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt


''' %%%%%%%%%  hyper-parameters  %%%%%%%%%% '''
# maximal number of iterations
max_it = 15000
# learning rate
lr = 0.01
# Langrangian multiplier for regularization
lam = 0.0


''' %%%%%%%%%  read data  %%%%%%%%%% '''
with open('./ionosphere.data', 'r') as dataset:
    row = dataset.read().splitlines()
length = len(row)
# data, list[list] -> list[array] -> array[array]
data_x = [row[i].split(",")[:-1] for i in range(length)]
data_x = [np.asarray(data_x[i], dtype=np.float32) for i in range(len(data_x))]  # (351, 34)
data_x = np.asarray(data_x)  # (351, 34)
# label
data_y = [row[i].split(",")[-1] for i in range(length)]
data_y = [1 if data_y[i] == 'g' else -1 for i in range(length)]  # (351,)
# list -> array, increase the dimension by 1
data_y = np.asarray(data_y)[:, np.newaxis]  # (351, 1)

# dimension of data_x
dim = data_x.shape[-1]
# initialization of parameters
theta_0 = np.random.rand(dim) / 1000
# record historical loss
hist_loss = np.zeros(max_it)


''' %%%%%%%%%  loss, gradient, minimal loss  %%%%%%%%%% '''
theta = theta_0
temp = np.exp(- data_y.T * np.matmul(data_x, theta)).T  # 351
loss = np.mean(np.log(1+temp)) + lam/2 * LA.norm(theta)**2
gradient = np.mean((- data_y * data_x * temp)/(1 + temp), axis=0) + lam * theta
hist_loss[0] = loss
for i in range(max_it):
    # update theta by GD
    theta = theta - lr * gradient
    temp = np.exp(- data_y.T * np.matmul(data_x, theta)).T
    # loss
    minimal_loss1 = np.mean(np.log(1 + temp)) + lam/2 * LA.norm(theta)**2
    # gradient
    gradient = np.mean((- data_y * data_x * temp)/(1 + temp), axis=0) + lam * theta

# for the case of lambda = 0.01
lam = 0.01
for i in range(max_it):
    # update theta by GD
    theta = theta - lr * gradient
    temp = np.exp(- data_y.T * np.matmul(data_x, theta)).T
    # loss
    minimal_loss2 = np.mean(np.log(1 + temp)) + lam/2 * LA.norm(theta)**2
    # gradient
    gradient = np.mean((- data_y * data_x * temp)/(1 + temp), axis=0) + lam * theta


''' %%%%%%%%%  SGD function  %%%%%%%%%% '''
def SGD_func(data_x, data_y, lr=0.01,lam=0.0, max_it=15000, epsilon=0.0, batch_size=50):
    if lam == 0.0:
        minimal_loss = minimal_loss1
    else:
        minimal_loss = minimal_loss2

    # initialization
    dim = data_x.shape[-1]
    theta_0 = np.random.rand(dim) / 1000
    # historical data
    hist_time = np.zeros(max_it)
    hist_grad = np.zeros(max_it)
    hist_loss = np.zeros(max_it)
    loss_error = np.zeros(max_it)

    theta = theta_0
    index = random.sample(range(data_x.shape[0]), batch_size)  # a list of index
    data_input = data_x[index]
    label = data_y[index]

    temp = np.exp(- label.T * np.matmul(data_input, theta)).T
    gradient = np.mean((- label * data_input * temp)/(1 + temp), axis=0) + lam * theta
    temp2 = np.exp(- data_y.T * np.matmul(data_x, theta)).T
    loss = np.mean(np.log(1 + temp2)) + lam/2 * LA.norm(theta)**2
    hist_grad[0] = LA.norm(gradient)
    hist_loss[0] = loss
    loss_error[0] = loss - minimal_loss

    # record time
    time_ini = time.time()
    for i in range(max_it):
        # stopping criteria
        if LA.norm(gradient) < epsilon:
            break
        # update
        theta = theta - lr * gradient
        index = random.sample(range(data_x.shape[0]), batch_size)
        data_input = data_x[index]
        label = data_y[index]
        temp = np.exp(-label.T * np.matmul(data_input, theta)).T
        gradient = np.mean((-label * data_input * temp)/(1 + temp), axis=0) + lam * theta
        hist_grad[i] = LA.norm(gradient)
        temp2 = np.exp(-data_y.T * np.matmul(data_x, theta)).T
        loss = np.mean(np.log(1 + temp2)) + lam/2 * LA.norm(theta)**2
        hist_loss[i] = loss 
        loss_error[i] = loss - minimal_loss
        hist_time[i] = time.time() - time_ini
    total_time = time.time() - time_ini
    return hist_loss, hist_grad, loss_error, hist_time, total_time


''' %%%%%%%%%  EXPERIMENTS: GD, SGD, Mini-batch SGD  %%%%%%%%%% '''
lam = 0.0
epsilon = 0.0
batch_list = [351, 1, 50]
color_list = ['red', 'blue', 'green']
max_it = 15000

hist_grad = np.zeros((len(batch_list), max_it))
hist_loss = np.zeros((len(batch_list), max_it))
hist_err = np.zeros((len(batch_list), max_it))
hist_time = np.zeros((len(batch_list), max_it))
time_rec = np.zeros(len(batch_list))


''' %%%%%%%%%%%%%%%%%%  Plots  %%%%%%%%%%%%%%%%%%% '''
for i in range(len(batch_list)):
    hist_loss[i], hist_grad[i], hist_err[i], hist_time[i], time_rec[i] = SGD_func(data_x, data_y, lr=lr, max_it=max_it, lam=lam, epsilon=epsilon, batch_size=batch_list[i])
plt.figure()
plt.grid()
for i in range(len(batch_list)):
    plt.plot(hist_err[i], color=color_list[i])
plt.legend(['GD', 'SGD', 'Mini-batch SGD'])
plt.xlabel('# of iteration')
plt.ylabel('Objective error')
plt.title('Objective error vs # of iteration (lambda = 0)')
plt.savefig('./iteration_lam0.png', dpi=700)

plt.figure()
plt.grid()
for i in range(len(batch_list)):
    plt.plot(hist_time[i], hist_err[i], color=color_list[i])
plt.legend(['GD', 'SGD', 'Mini-batch SGD'])
plt.xlabel('CPU time')
plt.ylabel('Objective error')
plt.title('Objective error vs CPU time (lambda = 0)')
plt.savefig('./CPUtime_lam0.png', dpi=700)


lam = 0.01
for i in range(len(batch_list)):
    hist_loss[i], hist_grad[i], hist_err[i], hist_time[i], time_rec[i] = SGD_func(data_x, data_y, lr=lr, max_it=max_it, lam=lam, epsilon=epsilon, batch_size=batch_list[i])
plt.figure()
plt.grid()
for i in range(len(batch_list)):
    plt.plot(hist_err[i], color=color_list[i])
plt.legend(['GD', 'SGD', 'Mini-batch SGD'])
plt.xlabel('# of iteration')
plt.ylabel('Objective error')
plt.title('Objective error vs # of iteration (lambda = 0.01)')
plt.savefig('./iteration_lam0_01.png', dpi=700)

plt.figure()
plt.grid()
for i in range(len(batch_list)):
    plt.plot(hist_time[i], hist_err[i], color=color_list[i])
plt.legend(['GD', 'SGD', 'Mini-batch SGD'])
plt.xlabel('CPU time')
plt.ylabel('Objective error')
plt.title('Objective error vs CPU time (lambda = 0.01)')
plt.savefig('./CPUtime_lam0_01.png', dpi=700)