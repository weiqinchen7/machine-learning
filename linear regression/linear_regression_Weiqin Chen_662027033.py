import random
import time
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import pandas as pd


''' %%%%%%%%%  hyper-parameters  %%%%%%%%%% '''
# maximal number of iterations
max_it = 2000
# learning rate
lr = 0.00000001


''' %%%%%%%%%  read data  %%%%%%%%%% '''
dataset = pd.read_excel('./AirQualityUCI.xlsx')
length = len(dataset)
data_x = [dataset.iloc[i].tolist()[:-1] for i in range(length)]
data_x = np.asarray(data_x)  # (9357, 12)
data_y = [dataset.iloc[i].tolist()[-1] for i in range(length)]
data_y = np.asarray(data_y)[:, np.newaxis]  # (9357, 1)


''' %%%%%%%%%  loss, gradient, minimal loss  %%%%%%%%%% '''
# dimension of data_x
dim = data_x.shape[-1]
# initialization of parameters
theta_0 = np.random.rand(dim)[:, np.newaxis] / 100000  # (12, 1)
theta = theta_0
temp = data_y - np.dot(data_x, theta)
loss = np.mean(np.dot(temp.T, temp) / data_x.shape[0])
gradient = -2 * np.dot(data_x.T, temp) / data_x.shape[0]  # (12, 1)
for i in range(max_it):
    # update theta by GD
    theta = theta - lr * gradient
    temp = data_y - np.dot(data_x, theta)
    # loss
    minimal_loss = np.mean(np.dot(temp.T, temp) / data_x.shape[0])
    # gradient
    gradient = -2 * np.dot(data_x.T, temp) / data_x.shape[0]

''' %%%%%%%%%  SGD function  %%%%%%%%%% '''
def SGD_func(data_x, data_y, lr=0.00000001, max_it=3000, epsilon=0.0, batch_size=50, minimal_loss=minimal_loss):
    # initialization
    dim = data_x.shape[-1]
    theta_0 = np.random.rand(dim)[:, np.newaxis] / 100000
    # historical data
    hist_time = np.zeros(max_it)
    hist_grad = np.zeros(max_it)
    hist_loss = np.zeros(max_it)
    loss_error = np.zeros(max_it)

    theta = theta_0
    index = random.sample(range(data_x.shape[0]), batch_size)  # a list of index
    data_input = data_x[index]  # (50, 12)
    label = data_y[index]  # (50, 1)
    temp = data_y - np.dot(data_x, theta)
    temp1 = label - np.dot(data_input, theta)

    gradient = -2 * np.dot(data_input.T, temp1) / data_input.shape[0]
    loss = np.mean(np.dot(temp.T, temp) / data_x.shape[0])
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
        temp = data_y - np.dot(data_x, theta)
        temp1 = label - np.dot(data_input, theta)

        gradient = -2 * np.dot(data_input.T, temp1) / data_input.shape[0]
        hist_grad[i] = LA.norm(gradient)
        loss = np.mean(np.dot(temp.T, temp) / data_x.shape[0])
        hist_loss[i] = loss
        loss_error[i] = loss - minimal_loss
        hist_time[i] = time.time() - time_ini
    total_time = time.time() - time_ini
    return hist_loss, hist_grad, loss_error, hist_time, total_time


''' %%%%%%%%%  EXPERIMENTS: GD, SGD, Mini-batch SGD  %%%%%%%%%% '''
epsilon = 0.0
batch_list = [data_x.shape[0], 1, 50]
color_list = ['red', 'blue', 'green']

hist_grad = np.zeros((len(batch_list), max_it))
hist_loss = np.zeros((len(batch_list), max_it))
hist_err = np.zeros((len(batch_list), max_it))
hist_time = np.zeros((len(batch_list), max_it))
time_rec = np.zeros(len(batch_list))


''' %%%%%%%%%%%%%%%%%%  Plots  %%%%%%%%%%%%%%%%%%% '''
for i in range(len(batch_list)):
    hist_loss[i], hist_grad[i], hist_err[i], hist_time[i], time_rec[i] = SGD_func(data_x, data_y, lr=lr, max_it=max_it, epsilon=epsilon, batch_size=batch_list[i], minimal_loss=minimal_loss)
plt.figure()
plt.grid()
for i in range(len(batch_list)):
    plt.plot(hist_err[i], color=color_list[i])
plt.legend(['GD', 'SGD', 'Mini-batch SGD'])
plt.xlabel('# of iteration')
plt.ylabel('Objective error')
plt.title('Objective error vs # of iteration')
plt.savefig('./iteration.png', dpi=700)

plt.figure()
plt.grid()
for i in range(len(batch_list)):
    plt.plot(hist_time[i], hist_err[i], color=color_list[i])
plt.legend(['GD', 'SGD', 'Mini-batch SGD'])
plt.xlabel('CPU time')
plt.ylabel('Objective error')
plt.title('Objective error vs CPU time')
plt.savefig('./CPUtime.png', dpi=700)