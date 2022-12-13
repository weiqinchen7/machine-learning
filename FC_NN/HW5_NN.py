import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from numpy import genfromtxt


# hyper-parameters
num_epoch = 200
lr = 1e-3
np.random.seed(10)
num_exp = 100


class One_hidden_NN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, activation):
        super(One_hidden_NN, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.activation = activation

    def forward(self, x):
        if self.activation == 'relu':
            x = torch.relu(self.linear1(x))
        elif self.activation == 'tanh':
            x = torch.tanh(self.linear1(x))

        x = torch.sigmoid(self.linear2(x))
        return x


class Two_hidden_NN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, activation):
        super(Two_hidden_NN, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)
        self.activation = activation

    def forward(self, x):
        if self.activation == 'relu':
            x = torch.relu(self.linear1(x))
            x = torch.relu(self.linear2(x))
        elif self.activation == 'tanh':
            x = torch.tanh(self.linear1(x))
            x = torch.tanh(self.linear2(x))

        x = torch.sigmoid(self.linear3(x))
        return x



def train_test(data, data_test, num_epoch, num_layer, hidden_size, activation, optim, lr):
    data_x = data[:, :-1]
    data_y = data[:, -1]
    data_x = torch.FloatTensor(data_x)
    data_y = torch.FloatTensor(data_y)

    data_test_x = data_test[:, :-1]
    data_test_y = data_test[:, -1]
    data_test_x = torch.FloatTensor(data_test_x)
    input_size = data_x.shape[1]
    output_size = 1


    if num_layer == 1:
        model = One_hidden_NN(input_size, hidden_size, output_size, activation)
    elif num_layer == 2:
        model = Two_hidden_NN(input_size, hidden_size, output_size, activation)

    if optim == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    elif optim == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for i in range(num_epoch):
        y_pred = model(data_x)

        loss_fn = nn.MSELoss()
        loss = loss_fn(y_pred, data_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # test
    y_pred_test = model(data_test_x)
    num = y_pred_test.shape[0]
    prediction = np.zeros(num)
    for i in range(num):
        if float(y_pred_test[i]) >= 1/2:
            prediction[i] = 1
        else:
            prediction[i] = 0

    error_rate = (np.linalg.norm(data_test_y - prediction) ** 2) / num
    return error_rate

def dataset_split(dataset):
    total_num = dataset.shape[0]
    test_num = int(total_num / 5)
    # the index that test data starts
    test_start = int(np.random.randint(0, total_num - test_num))
    # test data at random
    test_data = dataset[test_start:test_start + test_num, :]
    # the remaining of excluding test data is training data
    train_data = np.vstack((dataset[:test_start, :], dataset[test_start + test_num:, :]))
    return train_data, test_data

def train_epoc(dataset, train_percent, num_layer, hidden_size, activation, lr):

    error_arr_sgd = np.zeros((len(train_percent), num_exp))
    error_arr_adam = np.zeros((len(train_percent), num_exp))
    for i, percent in enumerate(train_percent):
        for j in range(num_exp):
            train_data, test_data = dataset_split(dataset)
            # number of training percent
            num_train_percent = int(train_data.shape[0] * (percent/100))
            # new training set after split
            train_data_split = train_data[:num_train_percent, :]

            error_arr_sgd[i, j] = train_test(train_data_split, test_data, num_epoch, num_layer, hidden_size, activation, 'sgd', lr)
            error_arr_adam[i, j] = train_test(train_data_split, test_data, num_epoch, num_layer, hidden_size, activation, 'adam', lr)
    error_sgd_mean = np.mean(error_arr_sgd, axis=1)
    error_sgd_std = np.std(error_arr_sgd, axis=1)
    error_adam_mean = np.mean(error_arr_adam, axis=1)
    error_adam_std = np.std(error_arr_adam, axis=1)

    return error_sgd_mean, error_sgd_std, error_adam_mean, error_adam_std


train_percent = [10, 20, 30]
num_layer = [1, 1, 1, 2]
hidden_size = [10, 10, 30, 10]
activation = ['relu', 'tanh', 'relu', 'relu']
dataset = genfromtxt('spam.csv', delimiter=',')

for i in range(4):
    error_sgd_mean, error_sgd_std, error_adam_mean, error_adam_std = train_epoc(dataset, train_percent, num_layer[i], hidden_size[i], activation[i], lr)
    plt.figure()
    plt.ylim([0, 1])
    plt.grid()
    plt.errorbar(train_percent, error_sgd_mean, yerr=error_sgd_std, label='SGD', fmt='-o', capthick=2)
    plt.errorbar(train_percent, error_adam_mean, yerr=error_adam_std, label='ADAM', fmt='--o', color='r', capthick=2)
    plt.legend()
    plt.ylabel('Test Error Rate')
    plt.xlabel('Training Percent')
    plt.title('Model{}: SGD V.S. ADAM'.format(i+2), fontsize=18, verticalalignment='bottom')
    plt.savefig('model{}.png'.format(i+2), bbox_inches='tight', dpi=600)