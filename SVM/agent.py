import numpy as np
import math


class Pre:
    def __init__(self, X, ad_in=True):
        X = np.copy(X)
        self.num_obs, self.num_features = X.shape
        self.if_add_intercept = ad_in
        X = self.rm_zeroVar(X)
        X = self.std(X)

    def rm_zeroVar(self, X):
        std_array = np.std(X, axis=0)
        self.valid_col_index = std_array > 1e-03
        return X[:, self.valid_col_index]

    def std(self, X):
        self.mean_array = np.mean(X, axis=0)
        self.std_array = np.std(X, axis=0, ddof=1)
        return (X - self.mean_array) / self.std_array

    def intercept(self, X):
        return np.concatenate((np.ones((self.num_obs, 1)), X), axis=1)

    def predict(self, X):
        assert (X.shape[1] == self.num_features)
        X = X[:, self.valid_col_index]
        X = (X - self.mean_array) / self.std_array
        if self.if_add_intercept:
            X = self.intercept(X)
        return X


class SVM():
    def __init__(self, X, y, regu_lam, k):
        assert k > 0
        self.data_pre = Pre(X)
        X = self.data_pre.predict(X)
        y = np.copy(y)
        self.y_vals = np.unique(y)
        y[y == self.y_vals[0]] = -1
        y[y == self.y_vals[1]] = 1
        self.loglist = []
        self.weight = self.cal_weigh(X, y, regu_lam, k)

    def predict(self, X, output=0):
        X = self.data_pre.predict(X)
        predicted_score = self.score(X)
        predicted_class = self.preclass(predicted_score)
        return predicted_class

    def validate(self, X, y, output=0):
        X = self.data_pre.predict(X)
        predicted_score = self.score(X)
        predicted_class = self.preclass(predicted_score)
        prediction_error = self.error(predicted_class, y)
        return prediction_error

    def score(self, X):
        return np.dot(X, self.weight)

    def preclass(self, predicted_score):
        return [self.y_vals[0] if s < 0 else self.y_vals[1] for s in predicted_score]

    def error(self, predicted_class, y):
        predicted_indicator = np.array(
            [predicted_class[i] == y[i] for i in range(0, y.size)])
        return 1 - sum(predicted_indicator) / y.size

    def cal_weigh(self, X, y, regu_lam, k):
        n, p = X.shape
        weight = self.initial(p, regu_lam)
        for i in range(1, 10000):
            X_work, y_work = self.select(X, y, weight, k)
            self.loglist.append(self.loss(
                X, y, weight, regu_lam))
            weight_new = self.weigh(
                X_work, y_work, weight, regu_lam, k, i)
            if sum((weight_new - weight) ** 2) < 0.01:
                break
            else:
                weight = weight_new
        return weight

    def select(self, X, y, weight, k):
        n, p = X.shape
        index = np.array([])
        while index.size == 0:
            index = np.random.choice(n, k)
            X_sub = X[index, :]
            y_sub = y[index]
            sub_index = (np.dot(X_sub, weight) * y_sub) < 1
            index = index[sub_index]
        return (X[index, :], y[index])

    def initial(self, p, regu_lam):
        weight = np.zeros(p)
        weight.fill(math.sqrt(1 / (p * regu_lam)))
        neg_index = np.random.choice(p, size=(int)(p / 2))
        weight[neg_index] = -weight[neg_index]
        return weight

    def weigh(self, X, y, weight, regu_lam, k, iter_num):
        eta = 1 / (regu_lam * iter_num)  # step size
        weight_half = (1 - eta * regu_lam) * weight + eta / k * np.dot(y, X)
        if sum(weight_half ** 2) < 1e-07:
            weight_half = np.maximum(weight_half, 1e-04)
        weight_new = np.minimum(1, 1 / math.sqrt(regu_lam) /
                             math.sqrt(sum(weight_half ** 2))) * weight_half
        return weight_new

    def loss(self, X, y, weight, regu_lam):
        n, p = X.shape
        tmp_loss = 1 - y * np.dot(X, weight)
        loss = sum(tmp_loss[tmp_loss > 0]) / n + \
            regu_lam / 2 * np.dot(weight, weight)
        return loss