import numpy as np
from math import log
from scipy.stats import norm


class Data_Preprocessor:
    def __init__(self, X, if_remove_zero_variance=True, if_standarize=True):
        X = np.copy(X)
        self.if_remove_zero_variance = if_remove_zero_variance
        self.if_standarize = if_standarize
        self.num_features = X.shape[1]
        if if_remove_zero_variance:
            X = self.remove_zero_variance_features(X)
        if if_standarize:
            X = self.standardize(X)

    def remove_zero_variance_features(self, X):
        # return new X and index of valid columns
        std_array = np.std(X, axis=0)
        self.valid_col_index = std_array > 1e-03
        return X[:, self.valid_col_index]

    def standardize(self, X):
        self.mean_array = np.mean(X, axis=0)
        self.std_array = np.std(X, axis=0, ddof=1)
        return (X - self.mean_array) / self.std_array

    def predict(self, X):
        assert (X.shape[1] == self.num_features)
        if self.if_remove_zero_variance:
            X = X[:, self.valid_col_index]
        if self.if_standarize:
            X = (X - self.mean_array) / self.std_array
        return X


class Classifier:
    def __init__(self, X, y):
        print("Overload the prototype")

    def predict(self, X_new, output):
        print("Overload the prototype")

    def validate(self, X_new, y_new, output):
        print("Overload the prototype")


class Naive_Bayes(Classifier):
    def __init__(self, X, y):
        # Data_Preprocessor will copy X
        self.data_preprocessor = Data_Preprocessor(X)
        X = self.data_preprocessor.predict(X)
        y = np.copy(y)
        self.y_vals = np.unique(y)
        self.mean_array, self.std_array = self.estimate_mean_std(X, y)
        self.prior = self.calc_prior(y)

    def predict(self, X_new, output=0):
        # used for making prediction
        X_new = self.data_preprocessor.predict(X_new)
        predicted_score = self.predict_score(X_new)
        predicted_class = self.predict_class(predicted_score)
        return predicted_class

    def validate(self, X_new, y_new, output=0):
        # used for validating the prediction performance
        X_new = self.data_preprocessor.predict(X_new)
        predicted_score = self.predict_score(X_new)
        predicted_class = self.predict_class(predicted_score)
        prediction_error = self.calc_predict_error(predicted_class, y_new)
        return prediction_error

    def estimate_mean_std(self, X, y):
        num_obs, num_features = X.shape
        num_classes = self.y_vals.size
        mean_array = np.zeros((num_classes, num_features))
        std_array = np.zeros((num_classes, num_features))
        for k in range(0, num_classes):
            index = y == self.y_vals[k]
            X_sub = X[index, :]
            mean_array[k, :] = np.mean(X_sub, axis=0)
            std_array[k, :] = np.std(X_sub, axis=0, ddof=1)
            std_array[std_array < 1e-03] = 1e-03
        return (mean_array, std_array)

    def calc_prior(self, y):
        prior = [np.sum(y == y_val) / y.size for y_val in self.y_vals]
        return prior

    def predict_score(self, X_new):
        num_obs, num_features = X_new.shape
        num_classes = self.mean_array.shape[0]
        ans = np.zeros((num_obs, num_classes))
        for k in range(num_classes):
            for j in range(num_features):
                ans[:, k] += norm.logpdf(X_new[:, j], loc=self.mean_array[k, j],
                                         scale=self.std_array[k, j])
        log_prior = [log(p) for p in self.prior]
        ans += log_prior
        return ans

    def predict_class(self, predicted_score):
        max_indicator = np.argmax(predicted_score, axis=1)
        return np.array([self.y_vals[i] for i in max_indicator])

    def calc_predict_error(self, predicted_class, y):
        predicted_indicator = np.array([predicted_class[i] == y[i]
                                        for i in range(0, y.size)])
        return 1 - np.sum(predicted_indicator) / y.size