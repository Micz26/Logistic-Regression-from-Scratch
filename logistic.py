import numpy as np
import pandas as pd
from math import e
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

class CustomLogisticRegression:

    def __init__(self, fit_intercept=True, l_rate=0.01, n_epoch=100):
        self.fit_intercept = fit_intercept
        self.l_rate = l_rate
        self.n_epoch = n_epoch

    def sigmoid(self, t):
        return 1 / (1 + e ** (-t))

    def predict_proba(self, row, coef_):
        if len(row) == 3:
            t = coef_[0] + np.dot(coef_[1:], row)
        elif len(row) == 4:
            t = coef_[0] + np.dot(coef_[1:], row[1:])
        return self.sigmoid(t)

    def fit_mse(self, X_train, y_train):
        if self.fit_intercept:
            X_train = np.column_stack((np.ones(X_train.shape[0]), X_train))

        n_samples, n_features = X_train.shape
        self.coef_ = np.random.random(n_features)

        for _ in range(self.n_epoch):
            for i, row in enumerate(X_train):
                y_hat = self.predict_proba(row, self.coef_)
                y_true = y_train[i]
                if len(row) == 3:
                    self.coef_[0] = self.coef_[0] - self.l_rate * (y_hat - y_true) * y_hat * (1 - y_hat)
                    for p in range(1, len(self.coef_)):
                        self.coef_[p] = self.coef_[p] - self.l_rate * (y_hat - y_true) * y_hat * (1 - y_hat) * row[p - 1]
                elif len(row) == 4:
                    self.coef_[0] = self.coef_[0] - self.l_rate * (y_hat - y_true) * y_hat * (1 - y_hat)
                    for p in range(1, len(self.coef_)):
                        self.coef_[p] = self.coef_[p] - self.l_rate * (y_hat - y_true) * y_hat * (1 - y_hat) * row[p]

    def fit_log_loss(self, X_train, y_train):
        if self.fit_intercept:
            X_train = np.column_stack((np.ones(X_train.shape[0]), X_train))

        n_samples, n_features = X_train.shape
        self.coef_ = np.random.random(n_features)

        for _ in range(self.n_epoch):
            for i, row in enumerate(X_train):
                y_hat = self.predict_proba(row, self.coef_)
                y_true = y_train[i]
                if len(row) == 3:
                    self.coef_[0] = self.coef_[0] - self.l_rate * (y_hat - y_true) / n_samples
                    for p in range(1, len(self.coef_)):
                        self.coef_[p] = self.coef_[p] - self.l_rate * (y_hat - y_true) / n_samples * row[p - 1]
                elif len(row) == 4:
                    self.coef_[0] = self.coef_[0] - self.l_rate * (y_hat - y_true) / n_samples
                    for p in range(1, len(self.coef_)):
                        self.coef_[p] = self.coef_[p] - self.l_rate * (y_hat - y_true) / n_samples * row[p]

    def predict(self, X_test, cut_off=0.5):
        predictions = []
        for row in X_test:
            y_hat = self.predict_proba(row, self.coef_)
            if y_hat < cut_off:
                predictions.append(0)
            else:
                predictions.append(1)
        return np.array(predictions)

def main():
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    selected_features = ['worst concave points', 'worst perimeter', 'worst radius']
    X = df[selected_features]
    y = data.target

    if isinstance(X, pd.DataFrame):
        X = X.values
    if isinstance(y, pd.Series):
        y = y.values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=43)
    lr = CustomLogisticRegression(fit_intercept=True, l_rate=0.01, n_epoch=1000)
    print(X_train)
    try:
        lr.fit_log_loss(X_train, y_train)
        y_pred = lr.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        result_dict = {
            'coef_': lr.coef_.tolist(),
            'accuracy': accuracy
        }

        print(result_dict)

    except np.core._exceptions.UFuncTypeError:
        print("An error occurred during fitting. Please ensure that the row array has numeric values.")


if __name__ == "__main__":
    main()
