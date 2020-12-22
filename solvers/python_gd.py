import time
import numpy as np


from benchopt.base import BaseSolver


class Solver(BaseSolver):
    name = 'Python-GD'  # gradient descent

    def set_objective(self, X, y, lmbd):
        self.X, self.y, self.lmbd = X, y, lmbd

    def init(self):
        # print("---------------------------------- Initializing GD weights")
        n_samples, n_features = self.X.shape
        self.w = np.zeros(n_features)

    def run(self, n_iter):
        time.sleep(.1)
        n_samples, n_features = self.X.shape
        # w = np.zeros(n_features)
        w = self.w

        L = (np.linalg.norm(self.X) ** 2 / 4) + self.lmbd
        step = 1. / L

        t_new = 1
        for i in range(n_iter):
            w -= step * self.grad_logreg_l2(w, self.X, self.y, self.lmbd) # GD step

        self.w = w

    def grad_logreg_l2(self, w, X, y, lmbd):
        return (-y * X.T @ (1. / (1. + np.exp(y * (X @ w)))) / X.shape[0]) + lmbd * w

    def get_result(self):
        return self.w