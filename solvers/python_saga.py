import time
import numpy as np


from benchopt.base import BaseSolver


class Solver(BaseSolver):
    name = 'Python-SAGA'

    def set_objective(self, X, y, lmbd):
        self.X, self.y, self.lmbd = X, y, lmbd

    def init(self):
        # print("---------------------------------- Initializing SAGA weights and gradients table")
        n_samples, n_features = self.X.shape
        self.w = np.zeros(n_features)

        L = np.max(np.sum(self.X ** 2, axis=1)) / 4 + self.lmbd
        self.step = 1. / L
        self.memory_grad = np.zeros((n_samples, n_features))
        # print("self.memory_grad: ", self.memory_grad.shape)
        self.grad_mean = self.memory_grad.mean(axis=0)
        # print("self.grad_mean: ", self.grad_mean.shape)

    def run(self, n_iter):
        n_samples, n_features = self.X.shape
        w = self.w

        t_new = 1
        for i in range(n_iter):
            idx = np.random.choice(n_samples)
            grad_i = self.grad_i_logreg_l2(w, self.X, self.y, self.lmbd, idx)
            w -= self.step * (grad_i - self.memory_grad[i] + self.grad_mean)

            self.grad_mean += (grad_i - self.memory_grad[i]) / n_samples
            self.memory_grad[i] = grad_i

        self.w = w

    def grad_i_logreg_l2(self, w, X, y, lmbd, i):
        return self.grad_i_logreg(w, X, y, i) + lmbd * w

    def grad_i_logreg(self, w, X, y, i):
        return - X[i] * y[i] / (1. + np.exp(y[i] * (X[i] @ w)))

    def get_result(self):
        return self.w