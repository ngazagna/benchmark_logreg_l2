import numpy as np


from benchopt.base import BaseSolver


class Solver(BaseSolver):
    name = 'Python-SGD'  # stochastic gradient descent

    # any parameter defined here is accessible as a class attribute
    parameters = {'step_init': [1.]}

    def set_objective(self, X, y, lmbd):
        self.X, self.y, self.lmbd = X, y, lmbd

    def initialization(self):
        # print("---------------------------------- Initializing SGD weights")
        n_samples, n_features = self.X.shape
        self.w = np.zeros(n_features)

    def run(self, n_iter):
        n_samples, n_features = self.X.shape
        w = self.w

        # idx_samples = np.random.choice(n_samples, n_iter)
        # steps = self.step_init / np.sqrt(1 + np.arange(1, n_iter + 1))

        for i in range(n_iter):
            # When n_iter is known in advance:
            # idx = idx_samples[i]
            # step = steps[i]

            # When n_iter is NOT known in advance:
            idx = np.random.choice(n_samples)
            step = self.step_init / np.sqrt(1 + i)

            # SGD step
            w -= step * self.grad_i_logreg_l2(w,
                                              self.X, self.y, self.lmbd, idx)

        self.w = w

    def grad_i_logreg_l2(self, w, X, y, lmbd, i):
        return self.grad_i_logreg(w, X, y, i) + lmbd * w

    def grad_i_logreg(self, w, X, y, i):
        return - X[i] * y[i] / (1. + np.exp(y[i] * (X[i] @ w)))

    def get_result(self):
        return self.w
