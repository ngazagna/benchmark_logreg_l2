import numpy as np


from benchopt.base import BaseSolver


class Solver(BaseSolver):
    name = 'Python-GD'  # gradient descent

    def set_objective(self, X, y, lmbd):
        self.X, self.y, self.lmbd = X, y, lmbd

    def initialization(self):
        n_samples, n_features = self.X.shape
        self.w = np.zeros(n_features)
        self.step = self._compute_step_size()

    def _compute_step_size(self):
        L = (np.linalg.norm(self.X) ** 2 / 4) + self.lmbd
        step = 1. / L
        # print("step size GD: ", step)
        return step

    def run(self, n_iter):
        w = self.w

        for i in range(n_iter):
            grad = self._grad_logreg_l2(w, self.X, self.y, self.lmbd)
            w -= self.step * grad  # GD step

        self.w = w

    def _grad_logreg_l2(self, w, X, y, lmbd):
        return -y * X.T @ (1. / (1. + np.exp(y * (X @ w)))) + lmbd * w

    def get_result(self):
        return self.w
