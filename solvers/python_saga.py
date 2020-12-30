import numpy as np


from benchopt.base import BaseSolver


class Solver(BaseSolver):
    name = 'Python-SAGA'

    def set_objective(self, X, y, lmbd):
        self.X, self.y, self.lmbd = X, y, lmbd

    def initialization(self):
        n_samples, n_features = self.X.shape
        self.w = np.zeros(n_features)
        self.step = self._compute_step_size()

        self.memory_grad = np.zeros((n_samples, n_features))
        for i in range(n_samples):
            self.memory_grad[i] = self._grad_i_logreg_l2(self.w,
                                                         self.X,
                                                         self.y,
                                                         self.lmbd,
                                                         i)
        self.grad_mean = self.memory_grad.mean(axis=0)

    def _compute_step_size(self):
        Lmax = np.max(np.sum(self.X ** 2, axis=1)) / 4 + self.lmbd
        step = 1. / Lmax
        # print("step size SAGA: ", step)
        return step

    def run(self, n_iter):
        n_samples, n_features = self.X.shape
        w = self.w

        for i in range(n_iter):
            idx = np.random.choice(n_samples)
            grad_i = self._grad_i_logreg_l2(w, self.X, self.y, self.lmbd, idx)
            w -= self.step * (grad_i - self.memory_grad[idx] + self.grad_mean)

            self.grad_mean += (grad_i - self.memory_grad[idx]) / n_samples
            self.memory_grad[idx] = grad_i

        self.w = w

    def _grad_i_logreg_l2(self, w, X, y, lmbd, i):
        return self._grad_i_logreg(w, X, y, i) + lmbd * w

    def _grad_i_logreg(self, w, X, y, i):
        n_samples, n_features = X.shape
        return - n_samples * y[i] * X[i] / (1. + np.exp(y[i] * (X[i] @ w)))

    def get_result(self):
        return self.w
