import numpy as np
import os
from descents import BaseDescent
from dataclasses import dataclass
from enum import auto, Enum
from typing import Dict, Type, Optional


class LossFunction(Enum):
    MSE = auto()
    MAE = auto()
    LogCosh = auto()
    Huber = auto()

class LinearRegression:
    def __init__(
        self,
        optimizer: Optional[BaseDescent | str] = None,
        l2_coef: float = 0.0,
        tolerance: float = 1e-6,
        max_iter: int = 1000,
        loss_function: LossFunction = LossFunction.MSE,
        verbose: bool = False, print_every: int = 10
    ):
        self.optimizer = optimizer
        if isinstance(optimizer, BaseDescent):
            self.optimizer.set_model(self)
        self.l2_coef = l2_coef
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.loss_function = loss_function
        self.w = None
        self.X_train = None
        self.y_train = None
        self.loss_history = []
        self.verbose = verbose
        self.print_every = print_every
        self.verbose = verbose or (os.getenv("LINREG_DEBUG") == "1")
        self.print_every = int(os.getenv("LINREG_PRINT_EVERY", str(print_every)))

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.w is None:
            raise ValueError("Model is not fitted yet")
        return X @ self.w


    def compute_gradients(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        if self.w is None:
            raise ValueError("Weights are not initialized")

        if self.loss_function is LossFunction.MSE:
            n = X.shape[0]
            r = (X @ self.w) - y
            g = (2.0 / n) * (X.T @ r)
            if self.l2_coef !=0.0:
                g = g + 2.0 * self.l2_coef * self.w
            return g

        # # elif self.loss_function is ...
        # return None
        raise NotImplementedError(f"Gradients for {self.loss_function} are not implemented")

    def compute_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        if self.w is None:
            raise ValueError("Weight are not initialized")

        if self.loss_function is LossFunction.MSE:
            MSE = float(np.mean((X @ self.w - y) ** 2))
            reg = float(self.l2_coef * np.sum(self.w ** 2))
            return MSE + reg

        raise NotImplementedError(f"Loss {self.loss_function} is not implemented")
        # # elif self.loss_function is ...
        # return 0.0

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.X_train, self.y_train = X, y
        if self.verbose:
            print(f"[fit] X_shape={X.shape}, y_shape={y.shape}, l2={self.l2_coef}")
        n, d = X.shape

        self.w = np.zeros(d, dtype = float)
        self._sigma2 = float(np.linalg.norm(X, ord = 2) ** 2)

        if self.optimizer is None:

            if self.l2_coef == 0.0:
                self.w = np.linalg.lstsq(X, y, rcond = None)[0]
            else:
                A = X.T @ X + (n * self.l2_coef) * np.eye(d)
                b = X.T @ y
                self.w = np.linalg.solve(A, b)

            self.loss_history = [self.compute_loss(X, y)]
            if self.verbose:
                loss = self.loss_history[-1]
                print(f"[analytic] loss={loss}, ||w|| = {np.linalg.norm(self.w)}, w[:5] = {self.w[:5]}")

            return self

        elif isinstance(self.optimizer, BaseDescent):
            self.optimizer.set_model(self)
            self.loss_history = []
            prev = self.compute_loss(X, y)
            self.loss_history.append(prev)

            for _ in range(self.max_iter):
                self.optimizer.step()
                tot = self.compute_loss(X, y)
                self.loss_history.append(tot)
                if abs(prev - tot) < self.tolerance:
                    break
                prev = tot

            return self

                # elif self.optimizer is ...

