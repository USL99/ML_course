import numpy as np
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
        loss_function: LossFunction = LossFunction.MSE
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

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.w is None:
            raise ValueError("Model is not fitted yet")
        return X @ self.w

        raise NotImplementedError("predict function is not implemented")

    def compute_gradients(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        if self.w is None:
            raise ValueError("Weight are not initialized")

        if self.loss_function is LossFunction.MSE:
            n = X.shape[0]
            r = (X @ self.w) - y
            g = (2.0 / n) * (X.T @ r)
            if self.l2_coef !=0.0:
                g = g + 2.0 * self.l2_coef * self.w
                return g
            raise NotImplementedError("MSE gradients is not implemented")
        # # elif self.loss_function is ...
        # return None

    def compute_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        if self.w is None:
            raise ValueError("Weight are not initialized")

        if self.loss_function is LossFunction.MSE:
            MSE = float(np.mean((X @ self.w - y) ** 2))
            reg = float(self.l2_coef * np.sum(self.w ** 2))
            return MSE + reg

        raise NotImplementedError("MSE is not implemented")
        # # elif self.loss_function is ...
        # return 0.0

    def fit(self, X: np.ndarray, y: np.ndarray):
        # TODO: реализовать обучение модели
        self.X_train, self.y_train = X, y
        n, d = X.shape

        self.w = np.zeros(d, dtype = float)
        self._sigma2 = float(np.linalg.norm(X, ord = 2) ** 2)

        if self.optimizer in None:
            # Analytic solution
            pass
        elif isinstance(self.optimizer, BaseDescent):
            # ...
            for _ in range(self.max_iter):
                # 1 шаг градиентного спуска
                _
        # elif self.optimizer is ...
        raise NotImplementedError("Linear Regression training is not implemented")
