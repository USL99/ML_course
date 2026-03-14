import numpy as np
from abc import ABC, abstractmethod


class LearningRateSchedule(ABC):
    @abstractmethod
    def get_lr(self, iteration: int) -> float:
        pass


class ConstantLR(LearningRateSchedule):
    def __init__(self, lr: float):
        self.lr = lr

    def get_lr(self, iteration: int) -> float:
        return self.lr


class TimeDecayLR(LearningRateSchedule):
    def __init__(self, lambda_: float = 1.0):
        self.s0 = 1
        self.p = 0.5
        self.lambda_ = lambda_

    def get_lr(self, iteration: int) -> float:
        lr_t = float(self.s0 / (1 + self.lambda_ * iteration) ** self.p)
        return lr_t

class BaseDescent(ABC):
    def __init__(self, lr_schedule: LearningRateSchedule = TimeDecayLR):
        self.lr_schedule = lr_schedule()
        self.iteration = 0
        self.model = None

    def set_model(self, model):
        self.model = model

    @abstractmethod
    def update_weights(self):
        pass

    def step(self):
        self.update_weights()
        self.iteration += 1


class VanillaGradientDescent(BaseDescent):
    def update_weights(self):
        X_train = self.model.X_train
        y_train = self.model.y_train
        lr = float(self.lr_schedule.get_lr(self.iteration))
        gradient = self.model.compute_gradients(X_train, y_train)
        self.model.w = self.model.w - lr * gradient

        if getattr(self.model, "verbose", False) and (
                self.iteration < 5 or self.iteration % getattr(self.model, "print_every", 10) == 0
        ):
            print(
                f"[GD] iter={self.iteration:04d} lr={lr:.3g} "
                f"||grad||={np.linalg.norm(gradient):.3g} ||w||={np.linalg.norm(self.model.w):.3g} "
                f"w[:5]={self.model.w[:5]}"
            )


class StochasticGradientDescent(BaseDescent):
    def __init__(self, lr_schedule: LearningRateSchedule = TimeDecayLR, batch_size=1):
        super().__init__(lr_schedule)
        self.batch_size = batch_size

    def update_weights(self):
        X_train = self.model.X_train
        y_train = self.model.y_train
        n = X_train.shape[0]
        batch_s = min (self.batch_size, n)
        index_b = np.random.choice(n, size = batch_s, replace = False)
        X_bs = X_train[index_b]
        y_bs = y_train[index_b]

        lr = float(self.lr_schedule.get_lr(self.iteration))
        gradient = self.model.compute_gradients(X_bs, y_bs)
        self.model.w = self.model.w - lr * gradient


class SAGDescent(BaseDescent):
    def __init__(self, lr_schedule: LearningRateSchedule = TimeDecayLR, batch_size: int = 1):
        super().__init__(lr_schedule)
        self.batch_size = int (batch_size)
        self.grad_memory = None
        self.grad_sum = None

    def update_weights(self):
        X_train = self.model.X_train
        y_train = self.model.y_train
        n, d = X_train.shape

        if self.grad_memory is None:
            self.grad_memory = np.zeros((n, d), dtype = float)
            self.grad_sum = np.zeros(d, dtype = float)

        ind = np.random.randint(0, n, size = self.batch_size)

        for j in np.atleast_ld(ind):
            g_old = self.grad_memory[j].copy()
            g_new = self.model.compute_gradients(X_train[j:j + 1], y_train[j:j + 1])  # градиент по 1 объекту
            self.grad_memory[j] = g_new
            self.grad_sum += (g_new - g_old)

        avg_grad = self.grad_sum / n

        lr = float(self.lr_schedule.get_lr(self.iteration))
        self.model.w = self.model.w - lr * avg_grad

        if getattr(self.model, "verbose", False) and (
                self.iteration < 5 or self.iteration % getattr(self.model, "print_every", 10) == 0
        ):
            print(f"[SAG] iter={self.iteration:04d} lr={lr:.3g} "
                  f"||avg_grad||={np.linalg.norm(avg_grad):.3g} ||w||={np.linalg.norm(self.model.w):.3g}")


class MomentumDescent(BaseDescent):
    def __init__(self, lr_schedule: LearningRateSchedule = TimeDecayLR, beta=0.9):
        super().__init__(lr_schedule)
        self.beta = float(beta)
        self.velocity = None

    def update_weights(self):
        X_train = self.model.X_train
        y_train = self.model.y_train

        grad =  self.model.compute_gradients(X_train, y_train)

        if self.velocity is None:
            self.velocity = np.zeros_like(self.model.w)

        lr = float(self.lr_schedule.get_lr(self.iteration))

        self.velocity = self.beta * self.velocity + lr * grad

        self.model.w = self.model.w - self.velocity

        if getattr(self.model, "verbose", False) and (
                self.iteration < 5 or self.iteration % getattr(self.model, "print_every", 10) == 0
        ):
            print(f"[Momentum] iter={self.iteration:04d} lr={lr:.3g} "
                  f"||grad||={np.linalg.norm(grad):.3g} ||v||={np.linalg.norm(self.velocity):.3g} "
                  f"||w||={np.linalg.norm(self.model.w):.3g}")

# class Adam(BaseDescent):
#     def __init__(self, lr_schedule: LearningRateSchedule = TimeDecayLR, beta1=0.9, beta2=0.999, eps=1e-8):
#         super().__init__(lr_schedule)
#         self.beta1 = beta1
#         self.beta2 = beta2
#         self.eps = eps
#         self.m = None
#         self.v = None
#
#     def update_weights(self):
#         # TODO: реализовать Adam по формуле из ноутбука
#         raise NotImplementedError