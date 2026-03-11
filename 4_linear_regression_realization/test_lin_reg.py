import numpy as np
from linear_regression import LinearRegression, LossFunction
import pytest
from descents import BaseDescent, VanillaGradientDescent, StochasticGradientDescent
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
import sklearn

np.random.seed(0)
num_objects = 100
dimension = 5
x = np.random.rand(num_objects, dimension)
y = np.random.rand(num_objects)

def huber_loss_mean(y_true, y_pred, delta: float = 1.0) -> float:
    r = y_pred - y_true
    abs_r = np.abs(r)
    quad = 0.5 * (r ** 2)
    lin = delta * abs_r - 0.5 * (delta ** 2)
    return float(np.mean(np.where(abs_r < delta, quad, lin)))

def numerical_grad(loss_fn, w, eps = 1e-6):
    g = np.zeros_like(w, dtype = float)
    for i in range(w.shape[0]):
        w1 = w.copy()
        w2 = w.copy()
        w1[i] += eps
        w2[i] -= eps
        g[i] = (loss_fn(w1) - loss_fn(w2)) / (2 * eps)
    return g


class TestLinReg:

    def test_analytic_solution(self):
        sklearn_linreg = sklearn.linear_model.LinearRegression(fit_intercept=False)
        sklearn_linreg.fit(x, y)
        print("Sklearn MSE", mse(sklearn_linreg.predict(x), y))

        your_linreg = LinearRegression(optimizer=None)
        your_linreg.fit(x, y)
        print("Your MSE", mse(your_linreg.predict(x), y))

        assert abs(mse(your_linreg.predict(x), y) - mse(sklearn_linreg.predict(x), y)) < 1e-12, "Не повезло, попробуйте еще раз"
        return True

    def test_vanilla_grad_descent(self):
        sklearn_linreg = sklearn.linear_model.LinearRegression(fit_intercept=False)
        sklearn_linreg.fit(x, y)
        print("Sklearn MSE", mse(sklearn_linreg.predict(x), y))

        optimizer = VanillaGradientDescent()
        your_linreg = LinearRegression(optimizer=optimizer, tolerance=1e-12)
        your_linreg.fit(x, y)
        print("Your MSE", mse(your_linreg.predict(x), y))

        assert abs(mse(your_linreg.predict(x), y) - mse(sklearn_linreg.predict(x),
                                                                  y)) < 1e-6, "Не повезло, попробуйте еще раз"
        return True

    def test_stochastic_grad_descent(self):
        sklearn_linreg = sklearn.linear_model.LinearRegression(fit_intercept=False)
        sklearn_linreg.fit(x, y)
        print("Sklearn MSE", mse(sklearn_linreg.predict(x), y))

        optimizer = StochasticGradientDescent(batch_size=30)
        your_linreg = LinearRegression(optimizer=optimizer, tolerance=1e-12)
        your_linreg.fit(x, y)
        print("Your MSE", mse(your_linreg.predict(x), y))

        assert abs(mse(your_linreg.predict(x), y) - mse(sklearn_linreg.predict(x),
                                                                  y)) < 1e-3, "Не повезло, попробуйте еще раз"
        return True


    def test_mae_loss_value_matches_sklearn_metric(self):
        w = np.random.randn(dimension)
        model = LinearRegression(optimizer=None, loss_function=LossFunction.MAE, l2_coef=0.0)
        model.w = w

        y_pred = x @ w
        ref = mae(y, y_pred)
        got = model.compute_loss(x, y)

        assert abs(got - ref) < 1e-12

    def test_huber_loss_value_matches_reference_formula(self):
        delta = 1.0
        w = np.random.randn(dimension)

        model = LinearRegression(
            optimizer=None,
            loss_function=LossFunction.Huber,
            l2_coef=0.0,
            huber_delta=delta,   # важно!
        )
        model.w = w

        y_pred = x @ w
        ref = huber_loss_mean(y, y_pred, delta=delta)
        got = model.compute_loss(x, y)

        assert abs(got - ref) < 1e-12

    def test_mae_gradient_matches_numerical(self):
        w = np.random.randn(dimension)
        model = LinearRegression(optimizer=None, loss_function=LossFunction.MAE, l2_coef=0.0)
        model.w = w

        def loss_at(w_):
            model.w = w_
            return model.compute_loss(x, y)

        g_num = numerical_grad(loss_at, w, eps=1e-6)
        model.w = w
        g = model.compute_gradients(x, y)

        assert np.allclose(g, g_num, atol=1e-4, rtol=1e-4)

    def test_huber_gradient_matches_numerical(self):
        delta = 1.0
        w = np.random.randn(dimension)

        model = LinearRegression(
            optimizer=None,
            loss_function=LossFunction.Huber,
            l2_coef=0.0,
            huber_delta=delta,
        )
        model.w = w

        def loss_at(w_):
            model.w = w_
            return model.compute_loss(x, y)

        g_num = numerical_grad(loss_at, w, eps=1e-6)
        model.w = w
        g = model.compute_gradients(x, y)

        assert np.allclose(g, g_num, atol=1e-5, rtol=1e-5)