import numpy as np
from linear_regression import LinearRegression
import pytest
from descents import BaseDescent, VanillaGradientDescent # StochasticGradientDescent

from sklearn.metrics import mean_squared_error as mse
import sklearn

num_objects = 100
dimension = 5

x = np.random.rand(num_objects, dimension)
y = np.random.rand(num_objects)

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

    # def test_stochastic_grad_descent(self):
    #     sklearn_linreg = sklearn.linear_model.LinearRegression(fit_intercept=False)
    #     sklearn_linreg.fit(x, y)
    #     print("Sklearn MSE", mse(sklearn_linreg.predict(x), y))
    #
    #     optimizer = StochasticGradientDescent(batch_size=30)
    #     your_linreg = LinearRegression(optimizer=optimizer, tolerance=1e-12)
    #     your_linreg.fit(x, y)
    #     print("Your MSE", mse(your_linreg.predict(x), y))
    #
    #     assert abs(mse(your_linreg.predict(x), y) - mse(sklearn_linreg.predict(x),
    #                                                               y)) < 1e-3, "Не повезло, попробуйте еще раз"
    #     return True

