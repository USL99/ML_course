import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")


def f(x, y):
    return x**2 + y**2

def grad_descent(f, x_0, y_0, eta = 0.01, N = 1000):
    dx = dy = 0.0001
    x, y = float(x_0), float(y_0)
    xs = [x]
    ys = [y]
    fs = [f(x, y)]

    for i in range(N):
        x_derivative = (f(x + dx, y) - f(x, y)) / dx
        y_derivative = (f(x, y + dy) - f(x, y)) / dy

        x_next = x - x_derivative * eta
        y_next = y - y_derivative * eta

        x, y = x_next, y_next

        xs.append(x)
        ys.append(y)
        fs.append(f(x,y))

    return np.array(xs), np.array(ys), np.array(fs)

x0 = float(input("Enter the starting point x0: "))
y0 = float(input("Enter the starting point y0: "))

xs, ys, fs = grad_descent(f, x0, y0, eta = 0.01, N = 1000)


plt.figure()
plt.plot(fs)
plt.title("f(x,y) for over iterations")
plt.xlabel("Iteration")
plt.ylabel("f(x,y)")
plt.grid(True)

pad = 1.0
x_min, x_max = xs.min() - pad, xs.max() + pad
y_min, y_max = ys.min() - pad, ys.max() + pad

xg = np.linspace(x_min, x_max, 200)
yg = np.linspace(y_min, y_max, 200)
X, Y = np.meshgrid(xg, yg)
Z = f(X, Y)

fig_3D = plt.figure()
ax = fig_3D.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, alpha = 0.5)
ax.plot(xs, ys, fs, linewidth = 2)
ax.set_title("Surface + path")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("f")

plt.show()