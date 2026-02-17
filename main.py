import numpy as np
import matplotlib.pyplot as plt



def f(x, y):
    return x**2 + y**2

def grad_descent(f, x_0, y_0, eta = 0.01, N = 1000):
    dx = dy = 0.0001
    x, y = x_0, y_0
    xs = [х]
    ys = [y]
    fs = [f(x, y)]

    for i in range(N):

        x_derivative = (f(x + dx, y) - f(x, y)) / dx
        y_derivative = (f(x, y + dy) - f(x, y)) / dy

        x_next = x - x_derivative * eta
        y_next = y - y_derivative * eta

        x = x_next
        y = y_next


        xs.append(x)
        ys.append(y)
        fs.append(f(x,y))

        xs = np.array(xs)
        ys = np.array(ys)
        fs = np.array(fs)

    return (xs, ys, fs)



x0 = float(input("Enter the starting point x0: "))
y0 = float(input("Enter the starting point y0: "))
xs, ys, fs = grad_descent(f, x0, y0, eta = 0.01, N = 1000)


plt.figure()
plt.plot(fs)
plt.title("f(x,y)")
plt.xlabel("Iteration")
plt.ylabel("f(x,y)")
plt.grid(True)



fig_3D = plt.figure()
ax = fig_3D.add_subplot(111, projection='3d')
ax.plot_surface(, Y, Z, alpha = 0.5)
ax.plot(xs, ys, fs, linewidth=2)
ax.set_title("Surface + path")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("f")

plt.show()