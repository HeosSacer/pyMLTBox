import numpy as np
import matplotlib.pyplot as plt
import src.linear_regression as lr
import sys

def linear_regression_test(plot=False):
    try:
        # Generate the Data:
        n = 200
        w_truth = 0.7
        noise_std = 2.5

        print("True Param. w=%s, Noise_Std=%s" % (w_truth, noise_std))

        X = np.linspace(-5, 5, n)
        Y = w_truth * X + np.random.normal(0, noise_std, n)

        # Linear Regression Analytical
        w_analytical = np.mean(Y/X)  # inv(X * X') * X * Y'
        Y_analytical = np.multiply(np.array([w_analytical]), X)
        J_analytical = np.linalg.norm(Y_analytical-Y, 2)**2

        print("Analytical Solution w=%s, Cost:%s" % (w_analytical, J_analytical))

        # Linear Regression Gradient
        (Y_grad, w) = lr.gradient(X, Y)
        print("Gradient Solution w=%s" % (w))

        if plot:
            plt.scatter(X, Y)
            plt.plot(X, Y_grad)
            plt.show()
    except:
        print("FAIL: Linear Regression")

if __name__ == "__main__":
    linear_regression_test()
