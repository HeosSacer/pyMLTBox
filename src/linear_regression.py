import numpy as np


def gradient(X, Y):

    # Gradient Function
    grad = lambda w, X, Y: np.dot(X, X)*w - np.dot(X, Y)

    # Cost Function
    cost = lambda w, X, Y: np.linalg.norm(Y - w * X)**2

    # Initialization of Parameters
    n_max = X.size
    w_gd = np.zeros(n_max)
    J_gd = np.zeros(n_max)
    g_gd = np.zeros(n_max)

    alpha = 8e-4
    w_gd[0] = -2.5

    for i in range(0, n_max-1):
        # Calc Cost
        J_gd[i] = cost(w_gd[i], X, Y)
        # Calc Gradient
        g_gd[i] = grad(w_gd[i], X, Y)
        # Update Parameter
        w_gd[i+1] = w_gd[i] - alpha * g_gd[i]

    J_gd[n_max-1] = cost(w_gd[n_max-1], X, Y)
    Y_out = X*w_gd[n_max-1]
    return(Y_out, w_gd[n_max-1])
