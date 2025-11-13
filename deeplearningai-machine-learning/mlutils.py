"""
mlutils.py
  Utility functions for the Machine Learning course
"""

import math
from copy import deepcopy


def compute_cost_v1(x, y, w, b):
    """
    Computes the 1/2 mean squared error for a given set (x, y) with given w and b params.
    Args:
      x ndarray(m,): feature vector.
      y dnarray(m,): target vector.
      w,b scalar: model weights
    Returns:
      The (scalar) cost value in the set while using w and b as parameters.
    """
    # number of training samples
    m = len(x)
    cost = 0

    for i in range(m):
        # predicted value
        f_wb = w * x[i] + b
        this_cost = (f_wb - y[i])**2
        cost += this_cost
    return (1/(2*m)) * cost

def compute_gradient_v1(x ,y, w, b):
    """
    Computes the gradient (the partial derivatives of the cost function with respect to both w and b) given a set (x, y) and a pair of weights (w, b).
    Args:
      x ndarray(m,): feature vector.
      y dnarray(m,): target vector.
      w,b scalar: model weights
    Returns:
     dj_dw: The gradient of the cost with respect to parameter w.
     dj_db: The gradient of the cost with respect to parameter b.
    """

    # the length of the training set
    m = len(x)
    dj_dw = 0
    dj_db = 0

    for i in range(m):
        f_wb = w * x[i] + b

        # The derivative of the cost at point x[i] with respect to w
        dj_dw_i = (f_wb - y[i]) * x[i]

        # The derivative of the cost at point x[i] with respect to b
        dj_db_i = f_wb - y[i]

        dj_dw += dj_dw_i
        dj_db += dj_db_i

    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_dw, dj_db


def gradient_descend_v1(x, y, initial_w, initial_b, a, n_iter, cost_f=compute_cost_v1, gradient_f=compute_gradient_v1):
    """
    Performs gradient descent to fit w and b with n_iter iterations and an alpha value of a.
    Args:
      x ndarray(m,):                          feature vector.
      y dnarray(m,):                          target vector.
      initial_w,initial_b scalar:             initial model weights.
      a float:                                alpha value.
      cost_f function:                        callback to compute the cost.
      gradient_f function:                    callback to compute the gradient.
    Returns:
      w,b (scalar):                           Updated parameters after running gradient descent.
      j_hist list(scalar):                    History of costs (for graphing purposes).
      p_hist list((scalar, scalar)):          History of parameters tested (for graphing purposes).
    """

    # Avoiding pass-by-reference creating a deep copy so that initial_* are not altered
    w, b = deepcopy(initial_w), deepcopy(initial_b)
    print(f'Initiating gradient descent with initial values w={w:0.3f}, b={b:0.3f}')
    # w = deepcopy(initial_w)
    j_hist = []
    p_hist = []
    # w, b = initial_w, initial_b

    for i in range(n_iter):
        # Caluclate gradients
        dj_dw, dj_db = gradient_f(x, y, w, b)

        # Update the parameters
        b = b - a * dj_db
        w = w - a * dj_dw

        # Save the cost in history (only for the first 100.000 iterations)
        if i < 100000:
            j_hist.append(cost_f(x, y, w, b))
            p_hist.append((w, b))

        # Print some logs every 1/10 of the total iterations
        if i % math.ceil(n_iter / 10)  == 0:
            print(f'== Iteration {i} ==')
            print(f'Last cost computed: {j_hist[-1]: 0.2e}')
            print(f'dj_dw gradient: {dj_dw: 0.3e}')
            print(f'dj_db gradient: {dj_db: 0.3e}')
            print(f'w: {w}')
            print(f'b: {b}')
            print('====================')
    
    return w, b ,j_hist, p_hist