"""
mlutils.py
  Utility functions for the Machine Learning course
"""

import math
from copy import deepcopy

import numpy as np

from lab_utils_multi import run_gradient_descent, plot_cost_i_w

# Single variable gradient descent utilities

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



# Multi-variable gradient descent utilities

def compute_cost_multi_v1(X, y, w, b):
    """
    compute the cost cost value
    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters  
      b (scalar)       : model parameter
      
    Returns:
      cost (scalar): cost
    """

    m = X.shape[0]

    cost = 0.0

    for i in range(m):
        # It's a DOT product this time; this is the value of the prediction
        f_wb = np.dot(X[i], w) + b # dot product vector by scalar = scalar
        cost += (f_wb - y[i])**2
    return cost / (2 * m)



def compute_gradient_multi_v1(X, y, w, b):
    """
    Computes the gradient for linear regression 
    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters  
      b (scalar)       : model parameter
      
    Returns:
      dj_dw (ndarray (n,)): The gradient of the cost w.r.t. the parameters w. 
      dj_db (scalar):       The gradient of the cost w.r.t. the parameter b. 
    """

    m, n = X.shape # it's now 2D: rows are examples, columns are the features
    dj_dw = np.zeros((n,)) # it's now a vector of gradients, one for each w (there's one w for each feature)
    dj_db = 0

    for i in range(m):
        # f_wb is now a dot product between vector X[i] and vector w (one w value for each feature) = still a scalar
        f_wb = np.dot(X[i], w) + b
        
        err = f_wb - y[i]
        for j in range(n):
            # compute the gradient (partial derivative) for each
            dj_dw[j] += err * X[i, j] # ith row, jth column (feature)
        dj_db += err

    dj_dw /= m
    dj_db /= m

    return dj_dw, dj_db



def gradient_descent_multi_v1(x, y, initial_w, initial_b, a, n_iter, cost_f=compute_cost_multi_v1, gradient_f=compute_gradient_multi_v1):
    """
    Performs gradient descent to fit w and b with n_iter iterations and an alpha value of a.
    Args:
      x ndarray(m, n):                        feature vector.
      y dnarray(m,):                          target vector.
      initial_w ndarray(n,):                  initial model w values
      initial_b scalar:                       initial model b.
      a float:                                alpha value.
      cost_f function:                        callback to compute the cost.
      gradient_f function:                    callback to compute the gradient.
    Returns:
      w (ndarray(n,)),b (scalar):             Updated parameters after running gradient descent.
      j_hist list(scalar):                    History of costs (for graphing purposes).
    """
    
    # Avoiding pass-by-reference creating a deep copy so that initial_* are not altered
    w, b = deepcopy(initial_w), deepcopy(initial_b)
    print(f'Initiating gradient descent with initial values w={w}, b={b:0.3f}')
    j_hist = []

    for i in range(n_iter):
        dj_dw, dj_db = gradient_f(x, y, w, b)
    
        w = w - a * dj_dw
        b = b - a * dj_db
    
        # Save the history of the cost for the first 100000 iterations
        if i < 100000:
            j_hist.append(cost_f(x, y, w, b))
    
        # Print log every 1/10 of total iterations
        if i % (math.ceil( n_iter / 10)) == 0:
            print(f"Iteration {i:4d}: Cost {j_hist[-1]:8.2f}   ")

    return w, b, j_hist


def plot_cost_with_alpha(X, y, alpha: float, n_iter: int) -> None:
    if alpha is None:
        raise ValueError('A value for alpha MUST be specified.')

    if n_iter is None:
        raise ValueError('A value for n_iter MUST be specified.')

    
    print(f'Current alpha value: {alpha}.')
    _, _, hist = run_gradient_descent(
        X,
        y,
        n_iter,
        alpha=alpha
    )

    plot_cost_i_w(X, y, hist)



def z_normalize(x: np.ndarray):
    """
    z-zscore normalizes x

    Args:
      x ndarray(n, m):               feature matrix
    Returns:
      x_norm (ndarray(m, n)):        z-score normalized x
      mu:    (ndarray(n,):           means vector (1 for each feature)
      sigma  (ndarray(m, n)):        std deviations vector (1 for each feature)
    """

    mu = np.mean(x, axis=0)
    sigma = np.std(x, axis=0)

    x_norm = (x - mu) / sigma

    return x_norm, mu, sigma
