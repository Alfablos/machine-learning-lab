"""
mlutils.py
  utility functions for the course
"""

import numpy as np

def sigmoid(z):
    """
    Returns the sigmoid function values
    Arg:
      z: scalar o np.array
    """
    z = np.clip(z, -500, 500)
    return 1.0/(1.0 + np.exp(-z))