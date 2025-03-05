# losses.py
import numpy as np

def cross_entropy(y_pred, y_true, eps=1e-8):
    """
    Computes cross-entropy loss:
      L = -sum( y_true * log(y_pred) ) / batch_size
    """
    m = y_true.shape[0]
    # Clip y_pred to avoid log(0)
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.sum(y_true * np.log(y_pred)) / m

def cross_entropy_deriv(y_pred, y_true):
    """
    Derivative of cross-entropy w.r.t. y_pred:
      dL/dy_pred = (y_pred - y_true)
    """
    return y_pred - y_true

def mean_squared_error(y_pred, y_true):
    """
    Computes MSE loss:
      L = sum( (y_pred - y_true)^2 ) / (batch_size)
    """
    m = y_true.shape[0]
    return np.sum((y_pred - y_true)**2) / m

def mean_squared_error_deriv(y_pred, y_true):
    """
    Derivative of MSE w.r.t. y_pred:
      dL/dy_pred = (y_pred - y_true)
    """
    m = y_true.shape[0]
    return 2 * (y_pred - y_true)

loss_functions = {
    "cross_entropy": (cross_entropy, cross_entropy_deriv),
    "mean_squared_error": (mean_squared_error, mean_squared_error_deriv)
}
