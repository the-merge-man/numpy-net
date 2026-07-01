import numpy as np

def relu(Z):
    """Computes the Rectified Linear Unit activation function."""
    return np.maximum(0, Z)

def relu_derivative(Z):
    """Computes the derivative of the ReLU function for backpropagation."""
    return Z > 0

def softmax(Z):
    """Computes stable softmax vector probabilities over columns."""
    exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

def one_hot(y, classes=10):
    """Converts a flat index label vector to a binary one-hot matrix."""
    one_hot_y = np.zeros((classes, y.size))
    one_hot_y[y, np.arange(y.size)] = 1
    return one_hot_y
