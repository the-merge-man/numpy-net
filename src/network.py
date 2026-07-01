import numpy as np
from src.activations import relu, relu_derivative, softmax, one_hot

def init_params(input_size, hidden_size, output_size=10):
    """Applies He (Kaiming) initialization to stabilize variance across deep boundaries."""
    W1 = np.random.randn(hidden_size, input_size) * np.sqrt(2.0 / input_size)
    b1 = np.zeros((hidden_size, 1))
    W2 = np.random.randn(output_size, hidden_size) * np.sqrt(2.0 / hidden_size)
    b2 = np.zeros((output_size, 1))
    return W1, b1, W2, b2

def forward_prop(W1, b1, W2, b2, X):
    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def backward_prop(Z1, A1, Z2, A2, W2, X, y, m):
    one_hot_y = one_hot(y)
    
    dZ2 = A2 - one_hot_y
    dW2 = (1 / m) * np.dot(dZ2, A1.T)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
    
    dZ1 = np.dot(W2.T, dZ2) * relu_derivative(Z1)
    dW1 = (1 / m) * np.dot(dZ1, X.T)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)
    
    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    return W1 - alpha * dW1, b1 - alpha * db1, W2 - alpha * dW2, b2 - alpha * db2

def train_framework(X, y, X_val, y_val, input_dim, hidden_dim, epochs, alpha, batch_size):
    """Executes mini-batch optimization loops and returns historical metrics."""
    W1, b1, W2, b2 = init_params(input_dim, hidden_dim)
    m = X.shape[1]
    history = []
    
    for epoch in range(1, epochs + 1):
        permutation = np.random.permutation(m)
        X_shuffled = X[:, permutation]
        y_shuffled = y[permutation]
        
        for i in range(0, m, batch_size):
            X_batch = X_shuffled[:, i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
            m_batch = X_batch.shape[1]
            
            Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X_batch)
            dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W2, X_batch, y_batch, m_batch)
            W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
            
        # Terminal accuracy pass per epoch
        _, _, _, A2_val = forward_prop(W1, b1, W2, b2, X_val)
        preds = np.argmax(A2_val, axis=0)
        acc = np.sum(preds == y_val) / y_val.size
        history.append(acc)
        
    return history
