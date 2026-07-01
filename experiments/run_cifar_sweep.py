import os
import pickle
import tarfile
import urllib.request
import numpy as np
import matplotlib.pyplot as plt

# --- INTERNAL DATA LOADER ---
def download_file(url, filename):
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, filename)

def load_cifar10_direct():
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    tar_filename = "cifar-10-python.tar.gz"
    download_file(url, tar_filename)
    
    extracted_dir = "cifar-10-batches-py"
    if not os.path.exists(extracted_dir):
        print("Extracting CIFAR-10 data batches...")
        with tarfile.open(tar_filename, "r:gz") as tar:
            tar.extractall()
            
    X_train_list, y_train_list = [], []
    for i in range(1, 6):
        with open(f"{extracted_dir}/data_batch_{i}", 'rb') as fo:
            batch = pickle.load(fo, encoding='bytes')
            X_train_list.append(batch[b'data'])
            y_train_list.append(batch[b'labels'])
            
    X_train = np.vstack(X_train_list).T / 255.0
    y_train = np.concatenate(y_train_list)
    
    with open(f"{extracted_dir}/test_batch", 'rb') as fo:
        batch = pickle.load(fo, encoding='bytes')
        X_test = batch[b'data'].T / 255.0
        y_test = np.array(batch[b'labels'])
        
    return X_train, y_train, X_test, y_test

# --- MATH ARCHITECTURE ---
def init_params(input_size, hidden_size):
    W1 = np.random.randn(hidden_size, input_size) * np.sqrt(2.0 / input_size)
    b1 = np.zeros((hidden_size, 1))
    W2 = np.random.randn(10, hidden_size) * np.sqrt(2.0 / hidden_size)
    b2 = np.zeros((10, 1))
    return W1, b1, W2, b2

def train_framework(X, y, X_val, y_val, input_dim, hidden_dim, epochs, alpha, batch_size):
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
            
            # Forward
            Z1 = np.dot(W1, X_batch) + b1
            A1 = np.maximum(0, Z1)
            Z2 = np.dot(W2, A1) + b2
            exp_Z2 = np.exp(Z2 - np.max(Z2, axis=0, keepdims=True))
            A2 = exp_Z2 / np.sum(exp_Z2, axis=0, keepdims=True)
            
            # Backward
            one_hot_y = np.zeros((10, y_batch.size))
            one_hot_y[y_batch, np.arange(y_batch.size)] = 1
            
            dZ2 = A2 - one_hot_y
            dW2 = (1 / m_batch) * np.dot(dZ2, A1.T)
            db2 = (1 / m_batch) * np.sum(dZ2, axis=1, keepdims=True)
            
            dZ1 = np.dot(W2.T, dZ2) * (Z1 > 0)
            dW1 = (1 / m_batch) * np.dot(dZ1, X_batch.T)
            db1 = (1 / m_batch) * np.sum(dZ1, axis=1, keepdims=True)
            
            # Update
            W1 -= alpha * dW1; b1 -= alpha * db1
            W2 -= alpha * dW2; b2 -= alpha * db2
            
        # Eval Pass
        Z1_v = np.dot(W1, X_val) + b1
        A1_v = np.maximum(0, Z1_v)
        Z2_v = np.dot(W2, A1_v) + b2
        preds = np.argmax(Z2_v, axis=0)
        acc = np.sum(preds == y_val) / y_val.size
        history.append(acc)
        
    return history

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_cifar10_direct()
    learning_rates = [0.005, 0.01, 0.05, 0.1, 0.25]
    os.makedirs("plots", exist_ok=True)
    
    plt.figure(figsize=(8, 5))
    for lr in learning_rates:
        print(f"Running CIFAR-10 Sweep: alpha = {lr}")
        history = train_framework(X_train, y_train, X_test, y_test, 3072, 128, 15, lr, 128)
        plt.plot(range(1, 16), [a * 100 for a in history], marker='s', label=f"α = {lr}")
        
    plt.title("CIFAR-10 Convergence Sweep")
    plt.xlabel("Epoch")
    plt.ylabel("Test Set Accuracy (%)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/cifar_sweep_metrics.png")
    print("Saved plot to plots/cifar_sweep_metrics.png")
    plt.show()
