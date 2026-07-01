import os
import sys
import matplotlib.pyplot as plt

# Ensure root paths are accessible by the Python runtime
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_mnist
from src.network import train_framework

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_mnist()
    
    learning_rates = [0.005, 0.01, 0.05, 0.1, 0.25]
    os.makedirs("plots", exist_ok=True)
    
    plt.figure(figsize=(8, 5))
    for lr in learning_rates:
        print(f"Processing MNIST Sweep: alpha = {lr}")
        history = train_framework(X_train, y_train, X_test, y_test, 
                                  input_dim=784, hidden_dim=128, 
                                  epochs=15, alpha=lr, batch_size=128)
        plt.plot(range(1, 16), [a * 100 for a in history], marker='o', label=f"α = {lr}")
        
    plt.title("MNIST Convergence Sweep (Dense Multi-Layer Support)")
    plt.xlabel("Epoch")
    plt.ylabel("Test Set Accuracy (%)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/mnist_sweep_metrics.png")
    plt.show()
