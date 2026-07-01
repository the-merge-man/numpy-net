import os
import sys
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_cifar10
from src.network import train_framework

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_cifar10()
    
    learning_rates = [0.005, 0.01, 0.05, 0.1, 0.25]
    os.makedirs("plots", exist_ok=True)
    
    plt.figure(figsize=(8, 5))
    for lr in learning_rates:
        print(f"Processing CIFAR-10 Sweep: alpha = {lr}")
        # Run across 3,072 input features
        history = train_framework(X_train, y_train, X_test, y_test, 
                                  input_dim=3072, hidden_dim=128, 
                                  epochs=15, alpha=lr, batch_size=128)
        plt.plot(range(1, 16), [a * 100 for a in history], marker='s', label=f"α = {lr}")
        
    plt.title("CIFAR-10 Optimization Spectrum (Dense Spatial Structural Limits)")
    plt.xlabel("Epoch")
    plt.ylabel("Test Set Accuracy (%)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/cifar_sweep_metrics.png")
    plt.show()
