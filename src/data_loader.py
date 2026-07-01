import os
import gzip
import pickle
import urllib.request
import tarfile
import numpy as np

def download_file(url, filename):
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, filename)

def load_mnist_raw():
    base_url = "https://raw.githubusercontent.com/fgnt/mnist/master/"
    files = ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz", 
             "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"]
    
    for f in files:
        download_file(base_url + f, f)
        
    with gzip.open(files[0], 'rb') as f:
        X_train = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28*28).T / 255.0
    with gzip.open(files[1], 'rb') as f:
        y_train = np.frombuffer(f.read(), np.uint8, offset=8)
    with gzip.open(files[2], 'rb') as f:
        X_test = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28*28).T / 255.0
    with gzip.open(files[3], 'rb') as f:
        y_test = np.frombuffer(f.read(), np.uint8, offset=8)
        
    return X_train, y_train, X_test, y_test

def load_cifar10_raw():
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    tar_filename = "cifar-10-python.tar.gz"
    download_file(url, tar_filename)
    
    extracted_dir = "cifar-10-batches-py"
    if not os.path.exists(extracted_dir):
        with tarfile.open(tar_filename, "r:gz") as tar:
            tar.extractall()
            
    # Combine the 5 separate training batches provided by CIFAR
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
