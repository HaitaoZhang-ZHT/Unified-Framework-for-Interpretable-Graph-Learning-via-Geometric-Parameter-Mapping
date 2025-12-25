import time  # 确保你已经导入了 time 模块

# 在代码开始时记录起始时间
start_time = time.time()

import numpy as np
import torch
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from eigenpro_rfm import hyperparam_train, train, test

def load_cora_data(content_file, cites_file):
    with open(content_file, 'r') as f:
        content = f.readlines()
    
    with open(cites_file, 'r') as f:
        cites = f.readlines()
    
    paper_ids = []
    features = []
    labels = []

    for line in content:
        split_line = line.strip().split()
        paper_id = split_line[0]
        feature = list(map(int, split_line[1:-1]))
        label = split_line[-1]
        
        paper_ids.append(paper_id)
        features.append(feature)
        labels.append(label)
    
    paper_ids = np.array(paper_ids)
    features = np.array(features, dtype=np.float32)  # 将features转换为float32类型
    labels = np.array(labels)
    
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)
    
    return paper_ids, features, labels, cites

def preprocess_data(features, labels, val_size=0.2, test_size=0.2, random_state=42):
    # 首先分割出测试集
    X_train_val, X_test, y_train_val, y_test = train_test_split(features, labels, test_size=test_size, random_state=random_state)
    
    # 然后从训练验证集中分割出验证集
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=val_size/(1-test_size), random_state=random_state)
    
    # 将标签转换为one-hot编码
    y_train = torch.nn.functional.one_hot(torch.tensor(y_train), num_classes=7).float().numpy()
    y_val = torch.nn.functional.one_hot(torch.tensor(y_val), num_classes=7).float().numpy()
    y_test = torch.nn.functional.one_hot(torch.tensor(y_test), num_classes=7).float().numpy()
    
    return X_train, X_val, X_test, y_train, y_val, y_test

# Load and preprocess the data
content_file = 'cora.content'
cites_file = 'cora.cites'
paper_ids, features, labels, cites = load_cora_data(content_file, cites_file)
X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(features, labels)

# Normalize the features
mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)
std[std == 0] = 1  # 将标准差为零的特征设置为1
X_train = (X_train - mean) / std
X_val = (X_val - mean) / std
X_test = (X_test - mean) / std

# Set parameters
iters = 5
ep_iter = 10
L = 6
depth = 3
normalize = True

# Hyperparameter training with detailed debug info
best_accuracy, iters_per_layer, best_M, ep_iters_per_layer, M_layers, best_M_layers, best_model_state_layers = hyperparam_train(
    X_train, y_train, X_val, y_val, iters=iters, ep_iter=ep_iter, L=L, normalize=normalize, depth=depth, save_dir='./hyperparam_train/'
)

print("Best Accuracy after Hyperparam Training: ", best_accuracy)
print("Iterations per Layer: ", iters_per_layer)
print("Epoch Iterations per Layer: ", ep_iters_per_layer)

# Final training with detailed debug info
sol, best_accuracy, _, accuracy, y_preds, M_layers, best_M_layers, best_model_state_layers = train(
    X_train, y_train, X_test, y_test, best_M, iters_per_layer=iters_per_layer, ep_iters_per_layer=ep_iters_per_layer, L=L, normalize=normalize, depth=depth, save_dir='./Final training/'
)

print("Best Accuracy after Final Training: ", best_accuracy)

# Test the model using all layers' transformation matrices
X_test_transformed = test(X_test, M_layers, best_model_state_layers)

# Convert predictions to class labels
y_preds_class = np.argmax(y_preds, axis=1)
y_test_class = np.argmax(y_test, axis=1)

# Calculate accuracy
accuracy = accuracy_score(y_test_class, y_preds_class)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# 在代码结束时记录结束时间
end_time = time.time()

# 计算并输出总用时
total_time = end_time - start_time
print(f"Total runtime: {total_time:.2f} seconds")