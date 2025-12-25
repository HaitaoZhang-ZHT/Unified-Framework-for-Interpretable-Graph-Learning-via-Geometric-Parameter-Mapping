import torch
import os
from torch.autograd import Variable
import torch.optim as optim
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import classic_kernel
from numpy.linalg import svd, solve, norm
import hickle
from tqdm import tqdm
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from sklearn.svm import SVC
import eigenpro


def laplace_kernel_M(pair1, pair2, bandwidth, M):
    return classic_kernel.laplacian_M(pair1, pair2, bandwidth, M)

def laplace_kernel(pair1, pair2, bandwidth):
    return classic_kernel.laplacian(pair1, pair2, bandwidth)

def kernel(pair1, pair2, nngp=False):
    out = pair1 @ pair2.transpose(1, 0)
    N1 = torch.sum(torch.pow(pair1, 2), dim=-1).view(-1, 1)
    N2 = torch.sum(torch.pow(pair2, 2), dim=-1).view(-1, 1)

    XX = torch.sqrt(N1 @ N2.transpose(1, 0))
    out = out / XX

    out = torch.clamp(out, -1, 1)

    first = 1/np.pi * (out * (np.pi - torch.acos(out)) \
                       + torch.sqrt(1. - torch.pow(out, 2))) * XX
    if nngp:
        out = first
    else:
        sec = 1/np.pi * out * (np.pi - torch.acos(out)) * XX
        out = first + sec

    return out


def get_grads(X, sol, L, P):
    M = 0.

    start = time.time()
    num_samples = 20000
    indices = np.random.randint(len(X), size=num_samples)

    if len(X) > len(indices):
        x = X[indices, :]
    else:
        x = X

    K = laplace_kernel_M(X, x, L, P)

    dist = classic_kernel.euclidean_distances_M(X, x, P, squared=False)
    dist = torch.where(dist < 1e-10, torch.zeros(1).float(), dist)

    K = K / dist
    K[K == float("Inf")] = 0.

    a1 = torch.from_numpy(sol.T).float()
    n, d = X.shape
    n, c = a1.shape
    m, d = x.shape

    a1 = a1.reshape(n, c, 1)
    X1 = (X @ P).reshape(n, 1, d)
    step1 = a1 @ X1
    del a1, X1
    step1 = step1.reshape(-1, c*d)

    step2 = K.T @ step1
    del step1

    step2 = step2.reshape(-1, c, d)

    a2 = torch.from_numpy(sol).float()
    step3 = (a2 @ K).T

    del K, a2

    step3 = step3.reshape(m, c, 1)
    x1 = (x @ P).reshape(m, 1, d)
    step3 = step3 @ x1

    G = (step2 - step3) * -1 / L

    M = 0.

    bs = 50
    batches = torch.split(G, bs)
    for i in tqdm(range(len(batches))):
        grad = batches[i].cuda()
        gradT = torch.transpose(grad, 1, 2)
        M += torch.sum(gradT @ grad, dim=0).cpu()
        del grad, gradT
    torch.cuda.empty_cache()
    M /= len(G)
    M = M.numpy()

    return M


def eigenpro_solve(X_train, y_train, X_test, y_test, L, steps, M):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    M = M.cuda()
    kernel_fn = lambda x, y: laplace_kernel_M(x, y, L, M)
    model = eigenpro.FKR_EigenPro(kernel_fn, X_train,  y_dim=y_train.shape[1], device=device)

    res, y_preds = model.fit(X_train, y_train, X_test, y_test,
                             epochs=list(range(steps)), mem_gb=12,
                             n_subsamples=1600)
    best_accuracy = 0
    best_ep_iter = 0
    best_model_state = None  # 添加一个变量来记录最佳模型状态

    for idx, r in enumerate(res):
        accuracy = res[r][1]['accuracy']  # 使用验证集的准确率
        if accuracy > best_accuracy:
            best_ep_iter = idx
            best_accuracy = accuracy
            best_model_state = model.state_dict()  # 保存最佳模型状态

    return model.weight.cpu().numpy(), best_accuracy, best_ep_iter, accuracy, y_preds, best_model_state


def hyperparam_train(X_train, y_train, X_test, y_test, iters=5, ep_iter=10, L=10, normalize=False, depth=3, save_dir='./'):
    if normalize:
        X_train /= norm(X_train, axis=-1).reshape(-1, 1)
        X_test /= norm(X_test, axis=-1).reshape(-1, 1)

    X_train = X_train.astype('float32')
    y_train = y_train.astype('long')  # Change to long for classification
    X_test = X_test.astype('float32')
    y_test = y_test.astype('long')  # Change to long for classification

    best_accuracy = 0.
    best_iter = 0.
    best_ep_iter = 0

    n, d = X_train.shape
    M = np.eye(d, dtype='float32')
    M_layers = []
    best_M_layers = []
    best_model_state_layers = []  # 保存每一层的最佳模型状态

    best_iters_per_layer = []
    best_ep_iters_per_layer = []

    for layer in range(depth):
        print(f"Hyperparam_train {layer + 1}/{depth}")
        best_layer_accuracy = 0
        best_layer_M = M.copy()
        best_layer_model_state = None

        for i in range(iters):
            sol, old_test_accuracy, s_ep_iter, _, _, best_model_state = eigenpro_solve(X_train, y_train, X_test, y_test, L, ep_iter, torch.from_numpy(M))

            sol = sol.T
            if old_test_accuracy >= best_layer_accuracy:
                best_iter = i
                best_layer_accuracy = old_test_accuracy
                best_layer_M = M.copy()
                best_ep_iter = s_ep_iter
                best_layer_model_state = best_model_state

            M = get_grads(torch.from_numpy(X_train).float(), sol, L, torch.from_numpy(M))

        # 保存每层的最佳迭代次数和模型状态
        best_iters_per_layer.append(best_iter)
        best_ep_iters_per_layer.append(best_ep_iter)
        best_model_state_layers.append(best_layer_model_state)

        # Apply the learned transformation to X_train for the next layer
        X_train = X_train @ M
        X_train = np.tanh(X_train)  # Nonlinear transformation

        # 保存每一层的M矩阵和最佳M矩阵
        M_layers.append(M.copy())
        best_M_layers.append(best_layer_M)
        np.save(os.path.join(save_dir, f'M_layer_{layer+1}.npy'), M)
        np.save(os.path.join(save_dir, f'best_M_layer_{layer+1}.npy'), best_layer_M)

        print(f"Layer {layer + 1}, Best Iter: {best_iter}, Best Accuracy: {best_layer_accuracy}")

    # 保存最好的M矩阵
    np.save(os.path.join(save_dir, 'best_M.npy'), best_M_layers[-1])
    np.save(os.path.join(save_dir, 'all_layers_M.npy'), np.array(M_layers))
    np.save(os.path.join(save_dir, 'best_layers_M.npy'), np.array(best_M_layers))

    return best_layer_accuracy, best_iters_per_layer, best_M_layers[-1], best_ep_iters_per_layer, M_layers, best_M_layers, best_model_state_layers


def train(X_train, y_train, X_test, y_test, M, iters_per_layer, ep_iters_per_layer, L=10, normalize=False, depth=3, save_dir='./'):
    if normalize:
        X_train /= norm(X_train, axis=-1).reshape(-1, 1)
        X_test /= norm(X_test, axis=-1).reshape(-1, 1)

    X_train = X_train.astype('float32')
    y_train = y_train.astype('float32')
    X_test = X_test.astype('float32')
    y_test = y_test.astype('float32')

    M_layers = []
    best_M_layers = []
    best_model_state_layers = []  # 保存每一层的最佳模型状态

    best_overall_accuracy = 0.0  # 添加一个变量来记录训练过程中最好的准确率

    for layer in range(depth):
        print(f"Training layer {layer + 1}/{depth}")
        best_layer_accuracy = 0
        best_layer_M = M.copy()
        best_layer_model_state = None

        for i in range(iters_per_layer[layer]+1):
            sol, layer_best_accuracy, _, accuracy, y_preds, best_model_state = eigenpro_solve(X_train, y_train, X_test, y_test, L, ep_iters_per_layer[layer]+1, torch.from_numpy(M))
            sol = sol.T
            if layer_best_accuracy >= best_layer_accuracy:
                best_layer_accuracy = layer_best_accuracy
                best_layer_M = M.copy()
                best_layer_model_state = best_model_state

            M = get_grads(torch.from_numpy(X_train).float(), sol, L, torch.from_numpy(M))

        # 更新整个训练过程中的最佳准确率
        if best_layer_accuracy > best_overall_accuracy:
            best_overall_accuracy = best_layer_accuracy

        # Apply the learned transformation to X_train for the next layer
        X_train = X_train @ M
        X_train = np.tanh(X_train)  # Nonlinear transformation

        # 保存每一层的M矩阵和最佳M矩阵
        M_layers.append(M.copy())
        best_M_layers.append(best_layer_M)
        best_model_state_layers.append(best_layer_model_state)
        np.save(os.path.join(save_dir, f'M_layer_{layer+1}.npy'), M)
        np.save(os.path.join(save_dir, f'best_M_layer_{layer+1}.npy'), best_layer_M)

        print(f"Layer {layer + 1}, Best Accuracy: {best_layer_accuracy}")

    # 保存最终的M矩阵
    np.save(os.path.join(save_dir, 'final_M.npy'), M)
    np.save(os.path.join(save_dir, 'all_layers_M.npy'), np.array(M_layers))
    np.save(os.path.join(save_dir, 'best_layers_M.npy'), np.array(best_M_layers))

    return sol, best_overall_accuracy, _, accuracy, y_preds, M_layers, best_M_layers, best_model_state_layers


def test(X_test, M_layers, best_model_state_layers):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    def laplace_kernel_M(pair1, pair2, bandwidth, M):
        return classic_kernel.laplacian_M(pair1, pair2, bandwidth, M)

    L = 10  # 这个值需要和训练时保持一致
    kernel_fn = lambda x, y: laplace_kernel_M(x, y, L, torch.eye(x.shape[1]))

    # Apply the learned transformations to X_test for each layer
    for M, best_model_state in zip(M_layers, best_model_state_layers):
        X_test = X_test @ M
        X_test = np.tanh(X_test)  # Nonlinear transformation
        model = eigenpro.FKR_EigenPro(kernel_fn, X_test, y_dim=7, device=device)  # 恢复模型
        model.load_state_dict(best_model_state)  # 加载最佳模型状态
    return X_test

