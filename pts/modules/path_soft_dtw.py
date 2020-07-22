import numpy as np
import torch
from torch.autograd import Function
from numba import jit


@jit(nopython=True)
def my_max(x, gamma):
    # use the log-sum-exp trick
    max_x = np.max(x)
    exp_x = np.exp((x - max_x) / gamma)
    Z = np.sum(exp_x)
    return gamma * np.log(Z) + max_x, exp_x / Z


@jit(nopython=True)
def my_min(x, gamma):
    min_x, argmax_x = my_max(-x, gamma)
    return -min_x, argmax_x


@jit(nopython=True)
def my_max_hessian_product(p, z, gamma):
    return (p * z - p * np.sum(p * z)) / gamma


@jit(nopython=True)
def my_min_hessian_product(p, z, gamma):
    return -my_max_hessian_product(p, z, gamma)


@jit(nopython=True)
def dtw_grad(theta, gamma):
    m = theta.shape[0]
    n = theta.shape[1]
    V = np.zeros((m + 1, n + 1))
    V[:, 0] = 1e10
    V[0, :] = 1e10
    V[0, 0] = 0

    Q = np.zeros((m + 2, n + 2, 3))

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            # theta is indexed starting from 0.
            v, Q[i, j] = my_min(
                np.array([V[i, j - 1], V[i - 1, j - 1], V[i - 1, j]]), gamma
            )
            V[i, j] = theta[i - 1, j - 1] + v

    E = np.zeros((m + 2, n + 2))
    E[m + 1, :] = 0
    E[:, n + 1] = 0
    E[m + 1, n + 1] = 1
    Q[m + 1, n + 1] = 1

    for i in range(m, 0, -1):
        for j in range(n, 0, -1):
            E[i, j] = (
                Q[i, j + 1, 0] * E[i, j + 1]
                + Q[i + 1, j + 1, 1] * E[i + 1, j + 1]
                + Q[i + 1, j, 2] * E[i + 1, j]
            )

    return V[m, n], E[1 : m + 1, 1 : n + 1], Q, E


@jit(nopython=True)
def dtw_hessian_prod(theta, Z, Q, E, gamma):
    m = Z.shape[0]
    n = Z.shape[1]

    V_dot = np.zeros((m + 1, n + 1))
    V_dot[0, 0] = 0

    Q_dot = np.zeros((m + 2, n + 2, 3))
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            # theta is indexed starting from 0.
            V_dot[i, j] = (
                Z[i - 1, j - 1]
                + Q[i, j, 0] * V_dot[i, j - 1]
                + Q[i, j, 1] * V_dot[i - 1, j - 1]
                + Q[i, j, 2] * V_dot[i - 1, j]
            )

            v = np.array([V_dot[i, j - 1], V_dot[i - 1, j - 1], V_dot[i - 1, j]])
            Q_dot[i, j] = my_min_hessian_product(Q[i, j], v, gamma)
    E_dot = np.zeros((m + 2, n + 2))

    for j in range(n, 0, -1):
        for i in range(m, 0, -1):
            E_dot[i, j] = (
                Q_dot[i, j + 1, 0] * E[i, j + 1]
                + Q[i, j + 1, 0] * E_dot[i, j + 1]
                + Q_dot[i + 1, j + 1, 1] * E[i + 1, j + 1]
                + Q[i + 1, j + 1, 1] * E_dot[i + 1, j + 1]
                + Q_dot[i + 1, j, 2] * E[i + 1, j]
                + Q[i + 1, j, 2] * E_dot[i + 1, j]
            )

    return V_dot[m, n], E_dot[1 : m + 1, 1 : n + 1]


class PathDTWBatch(Function):
    @staticmethod
    def forward(ctx, D, gamma):  # D.shape: [batch_size, N , N]
        batch_size, N, N = D.shape
        device = D.device
        D_cpu = D.detach().cpu().numpy()
        gamma_gpu = torch.FloatTensor([gamma]).to(device)

        grad_gpu = torch.zeros((batch_size, N, N)).to(device)
        Q_gpu = torch.zeros((batch_size, N + 2, N + 2, 3)).to(device)
        E_gpu = torch.zeros((batch_size, N + 2, N + 2)).to(device)

        for k in range(0, batch_size):  # loop over all D in the batch
            _, grad_cpu_k, Q_cpu_k, E_cpu_k = dtw_grad(D_cpu[k, :, :], gamma)
            grad_gpu[k, :, :] = torch.FloatTensor(grad_cpu_k).to(device)
            Q_gpu[k, :, :, :] = torch.FloatTensor(Q_cpu_k).to(device)
            E_gpu[k, :, :] = torch.FloatTensor(E_cpu_k).to(device)
        ctx.save_for_backward(grad_gpu, D, Q_gpu, E_gpu, gamma_gpu)
        return torch.mean(grad_gpu, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        device = grad_output.device
        grad_gpu, D_gpu, Q_gpu, E_gpu, gamma = ctx.saved_tensors
        D_cpu = D_gpu.detach().cpu().numpy()
        Q_cpu = Q_gpu.detach().cpu().numpy()
        E_cpu = E_gpu.detach().cpu().numpy()
        gamma = gamma.detach().cpu().numpy()[0]
        Z = grad_output.detach().cpu().numpy()

        batch_size, N, N = D_cpu.shape
        Hessian = torch.zeros((batch_size, N, N)).to(device)
        for k in range(0, batch_size):
            _, hess_k = dtw_hessian_prod(
                D_cpu[k, :, :], Z, Q_cpu[k, :, :, :], E_cpu[k, :, :], gamma
            )
            Hessian[k : k + 1, :, :] = torch.FloatTensor(hess_k).to(device)

        return Hessian, None

