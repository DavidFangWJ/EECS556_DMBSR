import numpy as np
from scipy.sparse import csr_matrix, kron, eye
from scipy.sparse.linalg import spsolve
from scipy.sparse import vstack
import matplotlib.pyplot as plt


def buildC1(N):
    i = np.arange(N - 1)
    j = np.arange(N - 1)
    i = np.concatenate([i, i])
    j = np.concatenate([j, j + 1])
    s = np.concatenate([np.ones(N - 1), -np.ones(N - 1)])
    C1 = csr_matrix((s, (i, j)), shape=(N - 1, N))
    return C1


def buildC(nx, ny):
    Cx = buildC1(nx)
    Cy = buildC1(ny)
    Ix = eye(ny)
    Iy = eye(nx)
    C = kron(Ix, Cx)
    C = vstack([C, kron(Cy, Iy)])
    return C


def wt(t, delta):
    return 1 / (1 + np.abs(t) / delta)


def npls_sps(yy, niter=80, beta=16, delta=0.5):
    nx, ny = yy.shape
    C = buildC(nx, ny)
    denom = 1 + beta * np.abs(C.T) @ (np.abs(C) @ np.ones(nx * ny))
    xx = yy.T.reshape(-1, 1)[:, 0]  # initial guess: the noisy image - in a vector
    for i in range(niter):
        Cx = C @ xx
        grad = xx - yy.T.reshape(-1, 1)[:, 0] + beta * (C.T @ (wt(Cx, delta) * Cx))
        xx = xx - grad / denom
    return (xx.T).reshape((yy.T).shape).T


def npls(images):
    device = images.device
    images = images.cpu().numpy()
    batch_size, channels, height, width = images.shape
    processed_images = np.zeros_like(images)

    for i in range(batch_size):
        for c in range(channels):
            processed_images[i, c] = npls_sps(images[i, c])
    
    return torch.from_numpy(processed_images).to(device)