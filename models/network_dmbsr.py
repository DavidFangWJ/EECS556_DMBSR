import torch
import torch.nn as nn
import models.basicblock as B
import numpy as np
from utils import utils_image as util
from math import sqrt
import os
import subprocess
import matplotlib.pyplot as plt
import torch.nn.functional as F
from scipy.sparse import csr_matrix, kron, eye
from scipy.sparse.linalg import spsolve
from scipy.sparse import vstack

def get_gpu_memory_map():
    """Get the current gpu usage.

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory_map


"""
# --------------------------------------------
# basic functions
# --------------------------------------------
"""
def upsample(x, sf=3):
    '''s-fold upsampler

    Upsampling the spatial size by filling the new entries with zeros

    x: tensor image, NxCxWxH
    '''
    st = 0
    z = torch.zeros((x.shape[0], x.shape[1], x.shape[2]*sf, x.shape[3]*sf)).type_as(x)
    z[..., st::sf, st::sf].copy_(x)
    return z


def downsample(x, sf=3):
    '''s-fold downsampler

    Keeping the upper-left pixel for each distinct sfxsf patch and discarding the others

    x: tensor image, NxCxWxH
    '''
    st = 0
    return x[..., st::sf, st::sf]


def downsample_np(x, sf=3):
    st = 0
    return x[st::sf, st::sf, ...]


def filter_tensor(x, sf=3):
    z = torch.zeros(x.shape)
    z[..., ::sf, ::sf].copy_(x[..., ::sf, ::sf])
    return z


def hadamard(x, kmap):
    # Compute hadamard product (pixel-wise)
    # x: input of shape (C,H,W)
    # kmap: input of shape (H,W)

    C,H,W = x.shape
    kmap = kmap.view(1, H, W)
    kmap = kmap.repeat(C, 1, 1)
    return (x * kmap)


def convolve_tensor(x, k):
    # Compute product convolution
    # x: input of shape (C,H,W)
    # k: input of shape (H_k,W_k)

    H_k, W_k = k.shape
    C, H, W = x.shape
    k = torch.flip(k, dims =(0,1))
    k = k.view(1, 1, H_k, W_k).repeat(C, 1, 1, 1)
    x = x[None]
    x = torch.nn.functional.pad(x, (W_k//2,W_k//2,H_k//2,H_k//2), mode='circular')
    o = torch.nn.functional.conv2d(x, k, groups=C, padding=0, stride=1)
    return o[0]


def cross_correlate_tensor(x, k):
    # x: input of shape (C,H,W)
    # k: input of shape (H_k,W_k)
    
    C, H, W = x.shape
    H_k, W_k = k.shape
    k = k.view(1, 1, H_k, W_k).repeat(C, 1, 1, 1)
    x = x[None]
    x = torch.nn.functional.pad(x, (W_k//2,W_k//2,H_k//2,H_k//2), mode='circular')
    o = torch.nn.functional.conv2d(x, k, groups=C, padding=0, stride=1)
    return o[0]


def o_leary(x, kmap, basis):
    # Apply O'Leary convolution model blurry = sum(U_i H_i x)
    # x: input of shape (C,H,W)
    # kmap: input of shape (P,H,W)
    # basis: input of shape (P,H_k,W_k)

    assert len(kmap) == len(basis), str(len(kmap)) + ',' +  str(len(basis))
    c = 0
    for i in range(len(kmap)):
        c += hadamard(convolve_tensor(x, basis[i]), kmap[i])
    return c



def o_leary_batch(x, kmap, basis):
    # Apply O'Leary convolution model blurry = sum(U_i H_i x)
    # x: input of shape (B,C,H,W)
    # kmap: input of shape (B,P,H,W)
    # basis: input of shape (B,P,H_k,W_k)

    assert len(x) == len(kmap) and len(kmap) == len(basis), print("Batch size must be the same for all inputs")
    
    return torch.cat([o_leary(x[i], kmap[i], basis[i])[None] for i in range(len(x))])


def transpose_o_leary(x, kmap, basis):
    # Apply the transpose of O'Leary convolution model blurry = sum(H_i^T U_i x)
    # x: input of shape (C,H,W)
    # kmap: input of shape (P,H,W)
    # basis: input of shape (P,H_k,W_k)
    
    assert len(kmap) == len(basis), str(len(kmap)) + ',' +  str(len(basis))
    c = 0
    for i in range(len(kmap)):
        c += cross_correlate_tensor(hadamard(x, kmap[i]), basis[i])
    return c


def transpose_o_leary_batch(x, kmap, basis):
    # Apply the transpose of O'Leary convolution model blurry = sum(H_i^T U_i x)
    # x: input of shape (B,C,H,W)
    # kmap: input of shape (B,P,H,W)
    # basis: input of shape (B,P,H_k,W_k)

    assert len(x) == len(kmap) and len(kmap) == len(basis), print("Batch size must be the same for all inputs")
    
    return torch.cat([transpose_o_leary(x[i], kmap[i], basis[i])[None] for i in range(len(x))])


def transpose_pmpb_batch(x, positions, intrinsics):
    # Apply the transpose of PMPB model model blurry = sum(H_i^T U_i x)
    # x: input of shape (B,C,H,W)
    # posiitons: input of shape (B,P,H,W)
    # intrinsics: input of shape (B,P,H_k,W_k)

    assert len(x) == len(kmap) and len(kmap) == len(basis), print("Batch size must be the same for all inputs")
    
    return torch.cat([transpose_o_leary(x[i], kmap[i], basis[i])[None] for i in range(len(x))])


"""
# --------------------------------------------
# (1) Prior module; ResUNet: act as a non-blind denoiser
# x_k = P(z_k, beta_k)
# --------------------------------------------
"""


class ResUNet(nn.Module):
    def __init__(self, in_nc=4, out_nc=3, nc=[64, 128, 256, 512], nb=2, act_mode='R', downsample_mode='strideconv', upsample_mode='convtranspose'):
        super(ResUNet, self).__init__()

        self.m_head = B.conv(in_nc, nc[0], bias=False, mode='C')

        # downsample
        if downsample_mode == 'avgpool':
            downsample_block = B.downsample_avgpool
        elif downsample_mode == 'maxpool':
            downsample_block = B.downsample_maxpool
        elif downsample_mode == 'strideconv':
            downsample_block = B.downsample_strideconv
        else:
            raise NotImplementedError('downsample mode [{:s}] is not found'.format(downsample_mode))

        self.m_down1 = B.sequential(*[B.ResBlock(nc[0], nc[0], bias=False, mode='C'+act_mode+'C') for _ in range(nb)], downsample_block(nc[0], nc[1], bias=False, mode='2'))
        self.m_down2 = B.sequential(*[B.ResBlock(nc[1], nc[1], bias=False, mode='C'+act_mode+'C') for _ in range(nb)], downsample_block(nc[1], nc[2], bias=False, mode='2'))
        self.m_down3 = B.sequential(*[B.ResBlock(nc[2], nc[2], bias=False, mode='C'+act_mode+'C') for _ in range(nb)], downsample_block(nc[2], nc[3], bias=False, mode='2'))

        self.m_body  = B.sequential(*[B.ResBlock(nc[3], nc[3], bias=False, mode='C'+act_mode+'C') for _ in range(nb)])

        # upsample
        if upsample_mode == 'upconv':
            upsample_block = B.upsample_upconv
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.upsample_pixelshuffle
        elif upsample_mode == 'convtranspose':
            upsample_block = B.upsample_convtranspose
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))

        self.m_up3 = B.sequential(upsample_block(nc[3], nc[2], bias=False, mode='2'), *[B.ResBlock(nc[2], nc[2], bias=False, mode='C'+act_mode+'C') for _ in range(nb)])
        self.m_up2 = B.sequential(upsample_block(nc[2], nc[1], bias=False, mode='2'), *[B.ResBlock(nc[1], nc[1], bias=False, mode='C'+act_mode+'C') for _ in range(nb)])
        self.m_up1 = B.sequential(upsample_block(nc[1], nc[0], bias=False, mode='2'), *[B.ResBlock(nc[0], nc[0], bias=False, mode='C'+act_mode+'C') for _ in range(nb)])

        self.m_tail = B.conv(nc[0], out_nc, bias=False, mode='C')

    def forward(self, x):
        
        h, w = x.size()[-2:]
        paddingBottom = int(np.ceil(h/8)*8-h)
        paddingRight = int(np.ceil(w/8)*8-w)
        x = nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(x)
        
        x1 = self.m_head(x)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body(x4)
        x = self.m_up3(x+x4)
        x = self.m_up2(x+x3)
        x = self.m_up1(x+x2)
        x = self.m_tail(x+x1)

        x = x[..., :h, :w]

        return x


"""
# --------------------------------------------
# (2) Data module, closed-form solution
# --------------------------------------------
"""


class DataNet(nn.Module):
    def __init__(self):
        super(DataNet, self).__init__()
        
    def forward_pos(self, x, STy, alpha, sf):
        I = torch.ones_like(STy) * alpha
        I[...,::sf,::sf] += 1
        return (STy + alpha * x) / I
        
    def forward_zer(self, x, STy, sf):
        res = x
        res[...,::sf,::sf] = STy[...,::sf,::sf]
        return res

    def forward(self, x, STy, alpha, sf, sigma):
        index_zer = (sigma.view(-1) == 0)
        index_pos = (sigma.view(-1) > 0)
        
        res = torch.zeros_like(x)
        
        res[index_zer,...] = self.forward_zer(x[index_zer,...], STy[index_zer,...], sf)
        res[index_pos,...] = self.forward_pos(x[index_pos,...], STy[index_pos,...], alpha[index_pos,...], sf)
        
        return res

"""
# --------------------------------------------
# (3) Hyper-parameter module
# --------------------------------------------
"""

class HyPaNet(nn.Module):
    def __init__(self, in_nc=2, out_nc=8, channel=64):
        super(HyPaNet, self).__init__()
        self.mlp = nn.Sequential(
                nn.Conv2d(in_nc, channel, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, out_nc, 1, padding=0, bias=True),
                nn.Softplus())

    def forward(self, x):
        x = self.mlp(x) + 1e-6
        return x


def apply_gradient_torch(image_tensor):
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=image_tensor.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=image_tensor.device).view(1, 1, 3, 3)

    sobel_x = sobel_x.repeat(image_tensor.shape[1], 1, 1, 1)
    sobel_y = sobel_y.repeat(image_tensor.shape[1], 1, 1, 1)

    grad_x = F.conv2d(image_tensor, sobel_x, padding=1, groups=image_tensor.shape[1])
    grad_y = F.conv2d(image_tensor, sobel_y, padding=1, groups=image_tensor.shape[1])
    
    grad = torch.sqrt(grad_x**2 + grad_y**2)
    return grad

def high_pass_filter_torch(image_tensor, size=3):
    kernel_value = -1 / (size * size)
    kernel = torch.full((size, size), kernel_value, dtype=torch.float32, device=image_tensor.device)
    kernel[size // 2, size // 2] = 1 + kernel_value
    kernel = kernel.view(1, 1, size, size)

    kernel = kernel.repeat(image_tensor.shape[1], 1, 1, 1)

    high_pass = F.conv2d(image_tensor, kernel, padding=size//2, groups=image_tensor.shape[1])
    return high_pass

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
"""
# --------------------------------------------
#   Main
# --------------------------------------------
"""


class DMBSR(nn.Module):
    def __init__(self, n_iter=8, h_nc=64, in_nc=4, out_nc=3, nc=[64, 128, 256, 512], nb=2, act_mode='R', downsample_mode='strideconv', upsample_mode='convtranspose'):
        super(DMBSR, self).__init__()

        self.d = DataNet()
        self.p = ResUNet(in_nc=in_nc, out_nc=out_nc, nc=nc, nb=nb, act_mode=act_mode, downsample_mode=downsample_mode, upsample_mode=upsample_mode)
        self.h = HyPaNet(in_nc=2, out_nc=(n_iter+1)*3, channel=h_nc)
        self.n = n_iter

    def forward(self, y, kmap, basis, sf, sigma):
        '''
        x: tensor, NxCxWxH
        k: tensor, Nx(1,3)xwxh
        sf: integer, 1
        sigma: tensor, Nx1x1x1
        '''

        '''
        numpy_array = y[1,:,:,:].permute(1, 2, 0).cpu().numpy()

        # Step 3: Use Matplotlib to visualize the tensor
        plt.imshow(numpy_array, cmap='viridis')  # 'viridis' is a colormap, you can change it
        plt.colorbar()  # optional, to show the color scale
        plt.title('Visualization of a 2D PyTorch Tensor')
        file_path = 'output_image.png'
        plt.savefig(file_path)
        exit(0)
        '''
        # y = npls(y) # npls denoise
        # y = apply_gradient_torch(y) # gradient
        # y = high_pass_filter_torch(y) # high-pass-filter
        # Initialization
        STy = upsample(y, sf)
        x_0 = nn.functional.interpolate(y, scale_factor=sf, mode='nearest')
        z_0 = x_0
        h_0 = o_leary_batch(x_0, kmap, basis)
        u_0 = torch.zeros_like(z_0)
        ab = self.h(torch.cat((sigma, torch.tensor(sf).type_as(sigma).expand_as(sigma)), dim=1))
        # print(sf, x_0.shape, h_0.shape,u_0.shape, STy.shape, y.shape)
        
        for i in range(self.n):
            # Hyper-params
            alpha = ab[:, i:i+1, ...]
            beta = ab[:, i+(self.n+1):i+(self.n+1)+1, ...]
            gamma = ab[:, i+2*(self.n+1):i+2*(self.n+1)+1, ...]

            # ADMM steps
            i_0 = x_0 - beta * transpose_o_leary_batch(h_0 - z_0 + u_0, kmap, basis)
            #print(x_0.shape, h_0.shape,u_0.shape, STy.shape)
            x_0 = self.p(torch.cat((i_0, gamma.repeat(1, 1, i_0.size(2), i_0.size(3))), dim=1))
            #print(x_0.shape, h_0.shape,u_0.shape, STy.shape)
            h_0 = o_leary_batch(x_0, kmap, basis)
            
            z_0 = self.d(h_0 + u_0, STy, alpha, sf, sigma)
            u_0 = u_0 + h_0 - z_0

        # Hyper-params
        beta = ab[:, 2*self.n+1:2*(self.n+1), ...]
        gamma = ab[:, 3*self.n+2:, ...]

        i_0 = x_0 - beta * transpose_o_leary_batch(h_0 - z_0 + u_0, kmap, basis)
        x_0 = self.p(torch.cat((i_0, gamma.repeat(1, 1, i_0.size(2), i_0.size(3))), dim=1))

        return x_0

