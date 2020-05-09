import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import torch
from torch.autograd import Variable

def chamfer_loss(x, y, ps=91):
    A = x.cuda()
    B = y.cuda()
    A = A.permute(0, 2, 1)
    B = B.permute(0, 2, 1)
    r = torch.sum(A * A, dim=2)
    r = r.unsqueeze(-1)
    r1 = torch.sum(B * B, dim=2)
    r1 = r1.unsqueeze(-1)
    t = (r.repeat(1, 1, ps) - 2 * torch.bmm(A, B.permute(0, 2, 1)) +
         r1.permute(0, 2, 1).repeat(1, ps, 1))
    d1, _ = t.min(dim=1)
    d2, _ = t.min(dim=2)
    ls = (d1 + d2) / 2
    return ls.mean()

def visualize_npy(npy_3d_matrix, save_img_fpath='./img/npy_01.png', remove_ticks=True):
    """
    npy_3d_matrix must of size num_points * 3
    """
    x = npy_3d_matrix[:, 0]
    y = npy_3d_matrix[:, 1]
    z = npy_3d_matrix[:, 2]

    fig = plt.figure(figsize=(10, 8))

    ax = fig.add_subplot(111, projection='3d')
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    if remove_ticks:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
    ax.scatter(x, y, z, zdir='z', c='red', s=0.9)

    if save_img_fpath is not None:
        plt.savefig(save_img_fpath)