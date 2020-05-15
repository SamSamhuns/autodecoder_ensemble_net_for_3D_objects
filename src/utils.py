import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import torch
from torch.autograd import Variable

def chamfer_loss(x, y, ps=91):
    """
    Chamfer distance from shape x to shape y
    ps = y.shape[-1]
    """
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

def find_encoding(X, y, autodecoder, encoding_iters=300,
                  encoding_size=256, lr=5e-4, l2_reg=False):
    """
    Generate the encoding (latent vector) for each data in X
    """

    def _adjust_lr(initial_lr, optimizer, num_iters, decreased_by, adjust_lr_every):

        lr = initial_lr * ((1 / decreased_by) **
                           (num_iters // adjust_lr_every))
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    decreased_by = 10
    adjust_lr_every = encoding_iters // 2

    encoding = torch.ones(X.shape[0], encoding_size).normal_(
        mean=0, std=1.0 / math.sqrt(encoding_size)).cuda()

    encoding.requires_grad = True
    optimizer = torch.optim.Adam([encoding], lr=lr)
    loss_num = 0

    for i in range(encoding_iters):
        autodecoder.eval()
        _adjust_lr(lr, optimizer, i, decreased_by, adjust_lr_every)
        optimizer.zero_grad()
        y_pred = autodecoder(X, encoding)
        loss = chamfer_loss(y_pred, y, ps=y.shape[-1])

        if l2_reg:
            loss += 1e-4 * torch.mean(encoding.pow(2))
        loss.backward()
        optimizer.step()

        if i % 50 == 0:
            print(i, loss.cpu().data.numpy(), encoding.norm())
        loss_num = loss.cpu().data.numpy()

    return loss_num, encoding
        
def print_model_metrics(total_loss, 
                        same_corr_cnt, 
                        same_incorr_cnt, 
                        diff_corr_cnt, 
                        diff_incorr_cnt,
                        len_test_ds):
    """
    Prints Total loss, Total Accuracy
    Precision, Recall, and f1 Score for both classes
    """
    
    precision = same_corr_cnt / (same_corr_cnt+diff_incorr_cnt)
    recall = same_corr_cnt / (same_corr_cnt+same_incorr_cnt)
    print("------------------ Evaluation Report ------------------")
    print(f"After {len_test_ds} test points")
    print(f"Total Accuracy: {(same_corr_cnt+diff_corr_cnt)/(2*len_test_ds)}")
    print(f"Total loss {total_loss}")
    print()

    print(f"Metrics for the same class:")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {(2*precision*recall)/(precision+recall)}")

    precision = diff_corr_cnt / (diff_corr_cnt+same_incorr_cnt)
    recall = diff_corr_cnt / (diff_corr_cnt+diff_incorr_cnt)
    print()
    print(f"Metrics for the diff class:")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {(2*precision*recall)/(precision+recall)}")