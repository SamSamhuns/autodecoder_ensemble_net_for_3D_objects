import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.metrics import auc

import torch
from torch.autograd import Variable


class HyperParameter:
    def __init__(self,
                 l2_reg=None,
                 encoding_size=256,
                 encoding_iters=50,
                 num_point_cloud=3,
                 epochs=4,
                 lr=0.00001,
                 batch_size=32):

        self.l2_reg = l2_reg
        self.learning_rate = lr
        self.encoding_size = encoding_size
        self.encoding_iters = encoding_iters
        self.num_point_cloud = num_point_cloud
        self.epochs = epochs
        self.batch_size = batch_size

    def __repr__(self):
        return f"l2_reg: {self.l2_reg}\n" + \
               f"learning_rate: {self.learning_rate}\n" + \
               f"encoding_size: {self.encoding_size}\n" + \
               f"encoding_iters: {self.encoding_iters}\n" + \
               f"num_point_cloud: {self.num_point_cloud}\n" + \
               f"epochs: {self.epochs}\n" + \
               f"batch_size: {self.batch_size}\n"


class DirectorySetting:

    def __init__(self,
                 AUTODECODER_TRAINED_WEIGHT_DIR="./autodecoder_trained_weights",
                 CLASSIFIER_TRAINED_WEIGHT_DIR="./classifier_trained_weights"):

        self.AUTODECODER_TRAINED_WEIGHT_DIR = AUTODECODER_TRAINED_WEIGHT_DIR
        self.CLASSIFIER_TRAINED_WEIGHT_DIR = CLASSIFIER_TRAINED_WEIGHT_DIR

        os.makedirs(self.AUTODECODER_TRAINED_WEIGHT_DIR, exist_ok=True)
        os.makedirs(self.CLASSIFIER_TRAINED_WEIGHT_DIR, exist_ok=True)

    def __repr__(self):
        return f"AUTODECODER_TRAINED_WEIGHT_DIR: {self.AUTODECODER_TRAINED_WEIGHT_DIR}\n" + \
               f"CLASSIFIER_TRAINED_WEIGHT_DIR: {self.CLASSIFIER_TRAINED_WEIGHT_DIR}\n"


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


def visualize_npy(npy_3d_matrix, save_img_fpath='./img/image.png', remove_ticks=True):
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
        ax.set_axis_off()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
    ax.scatter(x, y, z, zdir='z', c='red', s=0.9)

    if save_img_fpath is not None:
        plt.savefig(save_img_fpath)


def count_unique_labels(y):
    """
    Counts number of unique labels in y
    y must be a np.ndarray
    """
    print(y.shape)
    bcount = np.bincount(y.reshape(-1,))
    non_zero_idx = np.nonzero(bcount)[0]

    for label, count in zip(non_zero_idx, bcount[non_zero_idx]):
        print(label, count)
        

def get_X_y_from_npy(data_src, shuffle_seed=None):
    """
    Returns X, y from data_src directory (Different files such as aeroplane.npy, chari.npy etc)
    y = label is automatically assigned based on filename

    Example:
    # Warning: The train and test directories must have the same number of files

    train = './data/ModelNet10_train_npy/'
    test = './data/ModelNet10_test_npy/'

    X_train, y_train = get_X_y_from_npy(train, 100)
    X_test, y_test = get_X_y_from_npy(test, 100)
    """
    X, y = None, None
    label = 0
    for path in Path(data_src).iterdir():
        X_curr = np.load(path)
        if X is None and y is None:
            X = X_curr
            y = np.full((X_curr.shape[0], 1), label, dtype=np.int64)
        else:
            X = np.concatenate([X, X_curr], axis=0)
            y = np.concatenate([y, np.full((X_curr.shape[0], 1), label, dtype=np.int64)],
                               axis=0)
        label += 1
        print(f"Adding {path} to X. X is now {X.shape}")

    if shuffle_seed is not None:
        np.random.seed(shuffle_seed)
        shuffled_idxs = np.random.permutation(X.shape[0])
        X, y = X[shuffled_idxs], y[shuffled_idxs]

    return X, y


def get_train_test_split_from_npy(data_src, test_split=0.1, shuffle_seed=None):
    """
    Given data_src .npy file, returns 
    
    Returns X_train, X_test, y_train, y_test
    if shuffle_seed is None, X and y will not be shuffled
    """
    X, y = get_X_y_from_npy(data_src, shuffle_seed=shuffle_seed)

    test_idx_end = int(X.shape[0]*0.1)
    X_train, X_test = X[test_idx_end:], X[:test_idx_end]
    y_train, y_test = y[test_idx_end:], y[:test_idx_end]

    return X_train, X_test, y_train, y_test


def plot_roc_curve(fp_rate, tp_rate, title='Receiver operating characteristic',
                   figsize=(8, 6)):
    """
    fp_rate = False Positive Rate 
    tp_rate = True Positive Rate
    """
    roc_auc = auc(fp_rate, tp_rate)

    plt.figure(figsize=figsize)
    plt.plot(fp_rate, tp_rate, color='darkorange',
             lw=2, label='ROC curve (area = %0.3f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='green', lw=2, linestyle='--')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', size=12)
    plt.ylabel('True Positive Rate', size=12)
    plt.title(title, size=15)
    plt.legend(loc="lower right", fontsize=15)
    plt.show()


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