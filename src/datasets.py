import random
import numpy as np


import torch

from torch.utils.data import Dataset, DataLoader


class PointNetDS(Dataset):
    """
    Create train dataset
    """

    def __init__(self, data, sampling_interval=3):
        # sample every sampling_interval-th point to speed up
        self.data = data.transpose((0, 2, 1))[:, :, ::sampling_interval]
        self.data = torch.from_numpy(self.data).float()

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx]


class PointDriftDS(Dataset):
    """
    Pairs each shape with one shape from the same class and one shape from a different class
    """

    def __init__(self, data, labels, sampling_interval=3):
        # sample every sampling_interval-th point to speed up
        self.data = data.transpose((0, 2, 1))[:, :, ::sampling_interval]
        self.labels = labels.squeeze()

        self.same_cls = []
        self.diff_cls = []
        idx_arr = np.arange(self.data.shape[0])
        same_idx = []
        diff_idx = []
        for i in range(self.labels.max() + 1):
            same_idx.append(idx_arr[self.labels == i])
            diff_idx.append(idx_arr[self.labels != i])

        for i in range(data.shape[0]):
            same = same_idx[self.labels[i]]
            diff = diff_idx[self.labels[i]]
            self.same_cls.append(same[random.randint(0, len(same) - 1)])
            self.diff_cls.append(diff[random.randint(0, len(diff) - 1)])
        self.data = torch.from_numpy(self.data).float()

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        X = self.data[idx]
        same_cls_data = self.data[self.same_cls[idx]]
        diff_cls_data = self.data[self.diff_cls[idx]]

        return X, same_cls_data, diff_cls_data, idx


class EncodingDS(Dataset):
    """
    Generate encoding for each pair of shapes in the PointDriftDS
    """

    def __init__(self, PDDS, autodecoder, latent_size=256):
        self.PointDriftDS = PDDS
        self.autodecoder = autodecoder
        self.latent_size = latent_size
        self.same_cls = torch.zeros((len(self.PointDriftDS), latent_size))
        self.diff_cls = torch.zeros((len(self.PointDriftDS), latent_size))

    def train_encodings(
        self, find_encoding, num_iterations=50, lr=0.01, l2_reg=False, batch_size=16
    ):
        dl = DataLoader(self.PointDriftDS, batch_size=batch_size, shuffle=False)
        i = 0
        batch_cnt = 0
        same_cls_loss = 0.0
        diff_cls_loss = 0.0
        self.autodecoder.eval()

        for batch_idx, (x, same, diff, idx) in enumerate(dl):
            j = i + len(idx)
            loss, encoding = find_encoding(
                x,
                same,
                self.autodecoder,
                encoding_iters=num_iterations,
                encoding_size=self.latent_size,
                lr=lr,
                l2_reg=l2_reg,
            )
            same_cls_loss += loss
            self.same_cls[i:j] = encoding
            loss, encoding = find_encoding(
                x,
                diff,
                self.autodecoder,
                encoding_iters=num_iterations,
                encoding_size=self.latent_size,
                lr=lr,
                l2_reg=l2_reg,
            )
            diff_cls_loss += loss
            self.diff_cls[i:j] = encoding

            i = j
            batch_cnt += 1
        print("Encodings trained")
        return (
            self.same_cls,
            self.diff_cls,
            same_cls_loss / batch_cnt,
            diff_cls_loss / batch_cnt,
        )

    def __len__(self):
        return len(self.PointDriftDS)

    def __getitem__(self, idx):
        return (*self.PointDriftDS[idx], self.same_cls[idx], self.diff_cls[idx])
