# Change directory to kaolin
# The kaolin library must be downloaded from <https://github.com/NVIDIAGameWorks/kaolin> and installed with instructions from their README
# Note: The setenv.sh script must be run to set up proper kaolin paths

# import necessary libraries
import kaolin as kal

from kaolin.datasets import modelnet
from torch.utils.data import DataLoader

import torch
from torchvision import transforms

import numpy as np
from pathlib import Path
from functools import partial


def convert_mesh_to_pt_cloud(mesh, num_points=5000):
    return kal.conversions.trianglemesh_to_pointcloud(mesh, num_points)[0]


class ModelNet10Category:
    modelnet10_category_list = ['bathtub', 'bed',        'chair',  'desk',  'dresser',
                                'monitor', 'night_stand', 'sofa',  'table',  'toilet']


class ModelNet40Category:
    modelnet40_category_list = ['airplane', 'bench', 'bowl', 'cone', 'desk', 'flower_pot', 'keyboard', 'mantel', 'person', 'radio',
                                'sofa', 'table', 'tv_stand', 'xbox', 'bathtub', 'bookshelf', 'car', 'cup', 'door', 'glass_box', 'lamp', 'monitor',
                                'piano', 'range_hood', 'stairs', 'tent', 'vase', 'bed', 'bottle', 'chair', 'curtain', 'dresser',
                                'guitar', 'laptop', 'night_stand', 'plant', 'sink', 'stool', 'toilet', 'wardrobe']


def unload_off_to_npy(category_list, split='train', src_dir='./data/ModelNet10/',
                      dump_npy_dir='./data/ModelNet10_train_npy/'):
    """
    Unload all .off files into a .npy pickle format

    category_list = ['chair'] ...etc
    split = 'train' or 'test'
    Loads the `.off` files from ModelNet as torch.Tensors and saves them as `.npy` files for faster loading

    WARNING: Requires a lot of time
    The original `.off` ModelNet10 files must be downloaded from <https://modelnet.cs.princeton.edu/>

    Example usage
    # unloads ModelNet40 off to npy
    unload_off_to_npy(modelnet40_category_list, split='train', src_dir='./data/ModelNet40/', dump_npy_dir='./data/ModelNet40_train_npy/')
    unload_off_to_npy(modelnet40_category_list, split='test', src_dir='./data/ModelNet40/', dump_npy_dir='./data/ModelNet40_test_npy/')
    """
    for category in category_list:
        dump_npy_file = 'modelnet_' + category + '.npy'
        npy_dump_file = Path(dump_npy_dir + dump_npy_file)
        if npy_dump_file.is_file():
            print(f"{dump_npy_dir+dump_npy_file} already exists. Skipping now")
            continue

        print(f"Unloading {category} to npy")
        mdnet = modelnet.ModelNet(root=src_dir, categories=[category], split=split,
                                  transform=transforms.Compose([convert_mesh_to_pt_cloud]))
        print(len(mdnet))
        # data_loader = DataLoader(mdnet, batch_size=1,
        #                         shuffle=True, num_workers=0)
        X = None
        # Load the entire .off files into one Tensor object X one at a time
        for i in range(len(mdnet)):
            try:
                if X is None:
                    X = mdnet[i][0].unsqueeze(0)
                else:
                    X = torch.cat([X, mdnet[i][0].unsqueeze(0)], dim=0)
            except Exception as e:
                print(e)
                print(f"Skipped {i}")
            if i % 20:
                if X is not None:
                    print(X.shape)

        dump_npy_file = 'modelnet_' + category + '.npy'
        if X is not None:
            np.save(dump_npy_dir + dump_npy_file, X.numpy())
            print(
                f"Dumped category {category} with shape {X.shape} under {dump_npy_dir} as {dump_npy_file}")
            return
        print("Invalid off file. Count not read any values")
