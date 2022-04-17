import torch
from torch.utils.data import Dataset
import numpy as np
import os.path as osp


class DatasetBase(Dataset):
    def __init__(self, root, split, transform=None, target_transform=None, co_transforms=None):
        super(DatasetBase, self).__init__()

        self.root = root
        self.split = split
        self.data_dir = osp.join(self.root, split)
        self.index = self.collect_data_index()

        # for test
        if split == 'val':
            self.index = self.index[:-4]
            self.index.extend(['val_1032', 'val_1033', 'val_1034', 'val_1035'])

        self.transform = transform
        self.target_transform = target_transform
        self.co_transforms = co_transforms

        self.gt_max_num_classes = self.get_gt_max_num_classes()

    def get_gt_max_num_classes(self):
        raise NotImplementedError

    def collect_data_index(self):
        raise NotImplementedError

    def read_image(self, idx):
        raise NotImplementedError

    def read_label(self, idx):
        raise NotImplementedError

    def __getitem__(self, item):
        idx = self.index[item]

        img = self.read_image(idx)
        gt = self.read_label(idx)

        if self.co_transforms is not None:
            img, gt = self.co_transforms([img], gt)

        if self.transform is not None:
            img = self.transform(img[0])

        if self.target_transform is not None:
            gt = self.target_transform(gt)

        return img, gt

    def __len__(self):
        return len(self.index)
