import numpy as np
import os.path as osp
from osgeo import gdal
import os
import cv2

from .dataset_base import DatasetBase

import flow_transforms


class DatasetWhu(DatasetBase):
    def __init__(self, root, split, transform=None, target_transform=None, co_transforms=None):
        super(DatasetWhu, self).__init__(root, split, transform, target_transform, co_transforms)

    def get_gt_max_num_classes(self):
        return 2

    def collect_data_index(self):
        return [name[:-4] for name in os.listdir(osp.join(self.data_dir, 'image'))]

    def read_image(self, idx):
        dataset = gdal.Open(osp.join(self.data_dir, 'image', idx + '.tif'), gdal.GA_ReadOnly)
        band_arr_list = [dataset.GetRasterBand(i + 1).ReadAsArray() for i in range(dataset.RasterCount)]
        img = np.dstack(band_arr_list)

        return img

    def read_label(self, idx):

        # dataset = gdal.Open(osp.join(self.data_dir, 'label', idx + '.tif'), gdal.GA_ReadOnly)
        # band_arr = dataset.GetRasterBand(1).ReadAsArray()

        lbl = cv2.imread(osp.join(self.data_dir, 'label', idx + '.tif'), cv2.IMREAD_UNCHANGED) / 255
        lbl = np.expand_dims(lbl, axis=-1)

        return lbl


def WHU(root, transform=None, target_transform=None, val_transform=None, co_transform=None, co_transform_val=None):
    train_dataset = DatasetWhu(root, 'train', transform, target_transform, co_transform)
    val_dataset = DatasetWhu(root, 'val', val_transform, target_transform, co_transform_val)

    return train_dataset, val_dataset


def WHU_test(root, transform=None, target_transform=None, co_transform=None):
    test_dataset = DatasetWhu(root, 'test', transform, target_transform, co_transform)

    return test_dataset
