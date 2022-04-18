import numpy as np
import os.path as osp
from osgeo import gdal
import cv2

from .dataset_base import DatasetBase


class DatasetSpaceNet(DatasetBase):
    def __init__(self, root, split,  transform=None, target_transform=None, co_transforms=None):
        self.image_dir = osp.join(root, 'RGB-PanSharpen-color-rescaled')
        self.gt_dir = osp.join(root, 'GT')
        self.index_file_path = osp.join(root, split + '.txt')

        super(DatasetSpaceNet, self).__init__(root, split, transform, target_transform, co_transforms)

        self.data_dir = None

    def get_gt_max_num_classes(self):
        return 2

    def collect_data_index(self):
        with open(self.index_file_path, 'r') as f:
            names = f.readlines()

        return [name[:-5] for name in names]

    def read_image(self, idx):
        dataset = gdal.Open(osp.join(self.image_dir, 'RGB-PanSharpen_'+idx+'.tif'), gdal.GA_ReadOnly)
        band_arr_list = [dataset.GetRasterBand(i + 1).ReadAsArray() for i in range(dataset.RasterCount)]
        img = np.dstack(band_arr_list)

        return img

    def read_label(self, idx):
        lbl = cv2.imread(osp.join(self.gt_dir, idx+'.tif'), cv2.IMREAD_UNCHANGED) / 255
        lbl = np.expand_dims(lbl, axis=-1)

        return lbl


def _space_net(root, transform=None, target_transform=None, val_transform=None, co_transform=None, co_transform_val=None):
    train_dataset = DatasetSpaceNet(root, 'train', transform, target_transform, co_transform)
    val_dataset = DatasetSpaceNet(root, 'val', val_transform, target_transform, co_transform_val)

    return train_dataset, val_dataset


def SpaceNetSH(root, transform=None, target_transform=None, val_transform=None, co_transform=None, co_transform_val=None):
    return _space_net(root, transform, target_transform, val_transform, co_transform, co_transform_val)


def SpaceNetSH_test(root, transform=None, target_transform=None, co_transform=None):
    return DatasetSpaceNet(root, 'test', transform, target_transform, co_transform)


def SpaceNetVegas(root, transform=None, target_transform=None, val_transform=None, co_transform=None, co_transform_val=None):
    return _space_net(root, transform, target_transform, val_transform, co_transform, co_transform_val)


def SpaceNetVegas_test(root, transform=None, target_transform=None, co_transform=None):
    return DatasetSpaceNet(root, 'test', transform, target_transform, co_transform)
