import os
import os.path as osp

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np

import models
import datasets
import torchvision.transforms as transforms
import flow_transforms
from train_util import update_spixl_map, init_spixel_grid
from connectivity import enforce_connectivity

from tqdm import tqdm
import argparse


def get_dataset_mean_value(ds_name):
    if ds_name == 'WHU':
        return [0.411, 0.432, 0.45]
    elif ds_name == 'Inria':
        return [0.40640827, 0.42793751, 0.39354717]
    elif ds_name == 'SpaceNetVegas':
        return [0.4973, 0.4963, 0.4959]
    else:
        return [0, 0, 0]


def test(opts):
    # prepare output directory
    output_dir = osp.join(opts.output_dir, opts.dataset, f'ts{opts.target_size}')
    os.makedirs(output_dir, exist_ok=True)

    # prepare data
    mean_value = get_dataset_mean_value(opts.dataset)

    normalize_transform = transforms.Compose([
        flow_transforms.ArrayToTensor(),
        transforms.Normalize(mean=[0, 0, 0], std=[255, 255, 255]),
        transforms.Normalize(mean=mean_value, std=[1, 1, 1])
    ])

    test_set = datasets.__dict__[opts.dataset+'_test'](opts.data_dir, normalize_transform)

    tar_size = opts.target_size

    # prepare initial sp index map
    tmp_args = argparse.Namespace()
    setattr(tmp_args, 'input_img_height', tar_size)
    setattr(tmp_args, 'input_img_width', tar_size)
    setattr(tmp_args, 'downsize', 16)
    setattr(tmp_args, 'batch_size', 1)

    init_sp_idx, _ = init_spixel_grid(tmp_args, b_train=False)

    # prepare model
    model = models.SpixelNet1l_bn(data=torch.load(opts.pretrained_model, map_location='cuda')).cuda()
    model.eval()

    for i in tqdm(range(len(test_set)), total=len(test_set)):
        img, _ = test_set[i]
        img = img.unsqueeze(0)

        name_id = test_set.index[i]

        ori_size = img.shape[-1]

        # resize image to target size
        img = img.cuda()
        img_scaled = F.interpolate(img, size=(tar_size, tar_size), mode='bicubic', align_corners=True)

        # run
        pred_q = model(img_scaled)

        # update super-pixel map
        sp_map_scaled = update_spixl_map(init_sp_idx, pred_q)
        sp_map_ori = F.interpolate(sp_map_scaled.float(), size=(ori_size, ori_size), mode='nearest').type(torch.long)

        # enforce connectivity
        sp_map_ori_np = sp_map_ori.squeeze().detach().cpu().numpy()
        num_sp = np.max(sp_map_ori_np) + 1
        segment_size = ori_size ** 2 / num_sp
        min_size = int(0.06 * segment_size)
        max_size = int(3 * segment_size)
        sp_map = enforce_connectivity(sp_map_ori_np[None, :, :], min_size, max_size)[0]

        # save super-pixel map
        if opts.output_file_type == 'csv':
            with open(osp.join(output_dir, name_id+'.csv'), 'w') as f:
                for k in range(ori_size):
                    f.write(','.join([str(v) for v in sp_map[k]]))
                    f.write('\n')
        else:
            torch.save({'sp_map': torch.from_numpy(sp_map), 'n_spixels': np.max(sp_map)+1},
                       osp.join(output_dir, name_id+'.pt'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--output_file_type', type=str, choices=['pt', 'csv'], default='pt')
    parser.add_argument('--pretrained_model', type=str)
    parser.add_argument('--target_size', type=int, default=512)

    args = parser.parse_args()
    test(args)
