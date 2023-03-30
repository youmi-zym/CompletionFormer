"""
    CompletionFormer
    ======================================================================

    KITTI Depth Completion Dataset Helper
"""


import os
import numpy as np
import json
import random
from . import BaseDataset

from PIL import Image
import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from .lidar import sample_lidar_lines


"""
KITTI Depth Completion json file has a following format:

{
    "train": [
        {
            "rgb": "rawdata/2011_09_30/2011_09_30_drive_0018_sync/image_02/data/0000000188.png",
            "depth": "data_depth_velodyne/train/2011_09_30_drive_0018_sync/proj_depth/velodyne_raw/image_02/0000000188.png",
            "gt": "data_depth_annotated/train/2011_09_30_drive_0018_sync/proj_depth/groundtruth/image_02/0000000188.png",
            "K": "rawdata/2011_09_30/calib_cam_to_cam.txt"
        }, ...
    ],
    "val": [
        {
            "rgb": "data_depth_selection/val_selection_cropped/image/2011_10_03_drive_0047_sync_image_0000000761_image_02.png",
            "depth": "data_depth_selection/val_selection_cropped/velodyne_raw/2011_10_03_drive_0047_sync_velodyne_raw_0000000761_image_02.png",
            "gt": "data_depth_selection/val_selection_cropped/groundtruth_depth/2011_10_03_drive_0047_sync_groundtruth_depth_0000000761_image_02.png",
            "K": "data_depth_selection/val_selection_cropped/intrinsics/2011_10_03_drive_0047_sync_image_0000000761_image_02.txt"
        }, ...
    ],
    "test": [
        {
        }, ...
    ]
}

Reference : https://github.com/XinJCheng/CSPN/blob/master/nyu_dataset_loader.py
"""


def read_depth(file_name):
    # loads depth map D from 16 bits png file as a numpy array,
    # refer to readme file in KITTI dataset
    assert os.path.exists(file_name), "file not found: {}".format(file_name)
    image_depth = np.array(Image.open(file_name))

    # Consider empty depth
    assert (np.max(image_depth) == 0) or (np.max(image_depth) > 255), \
        "np.max(depth_png)={}, path={}".format(np.max(image_depth), file_name)

    image_depth = image_depth.astype(np.float32) / 256.0
    return image_depth


# Reference : https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
def read_calib_file(filepath):
    """Read in a calibration file and parse into a dictionary."""
    data = {}

    with open(filepath, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass

    return data


class KITTIDC(BaseDataset):
    def __init__(self, args, mode):
        super(KITTIDC, self).__init__(args, mode)

        self.args = args
        mode = 'test' if mode == 'val' else mode
        self.mode = mode
        self.lidar_lines = args.lidar_lines

        if mode != 'train' and mode != 'val' and mode != 'test':
            raise NotImplementedError

        self.height = args.patch_height
        self.width = args.patch_width

        self.augment = self.args.augment

        with open(self.args.split_json) as json_file:
            json_data = json.load(json_file)
            self.sample_list = json_data[mode]

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        rgb, depth, gt, K = self._load_data(idx)

        if self.augment and self.mode == 'train':
            # Top crop if needed
            if self.args.top_crop > 0:
                width, height = rgb.size
                rgb = TF.crop(rgb, self.args.top_crop, 0,
                              height - self.args.top_crop, width)
                depth = TF.crop(depth, self.args.top_crop, 0,
                                height - self.args.top_crop, width)
                gt = TF.crop(gt, self.args.top_crop, 0,
                             height - self.args.top_crop, width)
                K[3] = K[3] - self.args.top_crop

            width, height = rgb.size

            _scale = np.random.uniform(1.0, 1.5) # resize scale > 1.0, no info loss
            scale = int(height * _scale)
            degree = np.random.uniform(-5.0, 5.0)
            flip = np.random.uniform(0.0, 1.0)

            # Horizontal flip
            if flip > 0.5:
                rgb = TF.hflip(rgb)
                depth = TF.hflip(depth)
                gt = TF.hflip(gt)
                K[2] = width - K[2]

            # Rotation
            rgb = TF.rotate(rgb, angle=degree, interpolation=InterpolationMode.BICUBIC)
            depth = TF.rotate(depth, angle=degree, interpolation=InterpolationMode.NEAREST)
            gt = TF.rotate(gt, angle=degree, interpolation=InterpolationMode.NEAREST)

            # Color jitter
            brightness = np.random.uniform(0.6, 1.4)
            contrast = np.random.uniform(0.6, 1.4)
            saturation = np.random.uniform(0.6, 1.4)

            rgb = TF.adjust_brightness(rgb, brightness)
            rgb = TF.adjust_contrast(rgb, contrast)
            rgb = TF.adjust_saturation(rgb, saturation)

            # Resize
            rgb = TF.resize(rgb, scale, interpolation=InterpolationMode.BICUBIC)
            depth = TF.resize(depth, scale, interpolation=InterpolationMode.NEAREST)
            gt = TF.resize(gt, scale, interpolation=InterpolationMode.NEAREST)

            K[0] = K[0] * _scale
            K[1] = K[1] * _scale
            K[2] = K[2] * _scale
            K[3] = K[3] * _scale

            # Crop
            width, height = rgb.size

            assert self.height <= height and self.width <= width, \
                "patch size is larger than the input size"

            ch, cw = self.height, self.width
            # ch, cw = 240, 480
            h_start = random.randint(0, height - ch)
            w_start = random.randint(0, width - cw)

            rgb = TF.crop(rgb, h_start, w_start, ch, cw)
            depth = TF.crop(depth, h_start, w_start, ch, cw)
            gt = TF.crop(gt, h_start, w_start, ch, cw)

            K[2] = K[2] - w_start
            K[3] = K[3] - h_start

            rgb = TF.to_tensor(rgb)
            rgb = TF.normalize(rgb, (0.485, 0.456, 0.406),
                               (0.229, 0.224, 0.225), inplace=True)

            depth = TF.to_tensor(np.array(depth))
            depth = depth / _scale

            gt = TF.to_tensor(np.array(gt))
            gt = gt / _scale
        elif self.mode in ['train', 'val']:
            # Top crop if needed
            if self.args.top_crop > 0:
                width, height = rgb.size
                rgb = TF.crop(rgb, self.args.top_crop, 0,
                              height - self.args.top_crop, width)
                depth = TF.crop(depth, self.args.top_crop, 0,
                                height - self.args.top_crop, width)
                gt = TF.crop(gt, self.args.top_crop, 0,
                             height - self.args.top_crop, width)
                K[3] = K[3] - self.args.top_crop

            # Crop
            width, height = rgb.size

            assert self.height <= height and self.width <= width, \
                "patch size is larger than the input size"

            h_start = random.randint(0, height - self.height)
            w_start = random.randint(0, width - self.width)

            rgb = TF.crop(rgb, h_start, w_start, self.height, self.width)
            depth = TF.crop(depth, h_start, w_start, self.height, self.width)
            gt = TF.crop(gt, h_start, w_start, self.height, self.width)

            K[2] = K[2] - w_start
            K[3] = K[3] - h_start

            rgb = TF.to_tensor(rgb)
            rgb = TF.normalize(rgb, (0.485, 0.456, 0.406),
                               (0.229, 0.224, 0.225), inplace=True)

            depth = TF.to_tensor(np.array(depth))

            gt = TF.to_tensor(np.array(gt))
        else: # test
            if self.args.top_crop > 0 and self.args.test_crop:
                width, height = rgb.size
                rgb = TF.crop(rgb, self.args.top_crop, 0,
                              height - self.args.top_crop, width)
                depth = TF.crop(depth, self.args.top_crop, 0,
                                height - self.args.top_crop, width)
                gt = TF.crop(gt, self.args.top_crop, 0,
                             height - self.args.top_crop, width)
                K[3] = K[3] - self.args.top_crop

            rgb = TF.to_tensor(rgb)
            rgb = TF.normalize(rgb, (0.485, 0.456, 0.406),
                               (0.229, 0.224, 0.225), inplace=True)

            depth = TF.to_tensor(np.array(depth))

            gt = TF.to_tensor(np.array(gt))


        output = {'rgb': rgb, 'dep': depth, 'gt': gt, 'K': torch.Tensor(K)}

        return output

    def _load_data(self, idx):
        path_rgb = os.path.join(self.args.dir_data,
                                self.sample_list[idx]['rgb'])
        path_depth = os.path.join(self.args.dir_data,
                                  self.sample_list[idx]['depth'])
        path_gt = os.path.join(self.args.dir_data,
                               self.sample_list[idx]['gt'])
        path_calib = os.path.join(self.args.dir_data,
                                  self.sample_list[idx]['K'])

        depth = read_depth(path_depth)
        gt = read_depth(path_gt)

        if self.mode in ['train', 'val']:
            calib = read_calib_file(path_calib)
            if 'image_02' in path_rgb:
                K_cam = np.reshape(calib['P_rect_02'], (3, 4))
            elif 'image_03' in path_rgb:
                K_cam = np.reshape(calib['P_rect_03'], (3, 4))
            K = [K_cam[0, 0], K_cam[1, 1], K_cam[0, 2], K_cam[1, 2]]
        else:
            f_calib = open(path_calib, 'r')
            K_cam = f_calib.readline().split(' ')
            f_calib.close()
            K = [float(K_cam[0]), float(K_cam[4]), float(K_cam[2]),
                 float(K_cam[5])]

        # modify here to mimic various lidar lines, such as [0, 1/16, 1/4, 1/1] -> [0, 4, 16, 64]Lines
        keep_ratio = (self.lidar_lines / 64.0)
        assert keep_ratio >=0 and keep_ratio <= 1.0, keep_ratio
        if keep_ratio >= 0.9999:
            pass
        elif keep_ratio > 0:
            Km = np.eye(3)
            Km[0, 0] = K[0]
            Km[1, 1] = K[1]
            Km[0, 2] = K[2]
            Km[1, 2] = K[3]
            depth = sample_lidar_lines(depth[:, :, None], intrinsics=Km, keep_ratio=keep_ratio)[:, :, 0]
        else:
            depth = np.zeros_like(depth)

        rgb = Image.open(path_rgb)
        depth = Image.fromarray(depth.astype('float32'), mode='F')
        gt = Image.fromarray(gt.astype('float32'), mode='F')

        w1, h1 = rgb.size
        w2, h2 = depth.size
        w3, h3 = gt.size

        assert w1 == w2 and w1 == w3 and h1 == h2 and h1 == h3

        return rgb, depth, gt, K

