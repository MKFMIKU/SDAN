import os
import random

import cv2
import numpy as np
import torch.utils.data as data
from PIL import Image

from datasets import common
from datasets import zoom_utils as utils


def augment(*args, hflip=True, wflip=True, rot=True):
    wflip = wflip and random.random() < 0.5
    hflip = hflip and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        if wflip: img = img[:, ::-1, :]
        if hflip: img = img[::-1, :, :]
        if rot90: img.transpose(1, 0, 2)
        return img

    return [_augment(a) for a in args]


class SRRawDataset(data.Dataset):
    @staticmethod
    def modify_commandline_options(parser):
        parser.add_argument('--scale', type=str, default='8', help='super resolution scale')
        parser.add_argument('--patch_size', type=int, default=512, help='output patch size')
        parser.add_argument('--rgb', action='store_true', help='if true uses rgb instead of raw in input')
        return parser

    def __init__(self, args, dataroot, train=True, benchmark=False):
        self.args = args
        self.scale = args.scale
        self.idx_scale = 0
        self.train = train
        self.benchmark = benchmark
        self.patch_size = args.patch_size
        self.rgb = args.rgb

        self.up_ratio = int(args.scale[0]) // 2
        self.dir_path = os.path.join(dataroot)

        dir_names = os.listdir(self.dir_path)
        dir_names.sort()
        self.file_names = []

        wrong_file = [
            '00350/00001.JPG',
            '00350/00002.JPG',
            '00350/00003.JPG',
            '00560/00003.JPG',
            '00267/00003.JPG',
        ]

        if self.up_ratio == 4:
            for dir_name in dir_names:
                d_path = os.path.join(self.dir_path, dir_name)
                for i in range(1, 4) if self.train else range(2, 3):
                    self.file_name = []
                    hr_path = os.path.join(d_path, "0000" + str(i) + '.JPG')
                    if hr_path[-15:] in wrong_file:
                        continue

                    hr_raw_path = os.path.join(d_path, "0000" + str(i) + ".ARW")
                    if not os.path.exists(hr_raw_path):
                        continue

                    lr_rgb_path = os.path.join(d_path, "0000" + str(i + 4) + ".JPG")
                    if not os.path.exists(lr_rgb_path):
                        continue

                    lr_raw_path = os.path.join(d_path, "0000" + str(i + 4) + ".ARW")
                    if not os.path.exists(lr_raw_path):
                        continue

                    ref_rgb_path = os.path.join(d_path, "00001" + ".JPG")

                    self.file_name.append(lr_raw_path)
                    self.file_name.append(hr_raw_path)
                    self.file_name.append(ref_rgb_path)
                    self.file_name.append(hr_path)
                    self.file_name.append(d_path)
                    self.file_name.append(lr_rgb_path)

                    self.file_names.append(self.file_name)
        elif self.up_ratio == 8:
            for dir_name in dir_names:
                d_path = os.path.join(self.dir_path, dir_name)
                for i in range(1, 2):
                    self.file_name = []
                    hr_path = os.path.join(d_path, "0000" + str(i) + '.JPG')
                    if hr_path[-15:] in wrong_file:
                        continue

                    lr_raw_path = os.path.join(d_path, "0000" + str(i + 5) + ".ARW")
                    if not os.path.exists(lr_raw_path):
                        continue

                    lr_rgb_path = os.path.join(d_path, "0000" + str(i + 5) + ".JPG")
                    if not os.path.exists(lr_raw_path):
                        continue

                    hr_raw_path = os.path.join(d_path, "0000" + str(i) + ".ARW")
                    if not os.path.exists(hr_raw_path):
                        continue

                    ref_rgb_path = os.path.join(d_path, "00001" + ".JPG")

                    self.file_name.append(lr_raw_path)
                    self.file_name.append(hr_raw_path)
                    self.file_name.append(ref_rgb_path)
                    self.file_name.append(hr_path)
                    self.file_name.append(d_path)
                    self.file_name.append(lr_rgb_path)

                    self.file_names.append(self.file_name)
        else:
            raise ValueError("arg.scale should be 4 or 8")

    def __getitem__(self, idx):
        up_ratio = self.scale
        file_name = self.file_names[idx]

        # Define the filenames
        path_raw = file_name[0]
        path2_raw = file_name[1]
        ref_rgb_path = file_name[2]
        tar_rgb_path = file_name[3]
        tform_txt = os.path.join(file_name[4], "tform.txt")
        wb_txt = os.path.join(file_name[4], "wb.txt")
        input_rgb_path = file_name[5]

        focal1 = utils.readFocal_pil(path_raw)
        focal2 = utils.readFocal_pil(path2_raw)
        focal_ref = utils.readFocal_pil(ref_rgb_path)

        ratio = focal2 / focal1
        ratio_ref1 = focal_ref / focal1
        ratio_ref2 = focal_ref / focal2
        ratio_offset = ratio / up_ratio[0]

        # read in input raw, will be used as input raw
        input_raw = utils.get_bayer(path_raw, black_lv=512, white_lv=16383)
        input_raw = utils.reshape_raw(input_raw)
        input_raw_orig = utils.crop_fov(input_raw, 1. / ratio_ref1)

        # read in target rgb image, will be used as ground truth
        tar_rgb = np.array(Image.open(tar_rgb_path))
        cropped_tar_rgb = utils.crop_fov(tar_rgb, 1. / ratio_ref2)
        tar_rgb_orig = utils.image_float(cropped_tar_rgb)

        input_rgb = np.array(Image.open(input_rgb_path))
        input_rgb_orig = utils.crop_fov(input_rgb, 1. / ratio_ref1)
        input_rgb_orig = utils.image_float(input_rgb_orig)

        # read in alignment matrices, will be used to align the target image to source
        # (because we don't want to transform or sample the raw sensor)
        src_tform, src_tform_corner = utils.read_tform(tform_txt, key=os.path.basename(path_raw).split('.')[0])
        tar_tform, _ = utils.read_tform(tform_txt, key=os.path.basename(path2_raw).split('.')[0])

        tform = utils.concat_tform([
            np.append(tar_tform, [[0, 0, 1]], 0),
            np.append(cv2.invertAffineTransform(src_tform), [[0, 0, 1]], 0)])

        tar_temp1 = cv2.resize(tar_rgb_orig, None, fx=ratio_ref2, fy=ratio_ref2, interpolation=cv2.INTER_CUBIC)
        tar_temp2, transformed_corner = utils.post_process_rgb(
            tar_temp1,
            (tar_temp1.shape[0], tar_temp1.shape[1]),
            tform[0:2, ...])
        tar_temp3 = cv2.resize(tar_temp2, None, fx=1 / ratio_ref2, fy=1 / ratio_ref2, interpolation=cv2.INTER_CUBIC)

        input_raw = input_raw_orig[
                    int(transformed_corner['minh'] / (2 * ratio_ref1)):int(
                        transformed_corner['maxh'] / (2 * ratio_ref1)),
                    int(transformed_corner['minw'] / (2 * ratio_ref1)):int(
                        transformed_corner['maxw'] / (2 * ratio_ref1)), :]

        input_rgb = input_rgb_orig[
                    int(transformed_corner['minh'] / (1 * ratio_ref1)):int(
                        transformed_corner['maxh'] / (1 * ratio_ref1)),
                    int(transformed_corner['minw'] / (1 * ratio_ref1)):int(
                        transformed_corner['maxw'] / (1 * ratio_ref1)), :]

        # we have to scale it by a ratio offset to match our desired zoom factor (e.g. 4X)
        tarc, tarw = input_raw.shape[0] * self.up_ratio * 2, input_raw.shape[1] * self.up_ratio * 2
        target_rgb = cv2.resize(tar_temp3, (tarw, tarc), interpolation=cv2.INTER_CUBIC)
        input_rgb = cv2.resize(input_rgb, (tarw // self.up_ratio, tarc // self.up_ratio), interpolation=cv2.INTER_CUBIC)

        out_wb = utils.read_wb(wb_txt, key=os.path.basename(tar_rgb_path).split('.')[0] + ":")
        target_rgb[..., 0] /= np.power(out_wb[0, 0], 1 / 2.2)
        target_rgb[..., 1] /= np.power(out_wb[0, 1], 1 / 2.2)
        target_rgb[..., 2] /= np.power(out_wb[0, 3], 1 / 2.2)
        target_rgb = np.clip(target_rgb, 0., 1.)

        out_wb = utils.read_wb(wb_txt, key=os.path.basename(input_rgb_path).split('.')[0] + ":")
        input_rgb[..., 0] /= np.power(out_wb[0, 0], 1 / 2.2)
        input_rgb[..., 1] /= np.power(out_wb[0, 1], 1 / 2.2)
        input_rgb[..., 2] /= np.power(out_wb[0, 3], 1 / 2.2)
        input_rgb = np.clip(input_rgb, 0., 1.)

        if self.train:
            if self.rgb:
                input_data, target_rgb = common.get_patch(input_rgb, target_rgb, patch_size=512, scale=self.up_ratio)
            else:
                tol = 32  # if we want to consider boundary issue (not computing loss on boundary pixels)
                raw_tol = int(tol / (self.up_ratio * 2))  # corresponding boundary for raw
                input_data, target_rgb = utils.crop_pair(
                    input_raw, target_rgb,
                    croph=self.patch_size, cropw=self.patch_size,
                    tol=tol, raw_tol=raw_tol,
                    ratio=self.up_ratio,
                    type='central',
                    fixx=0.5, fixy=0.5)
                input_data = np.expand_dims(utils.reshape_back_raw(input_data), axis=-1)

            input_data, target_rgb = augment(input_data, target_rgb)
            input_data, target_rgb = common.np2Tensor(input_data, target_rgb, rgb_range=1.0)

            return {'data': input_data,
                    'label': target_rgb,
                    'path': tar_rgb_path.split('/')[-2]}
        else:
            if self.rgb:
                input_data = input_rgb
            else:
                input_data = np.expand_dims(utils.reshape_back_raw(input_raw), axis=-1)

            # Patch for CPA, which enable the input has 64x dimensions
            h, w, _ = input_data.shape
            pad_h, pad_w = 32 - h % 32, 32 - w % 32
            input_data = np.pad(input_data,
                                pad_width=((pad_h // 2, (pad_h + 1) // 2), (pad_w // 2, (pad_w + 1) // 2), (0, 0)),
                                mode='constant', constant_values=0)
            input_rgb = np.pad(input_rgb,
                               pad_width=((pad_h // 2, (pad_h + 1) // 2), (pad_w // 2, (pad_w + 1) // 2), (0, 0)),
                               mode='constant', constant_values=0)
            target_rgb = np.pad(target_rgb, pad_width=((pad_h // 2 * self.up_ratio, (pad_h + 1) // 2 * self.up_ratio),
                                                       (pad_w // 2 * self.up_ratio, (pad_w + 1) // 2 * self.up_ratio),
                                                       (0, 0)),
                                mode='constant', constant_values=0)

            input_data, input_rgb, target_rgb = common.np2Tensor(input_data, input_rgb, target_rgb, rgb_range=1.0)

            out_wb = utils.read_wb(wb_txt, key=os.path.basename(tar_rgb_path).split('.')[0] + ":")

            return {'data': input_data,
                    'data_rgb': input_rgb,
                    'label': target_rgb,
                    'path': tar_rgb_path.split('/')[-2],
                    'out_wb': out_wb,
                    'pad_h': pad_h,
                    'pad_w': pad_w}

    def __len__(self):
        return len(self.file_names)
