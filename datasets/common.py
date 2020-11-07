import random

import numpy as np
import torch
from torchvision.transforms.functional import normalize


def augment(*args, hflip=True, rot=True):
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        if hflip: img = img[:, ::-1, :]
        if vflip: img = img[::-1, :, :]
        if rot90: img = img.transpose(1, 0, 2)

        return img

    return [_augment(a) for a in args]


def get_patch(*args, patch_size_h, patch_size_w):
    ih, iw = args[0].shape[:2]
    ix = random.randrange(0, iw - patch_size_w + 1)
    iy = random.randrange(0, ih - patch_size_h + 1)
    return [a[iy:iy + patch_size_h, ix:ix + patch_size_w, :] for a in args]


def np2Tensor(*args):
    def _np2Tensor(img):
        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
        tensor = torch.from_numpy(np_transpose).float()
        tensor.mul_(1 / 255)
        tensor = normalize(tensor, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        return tensor

    return [_np2Tensor(a) for a in args]
