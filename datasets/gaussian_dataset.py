import os
import glob
from torch.utils.data import Dataset
import cv2
import datasets.common as com
import random

class GaussianDataset(Dataset):
    @staticmethod
    def modify_commandline_options(parser):
        parser.add_argument('--crop_size', type=int, default=512, help="Crop to the width of crop_size")
        parser.add_argument('--aspect_ratio', type=float, default=1.0, help='The ratio height/width')
        parser.add_argument('--gaussian_size', type=int, default=21, help="kernel size of gaussian kernel")
        return parser

    def __init__(self, hparams, dataroot, val=False):
        self.hparams = hparams
        self.val = val
        self.image_paths = glob.glob(os.path.join(dataroot, '*.png'))

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)[:, :, ::-1]

        gaussian_kernel = random.choice([5, 7, 9, 11, 13, 15, 17, 19, 21])

        if not self.val:
            image = com.get_patch(image, patch_size_h=int(self.hparams.crop_size * self.hparams.aspect_ratio),
                                  patch_size_w=self.hparams.crop_size)[0]
            image = com.augment(image)[0]

        data = cv2.GaussianBlur(image, (gaussian_kernel, gaussian_kernel), 0)
        label = image

        data, label = com.np2Tensor(data, label)

        residual = label - data

        return {'data': data,
                'label': label,
                'residual': residual,
                'path': image_path}

    def __len__(self):
        return len(self.image_paths)
