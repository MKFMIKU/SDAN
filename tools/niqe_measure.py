import argparse
import numpy as np
import glob
from os import listdir
from os.path import join
from tqdm import tqdm
from PIL import Image
from pytorch_msssim import ssim, ms_ssim
from lpips_pytorch import lpips
from torchvision.transforms.functional import to_tensor
import math

import cv2
from scipy.ndimage.filters import convolve
from scipy.special import gamma

from basicsr.metrics.metric_util import reorder_image, to_y_channel
from basicsr.metrics.niqe import calculate_niqe

def is_image_file(f):
    filename_lower = f.lower()
    return any(filename_lower.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.tif'])


def load_all_image(path):
    return [join(path, x) for x in listdir(path) if is_image_file(x)]


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-dir1', '--dir1', type=str, default='./imgs/ex_dir0')
    opt = parser.parse_args()

    list_im1 = load_all_image(opt.dir1)

    niqes = []

    for im1_path in tqdm(list_im1, total=len(list_im1)):
        data = np.asarray(Image.open(im1_path))
        niqes.append(calculate_niqe(data, 32))
    
    print("mean NIQE:", np.mean(niqes))
