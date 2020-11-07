# !/usr/bin/env python
# By MKFMIKU mikumkf@gmail.com
import argparse
import numpy as np
from PIL import Image
from os import listdir
from cv2 import filter2D
from os.path import join
from tqdm import tqdm


def is_image_file(f):
    filename_lower = f.lower()
    return any(filename_lower.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.tif'])


def load_all_image(path):
    return [join(path, x) for x in listdir(path) if is_image_file(x)]


parser = argparse.ArgumentParser(description="PyTorch DeepDehazing")
parser.add_argument("--data", type=str, default="output", help="path to load data images")
parser.add_argument("--gt", type=str, help="path to load gt images")

opt = parser.parse_args()
print(opt)

datas = load_all_image(opt.data)
datas.sort()

labels = load_all_image(opt.gt)
labels.sort()


def output_psnr(img_orig, img_out):
    img_orig = img_orig / 255.
    img_out = img_out / 255.
    squared_error = np.square(img_orig - img_out)
    mse = np.mean(squared_error)
    psnr = 10 * np.log10(1.0 / mse)
    return psnr


def fspecial_gauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x, y = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]
    g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
    return g / g.sum()


def output_ssim(img1, img2, cs_map=False):
    K1 = 0.01
    K2 = 0.03
    L = 255
    size = 11
    sigma = 1.5
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    window = fspecial_gauss(size, sigma)
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2
    mu1 = filter2D(img1, -1, window)
    mu2 = filter2D(img2, -1, window)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = filter2D(img1 ** 2, -1, window) - mu1_sq
    sigma2_sq = filter2D(img2 ** 2, -1, window) - mu2_sq
    sigma12 = filter2D(img1 * img2, -1, window) - mu1_mu2
    if cs_map:
        return np.mean(
            ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))), \
               np.mean((2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2))
    else:
        return np.mean(
            ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)))


def output_msssim(img1, img2):
    level = 5
    weight = np.array([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
    downsample_filter = np.ones((2, 2)) / 4.0
    im1 = img1.astype(np.float64)
    im2 = img2.astype(np.float64)
    mssim = np.array([])
    mcs = np.array([])
    for l in range(level):
        ssim_map, cs_map = output_ssim(im1, im2, cs_map=True)
        mssim = np.append(mssim, ssim_map)
        mcs = np.append(mcs, cs_map)
        filtered_im1 = filter2D(im1, -1, downsample_filter)
        filtered_im2 = filter2D(im2, -1, downsample_filter)
        im1 = filtered_im1[::2, ::2]
        im2 = filtered_im2[::2, ::2]
    return np.mean(np.prod(mcs[0:level - 1] ** weight[0:level - 1]) * (mssim[level - 1] ** weight[level - 1]))


psnrs = []
ssims = []
msssims = []
for idx, data_p in enumerate(tqdm(datas)):
    data = Image.open(data_p)
    # gt = Image.open(join(opt.gt, data_p.split('/')[-1]))
    gt = Image.open(labels[idx])

    data = np.asarray(data).astype(float)
    gt = np.asarray(gt).astype(float)
    psnr = output_psnr(data, gt)
    ssim = output_ssim(data, gt)
    msssim = output_msssim(data, gt)
    psnrs.append(psnr)
    ssims.append(ssim)
    msssims.append(msssim)
print("mean PSNR:", np.mean(psnrs))
print("mean SSIM:", np.mean(ssims))
print("mean MSSSIM:", np.mean(msssims))
