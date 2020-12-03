# Install via: pip install pytorch-msssim
# Install via: pip install git+https://github.com/S-aiueo32/lpips-pytorch.git
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


def is_image_file(f):
    filename_lower = f.lower()
    return any(filename_lower.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.tif'])


def load_all_image(path):
    return [join(path, x) for x in listdir(path) if is_image_file(x)]


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-dir1', '--dir1', type=str, default='./imgs/ex_dir0')
    parser.add_argument('-dir2', '--dir2', type=str, default='./imgs/ex_dir0')
    parser.add_argument('--cuda', action='store_true', help='turn on flag to use GPU')
    opt = parser.parse_args()

    device = 'cuda' if opt.cuda else 'cpu'

    list_im1 = load_all_image(opt.dir1)
    list_im2 = load_all_image(opt.dir2)
    list_im1.sort()
    list_im2.sort()

    psnrs = []
    ssims = []
    msssims = []
    lpipss = []
    for im1_path, im2_path in tqdm(zip(list_im1, list_im2), total=len(list_im1)):
        # print(im1_path, im2_path)
        data, gt = Image.open(im1_path), Image.open(im2_path)
        data_th, gt_th = to_tensor(data).unsqueeze(0).to(device), to_tensor(gt).unsqueeze(0).to(device)

        data_th, gt_th = data_th[:, :, 32:-32, 32:-32], gt_th[:, :, 32:-32, 32:-32]
        # print(data_th.shape, gt_th.shape)
        # if data_th.size(-1) != gt_th.size(-1):
        #     gt_th = gt_th[:, :, :, :data_th.size(-1)]

        psnrs.append(-10 * math.log10((data_th - gt_th).pow(2).mean().item()))
        ssims.append(ssim(data_th, gt_th, data_range=1., size_average=False).item())
        msssims.append(ms_ssim(data_th, gt_th, data_range=1., size_average=False).item())
        lpipss.append(lpips(gt_th * 2 - 1, data_th * 2 - 1, net_type='alex', version='0.1').item())
    print("mean PSNR:", np.mean(psnrs))
    print("mean SSIM:", np.mean(ssims))
    print("mean MSSSIM:", np.mean(msssims))
    print("mean LPIPS:", np.mean(lpipss))
