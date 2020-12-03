import cv2
import numpy as np
from os import path as osp

from basicsr.metrics import calculate_psnr, calculate_ssim
from basicsr.utils import scandir
from basicsr.utils.matlab_functions import bgr2ycbcr

from torchvision.transforms.functional import to_tensor
from lpips_pytorch import lpips as calculate_lpips


def main():
    """Calculate PSNR and SSIM for images.

    Configurations:
        folder_gt (str): Path to gt (Ground-Truth).
        folder_restored (str): Path to restored images.
        crop_border (int): Crop border for each side.
        suffix (str): Suffix for restored images.
        test_y_channel (bool): If True, test Y channel (In MatLab YCbCr format)
            If False, test RGB channels.
    """
    # Configurations
    # -------------------------------------------------------------------------
    folder_gt = '/data/Experiments/DFS/sdcn_color_619_4x_rgb_sdcn_results/hr/'
    folder_restored = '/data/Experiments/DFS/sdcn_color_619_4x_rgb_sdcn_results/sr/'
    crop_border = 32
    suffix = ''
    test_y_channel = True
    # -------------------------------------------------------------------------

    psnr_all = []
    ssim_all = []
    lpips_all = []
    img_list = sorted(scandir(folder_gt, recursive=True, full_path=True))

    if test_y_channel:
        print('Testing Y channel.')
    else:
        print('Testing RGB channels.')

    for i, img_path in enumerate(img_list):
        basename, ext = osp.splitext(osp.basename(img_path))
        img_gt = cv2.imread(img_path, cv2.IMREAD_UNCHANGED).astype(
            np.float32) / 255.
        img_restored = cv2.imread(
            osp.join(folder_restored, basename + suffix + ext),
            cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.

        if test_y_channel and img_gt.ndim == 3 and img_gt.shape[2] == 3:
            img_gt = bgr2ycbcr(img_gt, y_only=True)
            img_restored = bgr2ycbcr(img_restored, y_only=True)

        # calculate PSNR and SSIM
        psnr = calculate_psnr(
            img_gt * 255,
            img_restored * 255,
            crop_border=crop_border,
            input_order='HWC')
        ssim = calculate_ssim(
            img_gt * 255,
            img_restored * 255,
            crop_border=crop_border,
            input_order='HWC')
        lpips = calculate_lpips(
            to_tensor(img_gt),
            to_tensor(img_restored),
            net_type='alex', version='0.1'
        ).item()

        print(f'{i+1:3d}: {basename:25}. \tPSNR: {psnr:.6f} dB, '
              f'\tSSIM: {ssim:.6f} \t LPIPS: {lpips:.6f}')
        psnr_all.append(psnr)
        ssim_all.append(ssim)
        lpips_all.append(lpips)
    print(f'Average: PSNR: {sum(psnr_all) / len(psnr_all):.6f} dB, '
          f'SSIM: {sum(ssim_all) / len(ssim_all):.6f}, '
          f'LPIPS: {sum(lpips_all) / len(lpips_all):.6f}')


if __name__ == '__main__':
    main()