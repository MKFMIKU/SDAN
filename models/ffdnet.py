# Modified from https://github.com/cszn/KAIR/blob/master/models/network_ffdnet.py

import os
import numpy as np
import torch.nn as nn
import models.basicblock as B
import torch
import pytorch_lightning as pl
import math
from models.networks import VGGLoss
from torchvision.utils import save_image

from models.hrnet import  _C as config, HighResolutionNet

"""
# --------------------------------------------
# FFDNet (15 or 12 conv layers)
# --------------------------------------------
Reference:
@article{zhang2018ffdnet,
  title={FFDNet: Toward a fast and flexible solution for CNN-based image denoising},
  author={Zhang, Kai and Zuo, Wangmeng and Zhang, Lei},
  journal={IEEE Transactions on Image Processing},
  volume={27},
  number={9},
  pages={4608--4622},
  year={2018},
  publisher={IEEE}
}
"""


# --------------------------------------------
# FFDNet
# --------------------------------------------
class FFDNetwork(pl.LightningModule):
    def modify_commandline_options(parser):
        parser.add_argument('--vgg', action='store_true', help='if true uses vgg')
        parser.add_argument('--hrnet', action='store_true', help='if true uses hrnet')
        parser.add_argument('--save_output', type=str, default=None, help='save the output images during testing')
        return parser

    def __init__(self, args):
        """
        # ------------------------------------
        in_nc: channel number of input
        out_nc: channel number of output
        nc: channel number
        nb: total number of conv layers
        act_mode: batch norm + activation function; 'BR' means BN+ReLU.
        # ------------------------------------
        # ------------------------------------
        """
        super(FFDNetwork, self).__init__()
        self.args = args

        in_nc = 3
        out_nc = 3
        nc = 64
        nb = 15
        act_mode = 'R'

        assert 'R' in act_mode or 'L' in act_mode, 'Examples of activation function: R, L, BR, BL, IR, IL'
        bias = True
        sf = 2

        self.m_down = B.PixelUnShuffle(upscale_factor=sf)

        m_head = B.conv(in_nc * sf * sf + 1, nc, mode='C' + act_mode[-1], bias=bias)
        m_body = [B.conv(nc, nc, mode='C' + act_mode, bias=bias) for _ in range(nb - 2)]
        m_tail = B.conv(nc, out_nc * sf * sf, mode='C', bias=bias)

        self.model = B.sequential(m_head, *m_body, m_tail)

        self.m_up = nn.PixelShuffle(upscale_factor=sf)

        self.criterion = nn.L1Loss()

        if self.args.vgg:
            self.vgg_loss = VGGLoss(weights=[0, 0, 0, 0, 1.0])

        if self.args.hrnet:
            config.defrost()
            config.merge_from_file('models/seg_hrnet_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484.yaml')
            self.hrnet = HighResolutionNet(config)
            self.hrnet.init_weights('pretrained/hrnet_w48_cityscapes_cls19_1024x2048_trainset.pth')


    def forward(self, x, sigma):
        h, w = x.size()[-2:]
        paddingBottom = int(np.ceil(h / 2) * 2 - h)
        paddingRight = int(np.ceil(w / 2) * 2 - w)
        x = torch.nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(x)

        x = self.m_down(x)
        # m = torch.ones(sigma.size()[0], sigma.size()[1], x.size()[-2], x.size()[-1]).type_as(x).mul(sigma)
        m = sigma.repeat(1, 1, x.size()[-2], x.size()[-1])
        x = torch.cat((x, m), 1)
        x = self.model(x)
        x = self.m_up(x)

        x = x[..., :h, :w]
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(list(self.m_down.parameters()) +
                                     list(self.model.parameters()) +
                                     list(self.m_up.parameters()), lr=self.args.learning_rate, betas=(0.9, 0.99))
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [80, 160, 240, 320, 400, 800], gamma=0.5)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        data = batch['data']
        label = batch['label']
        noise_level = batch['noise_level']

        output = self.forward(data, noise_level)

        loss = self.criterion(output, label)

        if self.args.vgg:
            vgg_loss = self.vgg_loss(output, label)
            loss += vgg_loss
        
        if self.args.hrnet:
            hrnet_fea_out = self.hrnet(output, feature=True)
            with torch.no_grad():
                hrnet_fea_label = self.hrnet(label, feature=True)
            hr_loss = self.criterion(hrnet_fea_out, hrnet_fea_label)
            loss += hr_loss

        self.log_dict({'train_loss': loss}, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        data = batch['data']
        label = batch['label']
        noise_level = batch['noise_level']

        with torch.no_grad():
            output = self.forward(data, noise_level)

            diff = output - label
            mse = diff.pow(2).mean()
            psnr = -10 * math.log10(mse)

        return {'psnr': psnr, 'data': data, 'label': label, 'output': output}

    def validation_epoch_end(self, outputs):
        data_img = torch.clamp(outputs[0]['data'], 0., 1.)
        label_img = torch.clamp(outputs[0]['label'], 0., 1.)
        output_img = torch.clamp(outputs[0]['output'], 0., 1.)

        self.logger[0].experiment.add_images('Validation/input', data_img, self.current_epoch)
        self.logger[0].experiment.add_images('Validation/label', label_img, self.current_epoch)
        self.logger[0].experiment.add_images('Validation/output', output_img, self.current_epoch)

        loss_val = np.mean([x['psnr'] for x in outputs])

        self.log_dict({'avg_psnr': loss_val}, prog_bar=True, logger=True, on_epoch=True)
        torch.cuda.empty_cache()

    def test_step(self, batch, batch_idx):
        data = batch['data']
        label = batch['label']
        noise_level = batch['noise_level']
        filename = batch['path'][0]

        with torch.no_grad():
            output = self.forward(data, noise_level)

        if self.args.save_output:
            save_image(output, os.path.join(self.args.save_output, 'output', os.path.basename(filename)))
            save_image(label, os.path.join(self.args.save_output, 'label', os.path.basename(filename)))
