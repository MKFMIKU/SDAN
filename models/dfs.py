import math
import os

import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
from torchvision.utils import save_image

from models import common

from models.losses import charbonnier_loss

# Follow the color wheel from https://vision.middlebury.edu/flow/flowEval-iccv07.pdf
from flow_vis import flow_to_color


class DFSNetwork(pl.LightningModule):
    def modify_commandline_options(parser):
        parser.add_argument('--n_resblocks', type=int, default=16, help='number of residual blocks')
        parser.add_argument('--n_feats', type=int, default=64, help='number of feature maps')
        parser.add_argument('--n_colors', type=int, default=1, help='number of color channels to use')
        parser.add_argument('--norm', type=str, default='group', help='which norm to be use')
        parser.add_argument('--res_scale', type=float, default=1, help='residual scaling')
        parser.add_argument('--ps', default=True, help='use pixel shuffle in upsample')
        parser.add_argument('--save_output', type=str, default=None, help='save the output images during testing')
        parser.add_argument('--lam', type=float, default=150, help='lambda used in smooth')
        parser.add_argument('--sdcn', action='store_true', help='if true uses sdcn')
        return parser

    def __init__(self, args, conv=common.default_conv):
        super(DFSNetwork, self).__init__()
        self.args = args
        self.lam = args.lam
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3

        self.sdcn = args.sdcn

        # define head module
        m_head = [nn.Conv2d(args.n_colors, n_feats, 3, 1, 1),
                  nn.ReLU(inplace=True)]

        # define body module
        m_body = [
            common.ClassicalResINBlock(
                conv, n_feats, kernel_size, res_scale=args.res_scale, bias=False, nm=args.norm
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, 3, bias=False))

        # define tail module
        m_tail = [
            common.Upsampler(conv, args.scale[0] // 2, n_feats, act='relu', bn=False, ps=args.ps),
            nn.Conv2d(n_feats, 3, 3, 1, 1)
        ]

        self.head = nn.Sequential(*m_head)
        self.af_head = common.Sq_DFS_Attention(n_feats, n_feats, 3, bias=True)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):

        x, x_ref = x

        x = self.head(x)

        if self.sdcn:
            x, rgb, mask, offset = self.af_head(x, x_ref)

        res = x
        res = self.body(res)
        x = res + x

        x = self.tail(x)

        if self.sdcn:
            return x, rgb, mask, offset
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(list(self.head.parameters()) +
                                     list(self.af_head.parameters()) +
                                     list(self.body.parameters()) +
                                     list(self.tail.parameters()), lr=self.args.learning_rate, betas=(0.9, 0.99))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2, eta_min=1e-5)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        lr = batch['data']
        hr = batch['label']

        with torch.no_grad():
            lr_ref = nn.functional.interpolate(hr, size=lr.shape[-2:], mode='bicubic', align_corners=False)

        if self.sdcn:
            sr, tmp_sr, mask, offset = self.forward([lr, lr_ref])

            sr, hr = sr[:, :, 32:-32, 32:-32], \
                       hr[:, :, 32:-32, 32:-32]

            padding_shave = 32 // (self.args.scale[0] // 2)
            lr, lr_ref, tmp_sr, mask, offset = lr[:, :, padding_shave:-padding_shave, padding_shave:-padding_shave], \
                                               lr_ref[:, :, padding_shave:-padding_shave, padding_shave:-padding_shave], \
                                               tmp_sr[:, :, padding_shave:-padding_shave, padding_shave:-padding_shave], \
                                               mask[:, :, padding_shave:-padding_shave, padding_shave:-padding_shave], \
                                               offset[:, :, padding_shave:-padding_shave, padding_shave:-padding_shave]
            align_loss = torch.mean(charbonnier_loss(tmp_sr, lr_ref) * mask)

            mask = nn.functional.interpolate(mask, sr.shape[2:])
            sr_loss = torch.mean(charbonnier_loss(sr, hr) * mask)

            loss = sr_loss + align_loss
        else:
            sr = self.forward([lr, lr_ref])
            loss = nn.functional.l1_loss(sr, hr)

        if batch_idx % 100 == 0:
            self.logger[0].experiment.add_images('output/Training', sr, self.current_epoch)
            self.logger[0].experiment.add_images('lr_ref/Training', lr_ref, self.current_epoch)
            self.logger[0].experiment.add_images('label/Training', hr, self.current_epoch)
            self.logger[0].experiment.add_images('input/Training', lr, self.current_epoch)

            if self.sdcn:
                self.logger[0].experiment.add_scalar('sr_loss/Training', sr_loss, self.current_epoch)
                self.logger[0].experiment.add_scalar('align_loss/Training', align_loss, self.current_epoch)
                self.logger[0].experiment.add_images('tmp_output/Training', tmp_sr, self.current_epoch)
                self.logger[0].experiment.add_images('mask/Training', mask, self.current_epoch)
                flow = np.expand_dims(flow_to_color(offset[0].detach().cpu().numpy().transpose((1, 2, 0))).transpose(2, 0, 1), 0)
                self.logger[0].experiment.add_images('flow/Training', flow, self.current_epoch)

        self.log_dict({'train_loss': loss}, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        lr = batch['data']
        lr_y = batch['data_rgb']
        hr_y = batch['label']
        pad_h = batch['pad_h'][0]
        pad_w = batch['pad_w'][0]

        with torch.no_grad():
            lr_ref = nn.functional.interpolate(lr_y, size=lr.shape[-2:], mode='bicubic', align_corners=False)

            if self.sdcn:
                sr, rgb, mask, offset = self.forward([lr, lr_ref])
            else:
                sr = self.forward([lr, lr_ref])

            # Patch for CPA, which enable the input has 64x dimensions
            sr = sr[:, :, pad_h // 4 * self.args.scale[0]:-(pad_h + 1) // 4 * self.args.scale[0],
                 pad_w // 4 * self.args.scale[0]:-(pad_w + 1) // 4 * self.args.scale[0]]

            hr_y = hr_y[:, :, pad_h // 4 * self.args.scale[0]:-(pad_h + 1) // 4 * self.args.scale[0],
                   pad_w // 4 * self.args.scale[0]:-(pad_w + 1) // 4 * self.args.scale[0]]

            sr, hr_y = sr[:, :, 32:-32, 32:-32], hr_y[:, :, 32:-32, 32:-32]

            diff = (sr - hr_y) / 1.0
            mse = diff.pow(2).mean()
            loss = -10 * math.log10(mse)

        return {'psnr': loss, 'data': lr, 'data_ref': lr_ref, 'label': hr_y, 'output': sr}

    def validation_epoch_end(self, outputs):
        label_img = torch.clamp(outputs[0]['label'], 0., 1.)
        output_img = torch.clamp(outputs[0]['output'], 0., 1.)
        data_img = torch.clamp(outputs[0]['data'], 0., 1.)
        data_ref_img = torch.clamp(outputs[0]['data_ref'], 0., 1.)

        self.logger[0].experiment.add_images('label/Validation', label_img, self.current_epoch)
        self.logger[0].experiment.add_images('output/Validation', output_img, self.current_epoch)
        self.logger[0].experiment.add_images('input/Validation', data_img, self.current_epoch)
        self.logger[0].experiment.add_images('input_ref/Validation', data_ref_img, self.current_epoch)

        loss_val = np.mean([x['psnr'] for x in outputs])

        self.log_dict({'avg_psnr': loss_val}, prog_bar=True, logger=True, on_epoch=True)
        torch.cuda.empty_cache()

    def test_step(self, batch, batch_idx):
        lr = batch['data']
        lr_ref = batch['data_rgb']
        hr = batch['label']
        pad_h = batch['pad_h'][0]
        pad_w = batch['pad_w'][0]
        filename = batch['path'][0]
        out_wb = batch['out_wb'][0]

        with torch.no_grad():
            lr_ref = nn.functional.interpolate(hr, size=lr.shape[-2:], mode='bicubic', align_corners=False)

            if self.sdcn:
                sr, rgb, mask, offset = self.forward([lr, lr_ref])
            else:
                sr = self.forward([lr, lr_ref])

            # Patch for CPA, which enable the input has 64x dimensions
            sr = sr[:, :, pad_h * self.args.scale[0] // 4:-(pad_h + 1) * self.args.scale[0] // 4,
                 pad_w * self.args.scale[0] // 4:-(pad_w + 1) * self.args.scale[0] // 4]

            hr = hr[:, :, pad_h * self.args.scale[0] // 4:-(pad_h + 1) * self.args.scale[0] // 4,
                 pad_w * self.args.scale[0] // 4:-(pad_w + 1) * self.args.scale[0] // 4]

            sr, hr = sr[:, :, 32:-32, 32:-32], hr[:, :, 32:-32, 32:-32]

            # sr[:, 0] *= torch.pow(out_wb[0, 0], 1 / 2.2)
            # sr[:, 1] *= torch.pow(out_wb[0, 1], 1 / 2.2)
            # sr[:, 2] *= torch.pow(out_wb[0, 3], 1 / 2.2)

            hr[:, 0] *= torch.pow(out_wb[0, 0], 1 / 2.2)
            hr[:, 1] *= torch.pow(out_wb[0, 1], 1 / 2.2)
            hr[:, 2] *= torch.pow(out_wb[0, 3], 1 / 2.2)

            diff = (sr - hr) / 1.0
            shave = self.args.scale[0] + 8
            valid = diff[..., shave:-shave, shave:-shave]
            mse = valid.pow(2).mean()
            loss = -10 * math.log10(mse)

            if self.args.save_output:
                # sr = torch.cat([hr, sr], dim=2)
                save_image(sr, os.path.join(self.args.save_output, 'sr', os.path.basename(filename) + '.png'))
                save_image(hr, os.path.join(self.args.save_output, 'hr', os.path.basename(filename) + '.png'))
            return loss

    def test_epoch_end(self, outputs):
        psnrs = []
        for psnr in outputs:
            psnrs.append(psnr)
        print("Final psnr:", np.mean(psnrs))

    @staticmethod
    def reshape_back_raw_torch(bayer):
        N, C, H, W = bayer.shape
        newH = int(H * 2)
        newW = int(W * 2)
        bayer_back = torch.zeros((N, 1, newH, newW))
        bayer_back[:, :, 0:newH:2, 0:newW:2] = bayer[:, 0]
        bayer_back[:, :, 0:newH:2, 1:newW:2] = bayer[:, 1]
        bayer_back[:, :, 1:newH:2, 1:newW:2] = bayer[:, 2]
        bayer_back[:, :, 1:newH:2, 0:newW:2] = bayer[:, 3]
        return bayer_back

    @staticmethod
    def apply_gamma_torch(image, gamma=2.22):
        return image ** (1. / gamma)
