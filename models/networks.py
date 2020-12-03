import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from models.activation_norm import get_activation_norm_layer
from models.base_network import BaseNetwork

import math
from torch.nn import init

# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0, tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_tensor = None
        self.fake_label_tensor = None
        self.zero_tensor = None
        self.Tensor = tensor
        self.gan_mode = gan_mode
        if gan_mode == 'ls':
            pass
        elif gan_mode == 'original':
            pass
        elif gan_mode == 'w':
            pass
        elif gan_mode == 'hinge':
            pass
        else:
            raise ValueError('Unexpected gan_mode {}'.format(gan_mode))

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            if self.real_label_tensor is None:
                self.real_label_tensor = self.Tensor(1).fill_(self.real_label).cuda()
                self.real_label_tensor.requires_grad_(False)
            return self.real_label_tensor.expand_as(input)
        else:
            if self.fake_label_tensor is None:
                self.fake_label_tensor = self.Tensor(1).fill_(self.fake_label).cuda()
                self.fake_label_tensor.requires_grad_(False)
            return self.fake_label_tensor.expand_as(input)

    def get_zero_tensor(self, input):
        if self.zero_tensor is None:
            self.zero_tensor = self.Tensor(1).fill_(0).cuda()
            self.zero_tensor.requires_grad_(False)
        return self.zero_tensor.expand_as(input)

    def loss(self, input, target_is_real, for_discriminator=True):
        if self.gan_mode == 'original':  # cross entropy loss
            target_tensor = self.get_target_tensor(input, target_is_real)
            loss = F.binary_cross_entropy_with_logits(input, target_tensor)
            return loss
        elif self.gan_mode == 'ls':
            target_tensor = self.get_target_tensor(input, target_is_real)
            return F.mse_loss(input, target_tensor)
        elif self.gan_mode == 'hinge':
            if for_discriminator:
                if target_is_real:
                    minval = torch.min(input - 1, self.get_zero_tensor(input))
                    loss = -torch.mean(minval)
                else:
                    minval = torch.min(-input - 1, self.get_zero_tensor(input))
                    loss = -torch.mean(minval)
            else:
                assert target_is_real, "The generator's hinge loss must be aiming for real"
                loss = -torch.mean(input)
            return loss
        else:
            # wgan
            if target_is_real:
                return -input.mean()
            else:
                return input.mean()

    def __call__(self, input, target_is_real, for_discriminator=True):
        # computing loss is a bit complicated because |input| may not be
        # a tensor, but list of tensors in case of multiscale discriminator
        if isinstance(input, list):
            loss = 0
            for pred_i in input:
                if isinstance(pred_i, list):
                    pred_i = pred_i[-1]
                loss_tensor = self.loss(pred_i, target_is_real, for_discriminator)
                bs = 1 if len(loss_tensor.size()) == 0 else loss_tensor.size(0)
                new_loss = torch.mean(loss_tensor.view(bs, -1), dim=1)
                loss += new_loss
            return loss / len(input)
        else:
            return self.loss(input, target_is_real, for_discriminator)


# VGG architecter, used for the perceptual loss using a pretrained VGG network
class VGG19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = torchvision.models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


# Perceptual loss that uses a pretrained VGG network
class VGGLoss(nn.Module):
    def __init__(self, weights=[1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0 / 2, 1.0]):
        super(VGGLoss, self).__init__()
        self.vgg = VGG19()
        self.criterion = nn.L1Loss()
        self.weights = weights

    def forward(self, x, y):
        x_vgg = self.vgg(x)
        with torch.no_grad():
            y_vgg = self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss


class SpaceToDepth(nn.Module):
    def __init__(self, block_size):
        super(SpaceToDepth, self).__init__()
        self.block_size = block_size
        self.block_size_sq = block_size * block_size

    def forward(self, input):
        output = input.permute(0, 2, 3, 1)
        (batch_size, s_height, s_width, s_depth) = output.size()
        d_depth = s_depth * self.block_size_sq
        d_width = int(s_width / self.block_size)
        d_height = int(s_height / self.block_size)
        t_1 = output.split(self.block_size, 2)
        stack = [t_t.reshape(batch_size, d_height, d_depth) for t_t in t_1]
        output = torch.stack(stack, 1)
        output = output.permute(0, 2, 1, 3)
        output = output.permute(0, 3, 1, 2)
        return output


class DepthToSpace(nn.Module):
    def __init__(self, block_size):
        super(DepthToSpace, self).__init__()
        self.block_size = block_size
        self.block_size_sq = block_size * block_size

    def forward(self, input):
        output = input.permute(0, 2, 3, 1)
        (batch_size, d_height, d_width, d_depth) = output.size()
        s_depth = int(d_depth / self.block_size_sq)
        s_width = int(d_width * self.block_size)
        s_height = int(d_height * self.block_size)
        t_1 = output.reshape(batch_size, d_height, d_width, self.block_size_sq, s_depth)
        spl = t_1.split(self.block_size, 3)
        stack = [t_t.reshape(batch_size, d_height, s_width, s_depth) for t_t in spl]
        output = torch.stack(stack, 0).transpose(0, 1).permute(0, 2, 1, 3, 4).reshape(batch_size, s_height, s_width,
                                                                                      s_depth)
        output = output.permute(0, 3, 1, 2)
        return output


class PatchAttn(nn.Module):
    """" Patch Attention Module """

    def __init__(self, in_dim, block_size):
        super(PatchAttn, self).__init__()
        self.chanel_in = in_dim
        self.block_size = block_size
        self.block_size_sq = block_size * block_size

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

        self.s2d = SpaceToDepth(block_size)
        self.d2s = DepthToSpace(block_size)

    def forward(self, x):
        x = self.s2d(x).contiguous()
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x

        out = self.d2s(out).contiguous()

        return out


class ResBlock(nn.Module):
    def __init__(self, n_feats, kernel_size):
        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(nn.Conv2d(n_feats, n_feats, kernel_size, stride=1, padding=kernel_size // 2))
            m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(nn.LeakyReLU(0.2, inplace=True))
        self.body = nn.Sequential(*m)

    def forward(self, x):
        res = self.body(x)
        return res + x


# Such architecture is borrowed from self2self
class BaseEncoder(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser):
        parser.add_argument('--nef', type=int, default=128, help='number of features in encoder')
        parser.add_argument('--enorm', type=str, default='instance', help='type of encoder activation normalization.')
        return parser

    def __init__(self, hparams):
        super(BaseEncoder, self).__init__()
        self.hparams = hparams
        nf = hparams.nef
        bias = hparams.enorm == 'instance'

        actvn = nn.LeakyReLU(0.2, inplace=True)

        self.conv_img = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, nf, 7, stride=1, padding=0, bias=bias),
            actvn,
            nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1, bias=bias),
        )

        self.down_0 = nn.Sequential(
            get_activation_norm_layer(nf, hparams.enorm, 2),
            actvn,
            nn.Conv2d(nf, nf, kernel_size=3, stride=2, padding=1, bias=bias),
            get_activation_norm_layer(nf, hparams.enorm, 2),
            actvn,
            PatchAttn(nf, 4),
            nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1, bias=bias),
        )

        self.down_1 = nn.Sequential(
            get_activation_norm_layer(nf, hparams.enorm, 2),
            actvn,
            nn.Conv2d(nf, nf, kernel_size=3, stride=2, padding=1, bias=bias),
            get_activation_norm_layer(nf, hparams.enorm, 2),
            actvn,
            PatchAttn(nf, 4),
            nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1, bias=bias),
        )

        self.down_2 = nn.Sequential(
            get_activation_norm_layer(nf, hparams.enorm, 2),
            actvn,
            nn.Conv2d(nf, nf, kernel_size=3, stride=2, padding=1, bias=bias),
            get_activation_norm_layer(nf, hparams.enorm, 2),
            actvn,
            PatchAttn(nf, 4),
            nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1, bias=bias),
        )

        self.down_3 = nn.Sequential(
            get_activation_norm_layer(nf, hparams.enorm, 2),
            actvn,
            nn.Conv2d(nf, nf, kernel_size=3, stride=2, padding=1, bias=bias),
            get_activation_norm_layer(nf, hparams.enorm, 2),
            actvn,
            PatchAttn(nf, 4),
            nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1, bias=bias),
        )

        self.down_4 = nn.Sequential(
            get_activation_norm_layer(nf, hparams.enorm, 2),
            actvn,
            nn.Conv2d(nf, nf, kernel_size=3, stride=2, padding=1, bias=bias),
            get_activation_norm_layer(nf, hparams.enorm, 2),
            actvn,
            PatchAttn(nf, 4),
            nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1, bias=bias),
        )

        self.down_5 = nn.Sequential(
            get_activation_norm_layer(nf, hparams.enorm, 2),
            actvn,
            nn.Conv2d(nf, nf, kernel_size=3, stride=2, padding=1, bias=bias),
            get_activation_norm_layer(nf, hparams.enorm, 2),
            actvn,
            PatchAttn(nf, 4),
            nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1, bias=bias),
        )

    def forward(self, x):
        keys = []

        x = self.conv_img(x)

        x = self.down_0(x)
        keys.append(x)

        x = self.down_1(x)
        keys.append(x)

        x = self.down_2(x)
        keys.append(x)

        x = self.down_3(x)
        keys.append(x)

        x = self.down_4(x)
        keys.append(x)

        x = self.down_5(x)
        keys.append(x)

        return keys


class BaseGenerator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser):
        parser.add_argument('--ngf', type=int, default=128, help='number of features in generator')
        parser.add_argument('--gnorm', type=str, default='instance', help='type of generator activation normalization.')
        return parser

    def __init__(self, hparams):
        super(BaseGenerator, self).__init__()
        self.hparams = hparams
        nf = hparams.ngf
        bias = hparams.gnorm == 'instance'

        actvn = nn.LeakyReLU(0.2, inplace=True)

        self.up_0 = nn.Sequential(
            get_activation_norm_layer(nf, hparams.gnorm, 2),
            actvn,
            nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1, bias=bias),
            get_activation_norm_layer(nf, hparams.gnorm, 2),
            actvn,
            nn.ConvTranspose2d(nf, nf, kernel_size=3, stride=2, padding=1, output_padding=1, bias=bias),
        )

        self.up_1 = nn.Sequential(
            get_activation_norm_layer(nf * 2, hparams.gnorm, 2),
            actvn,
            nn.Conv2d(nf * 2, nf, kernel_size=3, stride=1, padding=1, bias=bias),
            get_activation_norm_layer(nf, hparams.gnorm, 2),
            actvn,
            nn.ConvTranspose2d(nf, nf, kernel_size=3, stride=2, padding=1, output_padding=1, bias=bias),
        )

        self.up_2 = nn.Sequential(
            get_activation_norm_layer(nf * 2, hparams.gnorm, 2),
            actvn,
            nn.Conv2d(nf * 2, nf, kernel_size=3, stride=1, padding=1, bias=bias),
            get_activation_norm_layer(nf, hparams.gnorm, 2),
            actvn,
            nn.ConvTranspose2d(nf, nf, kernel_size=3, stride=2, padding=1, output_padding=1, bias=bias),
        )

        self.up_3 = nn.Sequential(
            get_activation_norm_layer(nf * 2, hparams.gnorm, 2),
            actvn,
            nn.Conv2d(nf * 2, nf, kernel_size=3, stride=1, padding=1, bias=bias),
            get_activation_norm_layer(nf, hparams.gnorm, 2),
            actvn,
            nn.ConvTranspose2d(nf, nf, kernel_size=3, stride=2, padding=1, output_padding=1, bias=bias),
        )

        self.up_4 = nn.Sequential(
            get_activation_norm_layer(nf * 2, hparams.gnorm, 2),
            actvn,
            nn.Conv2d(nf * 2, nf, kernel_size=3, stride=1, padding=1, bias=bias),
            get_activation_norm_layer(nf, hparams.gnorm, 2),
            actvn,
            nn.ConvTranspose2d(nf, nf, kernel_size=3, stride=2, padding=1, output_padding=1, bias=bias),
        )

        self.up_5 = nn.Sequential(
            get_activation_norm_layer(nf * 2, hparams.gnorm, 2),
            actvn,
            nn.Conv2d(nf * 2, nf, kernel_size=3, stride=1, padding=1, bias=bias),
            get_activation_norm_layer(nf, hparams.gnorm, 2),
            actvn,
            nn.ConvTranspose2d(nf, nf, kernel_size=3, stride=2, padding=1, output_padding=1, bias=bias),
        )

        self.conv_img = nn.Sequential(
            get_activation_norm_layer(nf, hparams.gnorm, 2),
            actvn,
            nn.Conv2d(nf, 64, kernel_size=3, stride=1, padding=1),
            actvn,
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            actvn,
            nn.ReflectionPad2d(3),
            nn.Conv2d(32, 3, 7, padding=0),
        )

    def forward(self, zs):
        x = self.up_0(zs[-1])
        x = self.up_1(torch.cat([x, zs[-2]], dim=1))
        x = self.up_2(torch.cat([x, zs[-3]], dim=1))
        x = self.up_3(torch.cat([x, zs[-4]], dim=1))
        x = self.up_4(torch.cat([x, zs[-5]], dim=1))
        x = self.up_5(torch.cat([x, zs[-6]], dim=1))
        x = self.conv_img(x)
        return F.tanh(x)


class MultiscaleDiscriminator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser):
        parser.add_argument('--num_D', type=int, default=3, help='number of discriminators to be used in multiscale')
        parser = NLayerDiscriminator.modify_commandline_options(parser)
        return parser

    def __init__(self, hparams):
        super(MultiscaleDiscriminator, self).__init__()
        self.hparams = hparams

        for i in range(hparams.num_D):
            subnetD = NLayerDiscriminator(hparams)
            self.add_module('discriminator_%d' % i, subnetD)

    def downsample(self, input):
        return F.avg_pool2d(input, kernel_size=3,
                            stride=2, padding=[1, 1],
                            count_include_pad=False)

    # Returns list of lists of discriminator outputs.
    # The final result is of size opt.num_D x opt.n_layers_D
    def forward(self, input):
        result = []
        get_intermediate_features = self.hparams.ganFeat_loss
        for name, D in self.named_children():
            out = D(input)
            if not get_intermediate_features:
                out = [out]
            result.append(out)
            input = self.downsample(input)

        return result


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    @staticmethod
    def modify_commandline_options(parser):
        parser.add_argument('--n_layers_D', type=int, default=4, help='# layers in each discriminator')
        parser.add_argument('--ndf', type=int, default=64, help='number of features in discriminator')
        parser.add_argument('--ganFeat_loss', action='store_true', help='use discriminator feature matching loss')
        parser.add_argument('--lambda_feat', type=float, default=10.0, help='weight for feature matching loss')
        return parser

    def __init__(self, hparams):
        super(NLayerDiscriminator, self).__init__()
        self.hparams = hparams

        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        nf = hparams.ndf
        input_nc = 3

        sequence = [[nn.Conv2d(input_nc, nf, kernel_size=kw, stride=2, padding=padw),
                     nn.LeakyReLU(0.2, False)]]

        for n in range(1, hparams.n_layers_D):
            nf_prev = nf
            nf = min(nf * 2, 512)
            stride = 1 if n == hparams.n_layers_D - 1 else 2
            sequence += [[nn.InstanceNorm2d(nf, affine=False),
                          nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=stride, padding=padw),
                          nn.LeakyReLU(0.2, False)
                          ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        # We divide the layers into groups to extract intermediate layer outputs
        for n in range(len(sequence)):
            self.add_module('model' + str(n), nn.Sequential(*sequence[n]))

    def forward(self, input):
        results = [input]
        for submodel in self.children():
            intermediate_output = submodel(results[-1])
            results.append(intermediate_output)

        get_intermediate_features = self.hparams.ganFeat_loss
        if get_intermediate_features:
            return results[1:]
        else:
            return results[-1]
