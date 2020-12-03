import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from models.networks import BaseEncoder, BaseGenerator, MultiscaleDiscriminator, GANLoss, VGGLoss


class InverseVAENetwork(pl.LightningModule):
    @staticmethod
    def modify_commandline_options(parser):
        parser = BaseEncoder.modify_commandline_options(parser)
        parser = BaseGenerator.modify_commandline_options(parser)
        parser = MultiscaleDiscriminator.modify_commandline_options(parser)

        parser.add_argument('--lambda_content', type=float, default=1, help='weight for vgg loss')
        parser.add_argument('--lambda_gan', type=float, default=0.1, help='weight for vgg loss')
        parser.add_argument('--gan_mode', type=str, default='ls', help='(ls|original|hinge)')
        parser.add_argument('--rkld_loss', action='store_true', help='if specified, use reduced KLD loss')
        parser.add_argument('--lambda_rkld', type=float, default=0.1, help='weight for kld loss')
        parser.add_argument('--vgg_loss', action='store_true', help='if specified, use VGG feature matching loss')
        parser.add_argument('--lambda_vgg', type=float, default=10.0, help='weight for vgg loss')
        parser.add_argument('--low_loss', action='store_true', help='if specified, use loss in low frequency instead of high')
        return parser

    def __init__(self, hparams):
        super(InverseVAENetwork, self).__init__()
        self.hparams = hparams

        self.encoder = BaseEncoder(self.hparams)
        self.generator = BaseGenerator(self.hparams)
        self.discriminator = MultiscaleDiscriminator(self.hparams)

        self.criterionGAN = GANLoss(self.hparams.gan_mode, tensor=torch.cuda.FloatTensor)

        if self.hparams.vgg_loss:
            self.criterionVGG = VGGLoss(self.device)

    def configure_optimizers(self):
        lr = self.hparams.learning_rate

        opt_g = torch.optim.Adam(list(self.encoder.parameters()) + list(self.generator.parameters()), lr=lr,
                                 betas=(0.0, 0.9))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.0, 0.9))
        return [opt_g, opt_d], []

    def training_step(self, batch, batch_idx, optimizer_idx):
        G_losses = {}
        D_losses = {}

        data = batch['data']
        label = batch['label']

        # train generator
        if optimizer_idx == 0:
            # This is a reduced VAE implementation where we assume the outputs are multivariate Gaussian distribution
            # with mean = hiddens and std_dev = all ones.

            output, zs = self.fforward(data)

            if self.hparams.rkld_loss:
                rKLD_loss = torch.tensor([0.], device=data.device)
                for z in zs:
                    rKLD_loss += torch.mean(torch.pow(z, 2))
                G_losses['rKLD'] = rKLD_loss * self.hparams.lambda_rkld

            pred_fake, pred_real = self.discriminate(output, label)
            G_losses['GAN'] = self.criterionGAN(pred_fake, True, for_discriminator=False) * self.hparams.lambda_gan

            if self.hparams.vgg_loss:
                G_losses['VGG'] = self.criterionVGG(output, label) * self.hparams.lambda_vgg

            G_losses['Content'] = F.l1_loss(output, label) * self.hparams.lambda_content

            if self.hparams.ganFeat_loss:
                GAN_Feat_loss = torch.tensor([0.], device=data.device)
                num_D = len(pred_fake)
                for i in range(num_D):  # for each discriminator
                    # last output is the final prediction, so we exclude it
                    num_intermediate_outputs = len(pred_fake[i]) - 1
                    for j in range(num_intermediate_outputs):  # for each layer output
                        unweighted_loss = torch.nn.functional.l1_loss(
                            pred_fake[i][j], pred_real[i][j].detach())
                        GAN_Feat_loss += unweighted_loss * self.hparams.lambda_feat / num_D
                G_losses['GAN_Feat'] = GAN_Feat_loss * self.hparams.lambda_feat

            return {
                'loss': sum(G_losses.values()).mean(),
                'log': G_losses,
                'progress_bar': G_losses
            }

        # train discriminator
        if optimizer_idx == 1:
            with torch.no_grad():
                output, zs = self.fforward(data)
                output = output.detach()
                output.requires_grad_()
            pred_fake, pred_real = self.discriminate(output, label)
            D_losses['D_Fake'] = self.criterionGAN(pred_fake, False, for_discriminator=True)
            D_losses['D_real'] = self.criterionGAN(pred_real, True, for_discriminator=True)

            return {
                'loss': sum(D_losses.values()).mean(),
                'log': D_losses,
                'progress_bar': D_losses
            }

    def validation_step(self, batch, batch_idx):
        data = batch['data']
        label = batch['label']
        res = self.nm2(self.conv2(res))

        output, zs = self.fforward(data, no_noise=True)

        loss = torch.nn.functional.l1_loss(output, label)
        return {'val_loss': loss, 'output': output, 'data': data, 'label': label}

    def validation_epoch_end(self, outputs):
        data_img = torch.clamp(torch.cat([x['data'] for x in outputs], dim=0), -1., 1.)
        output_img = torch.clamp(torch.cat([x['output'] for x in outputs], dim=0), -1., 1.)
        label_img = torch.clamp(torch.cat([x['label'] for x in outputs], dim=0), -1., 1.)

        # Only show image in local logger
        self.logger[0].experiment.add_images('data', (data_img + 1) / 2, self.current_epoch)
        self.logger[0].experiment.add_images('output', (output_img + 1) / 2, self.current_epoch)
        self.logger[0].experiment.add_images('label', (label_img + 1) / 2, self.current_epoch)

        loss_val = torch.stack([x['val_loss'] for x in outputs]).mean()
        log_dict = {'val_loss': loss_val}
        return {'log': log_dict, 'val_loss': log_dict['val_loss'], 'progress_bar': log_dict}

    def discriminate(self, fake_image, real_image):
        # In Batch Normalization, the fake and real images are
        # recommended to be in the same batch to avoid disparate
        # statistics in fake and real images.
        # So both fake and real images are fed to D all at once.
        fake_and_real = torch.cat([fake_image, real_image], dim=0)

        discriminator_out = self.discriminator(fake_and_real)

        pred_fake, pred_real = self.divide_pred(discriminator_out)

        return pred_fake, pred_real

    def divide_pred(self, pred):
        # the prediction contains the intermediate outputs of multiscale GAN,
        # so it's usually a list
        if type(pred) == list:
            fake = []
            real = []
            for p in pred:
                fake.append([tensor[:tensor.size(0) // 2] for tensor in p])
                real.append([tensor[tensor.size(0) // 2:] for tensor in p])
        else:
            fake = pred[:pred.size(0) // 2]
            real = pred[pred.size(0) // 2:]

        return fake, real

    def fforward(self, data, no_noise=False):
        zs = self.encoder(data)
        new_zs = []
        if not no_noise:
            for z in zs:
                new_zs.append(z + torch.randn(z.size(), device=z.device))
        else:
            new_zs = zs

        if not self.hparams.low_loss:
            output = self.generator(new_zs) + data
        else:
            output = self.generator(new_zs)
        return output, zs
