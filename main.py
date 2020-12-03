from argparse import ArgumentParser

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

import models
import datasets


def main(hparams):
    model = models.__dict__[hparams.model](hparams)
    train_dataset = datasets.__dict__[hparams.dataset](hparams, hparams.train_data_root)
    val_dataset = datasets.__dict__[hparams.dataset](hparams, hparams.val_data_root, train=False)

    trainer = Trainer(
        max_epochs=800,
        gpus=hparams.gpus,
        benchmark=True,
        num_nodes=hparams.nodes,
        check_val_every_n_epoch=hparams.val,
        checkpoint_callback=ModelCheckpoint(
            filepath=hparams.checkpoint,
            verbose=True,
            prefix=hparams.name,
            period=hparams.val
        ),
        logger=[TensorBoardLogger('tb_logs', name=hparams.name)],
        callbacks=[LearningRateMonitor(logging_interval='step')],
        distributed_backend='ddp',
        precision=16 if hparams.use_amp else 32,
        sync_batchnorm=hparams.sync_bn,
        resume_from_checkpoint=hparams.resume
    )

    train_data_loader = DataLoader(train_dataset, batch_size=hparams.batch_size, pin_memory=True, num_workers=hparams.workers, shuffle=True)
    val_data_loader = DataLoader(val_dataset, batch_size=1, pin_memory=True, num_workers=hparams.workers, shuffle=False)

    if hparams.test:
        assert hparams.resume is not None
        trainer.test(model, val_data_loader)
    else:
        trainer.fit(model, train_data_loader, val_data_loader)


if __name__ == '__main__':

    model_names = sorted(name for name in models.__dict__
                         if not name.startswith("__") and 'Network' in name
                         and callable(models.__dict__[name]))
    dataset_names = sorted(name for name in datasets.__dict__
                           if not name.startswith("__") and 'Dataset' in name
                           and callable(datasets.__dict__[name]))

    parser = ArgumentParser(add_help=True)
    parser.add_argument('--name', required=True, type=str, help="name for experiments")
    parser.add_argument('--gpus', type=int, default=-1)
    parser.add_argument('--nodes', type=int, default=1)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=8, help="training batch size. Default=64")
    parser.add_argument("--threads", type=int, default=8, help="threads for data loader to use. Default=8")
    parser.add_argument("--decay_epoch", type=int, default=200, help="each epoch to decay lr. Default=200")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="learning rate. Default=1e-4")
    parser.add_argument("--val", type=int, default=1, help="check every n epoch to validation. Default=1")
    parser.add_argument("--resume", type=str, help="Path to checkpoint (default: none)")
    parser.add_argument('--use_amp', action='store_true', help='if true uses 16 bit precision')
    parser.add_argument('--sync_bn', action='store_true', help='if true uses sync batchnorm')

    parser.add_argument('--model', metavar='ARCH', required=True, choices=model_names,  help='model: ' + ' | '.join(model_names))
    parser.add_argument('--dataset', metavar='DATA', required=True, choices=dataset_names, help='dataset: ' + ' | '.join(dataset_names))
    parser.add_argument("--checkpoint", metavar='DIR', required=True, help="path to save checkpoints")
    parser.add_argument('--train_data_root', metavar='DIR', required=True, help="path to load train dataset")
    parser.add_argument('--val_data_root', metavar='DIR', required=True, help="path to load val dataset")

    parser.add_argument("--test", default=False, action="store_true")
    parser.add_argument("--output_root", default="", type=str, help="path to save results")

    temp_args, _ = parser.parse_known_args()

    parser = models.__dict__[temp_args.model].modify_commandline_options(parser)

    parser = datasets.__dict__[temp_args.dataset].modify_commandline_options(parser)

    hparams = parser.parse_args()
    main(hparams)
