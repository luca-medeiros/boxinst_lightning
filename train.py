import os
import wandb
import warnings
import pytorch_lightning as pl

from argparse import ArgumentParser
from pytorch_lightning.loggers import WandbLogger

from evaluation import callbacks
from trainer import Trainer
from config import cfg
from data.dataset import DataModule


warnings.filterwarnings("ignore")


def main(args):
    cfg.data.batch_size = cfg.data.batch_size // len(args.gpus)
    pl.utilities.seed.seed_everything(seed=42)
    data_module = DataModule(cfg)
    data_module.setup()
    cfg.trainer.val_check_interval = len(data_module.train) // (len(args.gpus) * cfg.data.batch_size)
    project, _ = cfg.instance.split('/')
    instance = cfg.model.backbone.name
    cfg.directory = f'./output/{project}--{instance}'
    if not os.path.isdir(cfg.directory):
        os.makedirs(cfg.directory)

    if cfg.resume != '':
        print('Resuming...')
        trainer_model = Trainer.load_from_checkpoint(cfg.resume)
    else:
        trainer_model = Trainer(cfg)

    # ------------
    # training
    # ------------
    job_type = 'train' if not args.eval else 'eval'
    wandb_logger = WandbLogger(project=project,
                               name=instance,
                               job_type=job_type,
                               offline=args.wandb_offline)
    accelerator = None

    if len(args.gpus) > 1:
        accelerator = 'ddp'

    callbacks_list = [callbacks.progress_bar,
                       callbacks.model_checkpoint(cfg),
                      callbacks.lr_monitor,
                       # callbacks.early_stopping,
                      callbacks.MetricsLogCallback(cfg),
                      callbacks.DistributionLogCallback(wandb_logger, data_module),
                      callbacks.CodeSnapshot(root='./',
                                             output_file=os.path.join(cfg.directory, 'source_code.zip'),
                                             filetype=['.py', '.yml']
                                             )
                      ]

    trainer = pl.Trainer(gpus=args.gpus,
                         accelerator=accelerator,
                         default_root_dir=cfg.directory,
                         logger=wandb_logger,
                         callbacks=callbacks_list,
                         **cfg.trainer,
                         )

    if args.eval:
        print('Eval mode')
        trainer.validate(trainer_model, val_dataloaders=data_module.val_dataloader())
        return

    if args.lr_finder:
        # Run learning rate finder
        lr_finder = trainer.tuner.lr_find(trainer_model, data_module, max_lr=0.1)
        lr_finder.results

        # Plot with
        fig = lr_finder.plot(suggest=True)
        new_lr = lr_finder.suggestion()
        trainer.logger.experiment.log({"Lr_FINDER": wandb.Image(fig, caption=f"{new_lr}")}, commit=True)
        trainer_model.lr = new_lr

    trainer.fit(trainer_model, data_module)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--gpus', type=int, nargs='+', default=[1], help='gpus to use')
    parser.add_argument('--eval', action='store_true', help='run evaluation')
    parser.add_argument('--lr_finder', action='store_true', help='run optimal lr finder')
    parser.add_argument('--wandb_offline', action='store_true', help='Dont push logging to wandb ui')
    args = parser.parse_args()
    main(args)
