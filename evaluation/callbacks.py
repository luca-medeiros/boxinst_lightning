# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 17:22:05 2021


@author: Luca Medeiros, lucamedeiros@outlook.com
"""
import zipfile
import wandb
import numpy as np

from pathlib import Path
from typing import Union, List, Optional
from pytorch_lightning.utilities import rank_zero_warn, rank_zero_info, rank_zero_only
from pytorch_lightning.callbacks import Callback, EarlyStopping, LearningRateMonitor, ModelCheckpoint, RichProgressBar
from nuvitide import TIDE, datasets

from evaluation import COCOEvaluator


early_stopping = EarlyStopping('val_loss', patience=5)
lr_monitor = LearningRateMonitor(logging_interval='step')
progress_bar = RichProgressBar()


def model_checkpoint(cfg):
    return ModelCheckpoint(dirpath=cfg.directory,
                           # monitor='val_metrics/val_f1',
                           # mode='max',
                           # filename="model_{epoch}",
                           verbose=True,
                           save_last=False,
                           save_top_k=-1,
                           every_n_val_epochs=1
                           )


class DistributionLogCallback(Callback):
    @rank_zero_only
    def __init__(self, logger, datamodule):
        self._log = logger
        self.classes = datamodule.classes
        self.log_distribution(datamodule.train.coco)
        self.log_distribution(datamodule.val.coco, type_='val')

    def log_distribution(self, dataset, type_='train'):
        num_classes = len(self.classes)
        hist_bins = np.arange(num_classes + 1)
        histogram = np.zeros((num_classes,), dtype=np.int)
        annos = list(dataset.anns.values())
        classes = np.asarray(
            [x["category_id"] for x in annos if not x.get("iscrowd", 0)], dtype=np.int
        )
        if len(classes):
            assert classes.min() >= 0, f"Got an invalid category_id={classes.min()}"
            assert (
                classes.max() < num_classes
            ), f"Got an invalid category_id={classes.max()} for a dataset of {num_classes} classes"
        histogram += np.histogram(classes, bins=hist_bins)[0]

        hist = [[k, v] for k, v in zip(self.classes, histogram)]
        table = wandb.Table(data=hist, columns=['Label', 'Value'])
        self._log.experiment.log({F'distribution/{type_}': wandb.plot.bar(table, "Label", "Value", title=f'{type_} distribution')})


class MetricsLogCallback(Callback):
    def __init__(self, cfg):
        self.cfg = cfg
        self.sanity = True
        self.evaluator = COCOEvaluator()
        self.tide = TIDE()
        self.gt_data = datasets.COCO(path=cfg.data.val_json, name='gt')
        self.aps = [50, 60, 70, 80, 90, 95]

    def on_sanity_check_end(self, trainer, pl_module):
        self.sanity = False

    def on_train_start(self, trainer, pl_module):
        self._log = trainer.logger

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        ...

    def on_train_epoch_end(self, trainer, pl_module):
        ...

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if self.sanity:
            return None
        self.evaluator.process(batch, outputs)

    def on_validation_epoch_start(self, trainer, pl_module):
        self.evaluator.reset()

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.sanity:
            return None
        pred_results = self.evaluator.get_results()
        if pred_results:
            pred_data = datasets.COCOResult(path='', dets=pred_results, name='pred')
            for ap in self.aps:
                self.tide.evaluate(self.gt_data,
                                   pred_data,
                                   pos_threshold=(ap/100),
                                   mode=TIDE.BOX,
                                   name=f'mAP-@{ap}')
            self.tide.summarize()
            output_results = self.tide.get_summarize()
            for ap, values in output_results.items():
                type_ = values.pop('Type')
                for key, value in values.items():
                    if isinstance(value, dict):
                        for sub_key, sub_value in value.items():
                            self.log(f'val_metrics_{type_}/{sub_key}_{ap}', sub_value)
                    else:
                        self.log(f'val_metrics_{type_}/{key}_{ap}', value)


class CodeSnapshot(Callback):

    DEFAULT_FILENAME = "code.zip"

    def __init__(
        self,
        root: Optional[Union[str, Path]] = ".",
        output_file: Optional[Union[str, Path]] = "./someplace/hello.zip",
        filetype: Union[str, List[str]] = ".py",
    ):
        """
        Callback that takes a snapshot of all source files and saves them to a ZIP file.
        By default, the file is saved to the folder where checkpoints are saved, i.e., the dirpath
        of ModelCheckpoint.
        Arguments:
            root: the root folder containing the files for collection
            output_file: path to zip file, e.g., "path/to/code.zip"
            filetype: list of file types, e.g., ".py", ".txt", etc.
        """
        self._root = root
        self._output_file = output_file
        self._filetype = filetype

    @rank_zero_only
    def on_train_start(self, trainer, pl_module):
        if not self._output_file and not trainer.checkpoint_callback:
            rank_zero_warn(
                "Trainer has no checkpoint callback and output file where to save snapshot was not specified."
                " Code snapshot not saved!"
            )
            return
        self._output_file = self._output_file or Path(
            trainer.checkpoint_callback.dirpath, CodeSnapshot.DEFAULT_FILENAME
        )
        self._output_file = Path(self._output_file).absolute()
        snapshot_files(
            root=self._root, output_file=self._output_file, filetype=self._filetype
        )
        rank_zero_info(
            f"Code snapshot saved to {self._output_file.relative_to(Path.cwd())}"
        )


def snapshot_files(
    root: Union[str, Path] = ".",
    output_file: Union[str, Path] = "code.zip",
    filetype: Union[str, List[str]] = ".py",
):
    """
    Collects all source files in a folder and saves them to a ZIP file.
    Arguments:
        root: the root folder containing the files for collection
        output_file: path to zip file, e.g., "path/to/code.zip"
        filetype: list of file types, e.g., ".py", ".txt", etc.
    """
    root = Path(root).absolute()
    output_file = Path(output_file).absolute()
    output_file.parent.mkdir(exist_ok=True, parents=True)
    suffixes = [filetype] if isinstance(filetype, str) else filetype

    zip_file = zipfile.ZipFile(output_file, "w", zipfile.ZIP_DEFLATED)

    for p in root.rglob("*"):
        if p.suffix in suffixes:
            zip_file.write(p.relative_to(root))

    zip_file.close()