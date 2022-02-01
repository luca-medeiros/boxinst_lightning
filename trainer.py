import torch
import numpy as np
import pytorch_lightning as pl
import torch.optim as optim
import torch.nn.functional as F
import torch_optimizer as optim_jet
import torchvision.transforms as transforms
from modeling.condinst.condinst import CondInst
from modeling.lr_scheduler import WarmupMultiStepLR, WarmupCosineLR


class Trainer(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        print('==> Initializing trainer..')
        self.cfg = cfg
        self.opt = self.cfg.opt
        self.lr = self.cfg.lr
        self.model = CondInst(cfg)
        self.save_hyperparameters()

    def get_parameters(self):
        params = [{'params': self.model.parameters()}]
        return params

    def configure_optimizers(self):
        pars = self.get_parameters()
        optimizer = eval(self.cfg.optimizer_name)(pars, lr=self.lr, **self.cfg.optimizer)
        scheduler = eval(self.cfg.lr_scheduler_name)(optimizer, **self.cfg.lr_scheduler)
        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

    def training_step(self, batch, batch_idx):
        loss_dict = self.model(batch)
        for loss_name, val in loss_dict.items():
            self.log(f'train/{loss_name}', val, on_step=True, on_epoch=True, logger=True)

        loss = sum(loss_dict.values())
        self.log('train/total_loss', loss, on_step=True, on_epoch=True, logger=True)
        return {'loss': loss}

    def validation_step_end(self, outs):
        return outs

    def validation_step(self, batch, batch_idx):
        loss_dict, results_instances = self.model(batch)
        for loss_name, val in loss_dict.items():
            self.log(f'val/{loss_name}', val, on_step=True, on_epoch=True, logger=True)

        loss = sum(loss_dict.values())
        self.log('val/total_loss', loss, on_step=True, on_epoch=True, logger=True)
        return results_instances

    def forward(self, batch, batch_idx):
        return self.model(batch)

    def preprocess(self, sample):
        sample = transforms.ToPILImage()(sample)
        sample = self.t(sample)
        sample = np.array(sample, dtype=np.uint8)
        sample = self.normalizer(self.ToTensor(sample))

        return sample.float()

    def predict(self, imgs, texts=None, pred_classes=True):
        '''
        imgs should be RGB
        texts list
        '''
        if texts is None:
            texts = self.classes
        data = torch.stack([self.preprocess(k) for k in imgs])
        with torch.no_grad():
            output = self(data.to(self.device), texts.to(self.device))
        if pred_classes:
            output['pred_classes'] = [texts[k] for k in output['preds'].cpu().numpy().flatten()]
        return output
