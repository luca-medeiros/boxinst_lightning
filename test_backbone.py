#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 18:43:47 2021

@author: sehwan.joo
"""

from config import config
from box import Box
from timm import create_model
import torch

cfg = Box(config)


model = create_model('efficientnet_b3a',
                     features_only=True, pretrained=True)
output = model(torch.randn(cfg.data.batch_size,
                           3,
                           cfg.data.input_size,
                           cfg.data.input_size,
                           ))

for i, o in enumerate(output):
    print(f'you should use cfg.model.backbone.channels p{i+1}: {o.shape[1]}')
