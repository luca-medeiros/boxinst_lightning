#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 18:43:47 2021

@author: sehwan.joo
"""

from modeling.backbone.fpn import FPN
from timm import create_model


def build_fpn_backbone(cfg):

    model = create_model(cfg.model.backbone.name,
                         features_only=True, pretrained=True)
    fpn = FPN(cfg, model, cfg.model.backbone.in_features,
              cfg.model.backbone.output_channel)

    return fpn
