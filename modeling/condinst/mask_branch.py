from typing import Dict
import math

import torch
from torch import nn

from layers import sigmoid_focal_loss
from layers import ShapeSpec

from layers import aligned_bilinear, conv_with_kaiming_uniform


INF = 100000000


def build_mask_branch(cfg, input_shape):
    return MaskBranch(cfg, input_shape)


class MaskBranch(nn.Module):
    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__()
        self.in_features = cfg.model.head.condinst.mask_branch.in_features
        self.sem_loss_on = cfg.model.head.condinst.mask_branch.semantic_loss_on
        self.num_outputs = cfg.model.head.condinst.mask_branch.out_channels
        norm = cfg.model.head.condinst.mask_branch.norm
        num_convs = cfg.model.head.condinst.mask_branch.num_convs
        channels = cfg.model.head.condinst.mask_branch.channels
        self.out_stride = input_shape[self.in_features[0]].stride

        feature_channels = {k: v.channels for k, v in input_shape.items()}

        conv_block = conv_with_kaiming_uniform(norm, activation=True)

        self.refine = nn.ModuleList()
        for in_feature in self.in_features:
            self.refine.append(conv_block(
                feature_channels[in_feature],
                channels, 3, 1
            ))

        tower = []
        for i in range(num_convs):
            tower.append(conv_block(
                channels, channels, 3, 1
            ))
        tower.append(nn.Conv2d(
            channels, max(self.num_outputs, 1), 1
        ))
        self.add_module('tower', nn.Sequential(*tower))

        if self.sem_loss_on:
            num_classes = cfg.model.rpn.fcos.num_classes
            self.focal_loss_alpha = cfg.model.rpn.fcos.loss_alpha
            self.focal_loss_gamma = cfg.model.rpn.fcos.loss_gamma

            in_channels = feature_channels[self.in_features[0]]
            self.seg_head = nn.Sequential(
                conv_block(in_channels, channels, kernel_size=3, stride=1),
                conv_block(channels, channels, kernel_size=3, stride=1)
            )

            self.logits = nn.Conv2d(
                channels, num_classes, kernel_size=1, stride=1)

            prior_prob = cfg.model.rpn.fcos.prior_prob
            bias_value = -math.log((1 - prior_prob) / prior_prob)
            torch.nn.init.constant_(self.logits.bias, bias_value)

    def forward(self, features, gt_instances=None):
        for i, f in enumerate(self.in_features):
            if i == 0:
                x = self.refine[i](features[f])
            else:
                x_p = self.refine[i](features[f])

                target_h, target_w = x.size()[2:]
                h, w = x_p.size()[2:]
                assert target_h % h == 0
                assert target_w % w == 0
                factor_h, factor_w = target_h // h, target_w // w
                assert factor_h == factor_w
                x_p = aligned_bilinear(x_p, factor_h)
                x = x + x_p

        mask_feats = self.tower(x)

        if self.num_outputs == 0:
            mask_feats = mask_feats[:, :self.num_outputs]

        losses = {}
        # auxiliary thing semantic loss
        if self.training and self.sem_loss_on:
            logits_pred = self.logits(self.seg_head(
                features[self.in_features[0]]
            ))

            # compute semantic targets
            semantic_targets = []
            for per_im_gt in gt_instances:
                h, w = per_im_gt.gt_bitmasks_full.size()[-2:]
                areas = per_im_gt.gt_bitmasks_full.sum(dim=-1).sum(dim=-1)
                areas = areas[:, None, None].repeat(1, h, w)
                areas[per_im_gt.gt_bitmasks_full == 0] = INF
                areas = areas.permute(1, 2, 0).reshape(h * w, -1)
                min_areas, inds = areas.min(dim=1)
                per_im_sematic_targets = per_im_gt.gt_classes[inds] + 1
                per_im_sematic_targets[min_areas == INF] = 0
                per_im_sematic_targets = per_im_sematic_targets.reshape(h, w)
                semantic_targets.append(per_im_sematic_targets)

            semantic_targets = torch.stack(semantic_targets, dim=0)

            # resize target to reduce memory
            semantic_targets = semantic_targets[
                :, None, self.out_stride // 2::self.out_stride,
                self.out_stride // 2::self.out_stride
            ]

            # prepare one-hot targets
            num_classes = logits_pred.size(1)
            class_range = torch.arange(
                num_classes, dtype=logits_pred.dtype,
                device=logits_pred.device
            )[:, None, None]
            class_range = class_range + 1
            one_hot = (semantic_targets == class_range).float()
            num_pos = (one_hot > 0).sum().float().clamp(min=1.0)

            loss_sem = sigmoid_focal_loss(
                logits_pred, one_hot,
                alpha=self.focal_loss_alpha,
                gamma=self.focal_loss_gamma,
                reduction="sum",
            ) / num_pos
            losses['loss_sem'] = loss_sem

        return mask_feats, losses
