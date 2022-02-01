#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 17:23:10 2021

@author: sehwan.joo
"""

import cv2
import types
import torch
import random
import numpy as np
import torch.nn.functional as F
import albumentations as A
from math import sqrt
from collections import namedtuple
from albumentations import DualTransform
from albumentations.augmentations.bbox_utils import denormalize_bbox, normalize_bbox


def intersect(box_a, box_b):
    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]


def jaccard_numpy(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: Multiple bounding boxes, Shape: [num_boxes,4]
        box_b: Single bounding box, Shape: [4]
    Return:
        jaccard overlap: Shape: [box_a.shape[0], box_a.shape[1]]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1]))  # [A,B]
    area_b = ((box_b[2]-box_b[0]) *
              (box_b[3]-box_b[1]))  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


class Compose(object):
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> augmentations.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, masks=None, boxes=None, labels=None):
        for t in self.transforms:
            img, masks, boxes, labels = t(img, masks, boxes, labels)
        return img, masks, boxes, labels


class Lambda(object):
    """Applies a lambda as a transform."""

    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, img, masks=None, boxes=None, labels=None):
        return self.lambd(img, masks, boxes, labels)


class ConvertFromInts(object):
    def __call__(self, image, masks=None, boxes=None, labels=None):
        return image.astype(np.float32), masks, boxes, labels


class ToAbsoluteCoords(object):
    def __call__(self, image, masks=None, boxes=None, labels=None):
        height, width, channels = image.shape
        boxes[:, 0] *= width
        boxes[:, 2] *= width
        boxes[:, 1] *= height
        boxes[:, 3] *= height

        return image, masks, boxes, labels


class ToPercentCoords(object):
    def __call__(self, image, masks=None, boxes=None, labels=None):
        height, width, channels = image.shape
        boxes[:, 0] /= width
        boxes[:, 2] /= width
        boxes[:, 1] /= height
        boxes[:, 3] /= height

        return image, masks, boxes, labels


class Pad(object):
    """
    Pads the image to the input width and height, filling the
    background with mean and putting the image in the top-left.
    Note: this expects im_w <= width and im_h <= height
    """

    def __init__(self, width, height, mean, pad_gt=True):
        self.mean = mean
        self.width = width
        self.height = height
        self.pad_gt = pad_gt

    def __call__(self, image, masks, boxes=None, labels=None):
        im_h, im_w, depth = image.shape

        expand_image = np.zeros(
            (self.height, self.width, depth),
            dtype=image.dtype)
        expand_image[:, :, :] = self.mean
        expand_image[:im_h, :im_w] = image

        if self.pad_gt:
            expand_masks = np.zeros(
                (masks.shape[0], self.height, self.width),
                dtype=masks.dtype)
            expand_masks[:, :im_h, :im_w] = masks
            masks = expand_masks

        return expand_image, masks, boxes, labels


class Resize(object):
    """ If preserve_aspect_ratio is true, this resizes to an approximate area of max_size * max_size """

    @staticmethod
    def calc_size_preserve_ar(img_w, img_h, max_size):
        """ I mathed this one out on the piece of paper. Resulting width*height = approx max_size^2 """
        ratio = sqrt(img_w / img_h)
        w = max_size * ratio
        h = max_size / ratio
        return int(w), int(h)

    def __init__(self, cfg, resize_gt=True, val=False):
        self.cfg = cfg
        self.resize_gt = resize_gt
        self.max_size = cfg.data.input_size
        self.preserve_aspect_ratio = cfg.data.preserve_aspect_ratio
        self.val = val

    def __call__(self, image, masks, boxes, labels=None):
        img_h, img_w, _ = image.shape
        if self.val:
            max_size = self.max_size[-1]
        else:
            max_size = random.choice(self.max_size)
        if self.preserve_aspect_ratio:
            width, height = Resize.calc_size_preserve_ar(
                img_w, img_h, max_size)
        else:
            width, height = max_size, max_size

        image = cv2.resize(image, (width, height))

        if self.resize_gt:
            # Act like each object is a color channel
            masks = masks.transpose((1, 2, 0))
            masks = cv2.resize(masks, (width, height))

            # OpenCV resizes a (w,h,1) array to (s,s), so fix that
            if len(masks.shape) == 2:
                masks = np.expand_dims(masks, 0)
            else:
                masks = masks.transpose((2, 0, 1))

            # Scale bounding boxes (which are currently absolute coordinates)
            boxes[:, [0, 2]] *= (width / img_w)
            boxes[:, [1, 3]] *= (height / img_h)

        # Discard boxes that are smaller than we'd like
        w = boxes[:, 2] - boxes[:, 0]
        h = boxes[:, 3] - boxes[:, 1]

        keep = (w > self.cfg.data.discard_box_width) * \
            (h > self.cfg.data.discard_box_height)
        # masks = masks[keep]
        boxes = boxes[keep]
        labels['labels'] = labels['labels'][keep]
        labels['num_crowds'] = (labels['labels'] < 0).sum()

        return image, masks, boxes, labels


class RandomSaturation(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image, masks=None, boxes=None, labels=None):
        if random.randint(2):
            image[:, :, 1] *= random.uniform(self.lower, self.upper)

        return image, masks, boxes, labels


class RandomHue(object):
    def __init__(self, delta=18.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta

    def __call__(self, image, masks=None, boxes=None, labels=None):
        if random.randint(2):
            image[:, :, 0] += random.uniform(-self.delta, self.delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
        return image, masks, boxes, labels


class RandomLightingNoise(object):
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def __call__(self, image, masks=None, boxes=None, labels=None):
        # Don't shuffle the channels please, why would you do this

        # if random.randint(2):
        #     swap = self.perms[random.randint(len(self.perms))]
        #     shuffle = SwapChannels(swap)  # shuffle channels
        #     image = shuffle(image)
        return image, masks, boxes, labels


class ConvertColor(object):
    def __init__(self, current='BGR', transform='HSV'):
        self.transform = transform
        self.current = current

    def __call__(self, image, masks=None, boxes=None, labels=None):
        if self.current == 'BGR' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.current == 'HSV' and self.transform == 'BGR':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        else:
            raise NotImplementedError
        return image, masks, boxes, labels


class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    # expects float image
    def __call__(self, image, masks=None, boxes=None, labels=None):
        if random.randint(2):
            alpha = random.uniform(self.lower, self.upper)
            image *= alpha
        return image, masks, boxes, labels


class RandomBrightness(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, image, masks=None, boxes=None, labels=None):
        if random.randint(2):
            delta = random.uniform(-self.delta, self.delta)
            image += delta
        return image, masks, boxes, labels


class ToCV2Image(object):
    def __call__(self, tensor, masks=None, boxes=None, labels=None):
        return tensor.cpu().numpy().astype(np.float32).transpose((1, 2, 0)), masks, boxes, labels


class ToTensor(object):
    def __call__(self, cvimage, masks=None, boxes=None, labels=None):
        return torch.from_numpy(cvimage.astype(np.float32)).permute(2, 0, 1), masks, boxes, labels


class RandomSampleCrop(object):
    """Crop
    Arguments:
        img (Image): the image being input during training
        boxes (Tensor): the original bounding boxes in pt form
        labels (Tensor): the class labels for each bbox
        mode (float tuple): the min and max jaccard overlaps
    Return:
        (img, boxes, classes)
            img (Image): the cropped image
            boxes (Tensor): the adjusted bounding boxes in pt form
            labels (Tensor): the class labels for each bbox
    """

    def __init__(self):
        self.sample_options = (
            # using entire original input image
            None,
            # sample a patch s.t. MIN jaccard w/ obj in .1,.3,.4,.7,.9
            (0.1, None),
            (0.3, None),
            (0.7, None),
            (0.9, None),
            # randomly sample a patch
            (None, None),
        )

    def __call__(self, image, masks, boxes=None, labels=None):
        height, width, _ = image.shape
        while True:
            # randomly choose a mode
            mode = random.choice(self.sample_options)
            if mode is None:
                return image, masks, boxes, labels

            min_iou, max_iou = mode
            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('inf')

            # max trails (50)
            for _ in range(50):
                current_image = image

                w = random.uniform(0.3 * width, width)
                h = random.uniform(0.3 * height, height)

                # aspect ratio constraint b/t .5 & 2
                if h / w < 0.5 or h / w > 2:
                    continue

                left = random.uniform(width - w)
                top = random.uniform(height - h)

                # convert to integer rect x1,y1,x2,y2
                rect = np.array([int(left), int(top), int(left+w), int(top+h)])

                # calculate IoU (jaccard overlap) b/t the cropped and gt boxes
                overlap = jaccard_numpy(boxes, rect)

                # This piece of code is bugged and does nothing:
                # https://github.com/amdegroot/ssd.pytorch/issues/68
                #
                # However, when I fixed it with overlap.max() < min_iou,
                # it cut the mAP in half (after 8k iterations). So it stays.
                #
                # is min and max overlap constraint satisfied? if not try again
                if overlap.min() < min_iou and max_iou < overlap.max():
                    continue

                # cut the crop from the image
                current_image = current_image[rect[1]:rect[3], rect[0]:rect[2],
                                              :]

                # keep overlap with gt box IF center in sampled patch
                centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0

                # mask in all gt boxes that above and to the left of centers
                m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])

                # mask in all gt boxes that under and to the right of centers
                m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])

                # mask in that both m1 and m2 are true
                mask = m1 * m2

                # [0 ... 0 for num_gt and then 1 ... 1 for num_crowds]
                num_crowds = labels['num_crowds']
                crowd_mask = np.zeros(mask.shape, dtype=np.int32)

                if num_crowds > 0:
                    crowd_mask[-num_crowds:] = 1

                # have any valid boxes? try again if not
                # Also make sure you have at least one regular gt
                if not mask.any() or np.sum(1-crowd_mask[mask]) == 0:
                    continue

                # take only the matching gt masks
                current_masks = masks[mask, :, :].copy()

                # take only matching gt boxes
                current_boxes = boxes[mask, :].copy()

                # take only matching gt labels
                labels['labels'] = labels['labels'][mask]
                current_labels = labels

                # We now might have fewer crowd annotations
                if num_crowds > 0:
                    labels['num_crowds'] = np.sum(crowd_mask[mask])

                # should we use the box left and top corner or the crop's
                current_boxes[:, :2] = np.maximum(current_boxes[:, :2],
                                                  rect[:2])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, :2] -= rect[:2]

                current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:],
                                                  rect[2:])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, 2:] -= rect[:2]

                # crop the current masks to the same dimensions as the image
                current_masks = current_masks[:,
                                              rect[1]:rect[3], rect[0]:rect[2]]

                return current_image, current_masks, current_boxes, current_labels


class Expand(object):
    def __init__(self, mean):
        self.mean = mean

    def __call__(self, image, masks, boxes, labels):
        if random.randint(2):
            return image, masks, boxes, labels

        height, width, depth = image.shape
        ratio = random.uniform(1, 4)
        left = random.uniform(0, width*ratio - width)
        top = random.uniform(0, height*ratio - height)

        expand_image = np.zeros(
            (int(height*ratio), int(width*ratio), depth),
            dtype=image.dtype)
        expand_image[:, :, :] = self.mean
        expand_image[int(top):int(top + height),
                     int(left):int(left + width)] = image
        image = expand_image

        expand_masks = np.zeros(
            (masks.shape[0], int(height*ratio), int(width*ratio)),
            dtype=masks.dtype)
        expand_masks[:, int(top):int(top + height),
                     int(left):int(left + width)] = masks
        masks = expand_masks

        boxes = boxes.copy()
        boxes[:, :2] += (int(left), int(top))
        boxes[:, 2:] += (int(left), int(top))

        return image, masks, boxes, labels


class RandomMirror(object):
    def __call__(self, image, masks, boxes, labels):
        _, width, _ = image.shape
        if random.randint(2):
            image = image[:, ::-1]
            masks = masks[:, :, ::-1]
            boxes = boxes.copy()
            boxes[:, 0::2] = width - boxes[:, 2::-2]
        return image, masks, boxes, labels


class RandomFlip(object):
    def __call__(self, image, masks, boxes, labels):
        height, _, _ = image.shape
        if random.randint(2):
            image = image[::-1, :]
            masks = masks[:, ::-1, :]
            boxes = boxes.copy()
            boxes[:, 1::2] = height - boxes[:, 3::-2]
        return image, masks, boxes, labels


class RandomRot90(object):
    def __call__(self, image, masks, boxes, labels):
        old_height, old_width, _ = image.shape
        k = random.randint(4)
        image = np.rot90(image, k)
        masks = np.array([np.rot90(mask, k) for mask in masks])
        boxes = boxes.copy()
        for _ in range(k):
            boxes = np.array([[box[1], old_width - 1 - box[2],
                             box[3], old_width - 1 - box[0]] for box in boxes])
            old_width, old_height = old_height, old_width
        return image, masks, boxes, labels


class SwapChannels(object):
    """Transforms a tensorized image by swapping the channels in the order
     specified in the swap tuple.
    Args:
        swaps (int triple): final order of channels
            eg: (2, 1, 0)
    """

    def __init__(self, swaps):
        self.swaps = swaps

    def __call__(self, image):
        """
        Args:
            image (Tensor): image tensor to be transformed
        Return:
            a tensor with channels swapped according to swap
        """
        # if torch.is_tensor(image):
        #     image = image.data.cpu().numpy()
        # else:
        #     image = np.array(image)
        image = image[:, :, self.swaps]
        return image


class PhotometricDistort(object):
    def __init__(self):
        self.pd = [
            RandomContrast(),
            ConvertColor(transform='HSV'),
            RandomSaturation(),
            RandomHue(),
            ConvertColor(current='HSV', transform='BGR'),
            RandomContrast()
        ]
        self.rand_brightness = RandomBrightness()
        self.rand_light_noise = RandomLightingNoise()

    def __call__(self, image, masks, boxes, labels):
        im = image.copy()
        im, masks, boxes, labels = self.rand_brightness(
            im, masks, boxes, labels)
        if random.randint(2):
            distort = Compose(self.pd[:-1])
        else:
            distort = Compose(self.pd[1:])
        im, masks, boxes, labels = distort(im, masks, boxes, labels)
        return self.rand_light_noise(im, masks, boxes, labels)


class PrepareMasks(object):
    """
    Prepares the gt masks for use_gt_bboxes by cropping with the gt box
    and downsampling the resulting mask to mask_size, mask_size. This
    function doesn't do anything if cfg.use_gt_bboxes is False.
    """

    def __init__(self, mask_size, use_gt_bboxes):
        self.mask_size = mask_size
        self.use_gt_bboxes = use_gt_bboxes

    def __call__(self, image, masks, boxes, labels=None):
        if not self.use_gt_bboxes:
            return image, masks, boxes, labels

        height, width, _ = image.shape

        new_masks = np.zeros((masks.shape[0], self.mask_size ** 2))

        for i in range(len(masks)):
            x1, y1, x2, y2 = boxes[i, :]
            x1 *= width
            x2 *= width
            y1 *= height
            y2 *= height
            x1, y1, x2, y2 = (int(x1), int(y1), int(x2), int(y2))

            # +1 So that if y1=10.6 and y2=10.9 we still have a bounding box
            cropped_mask = masks[i, y1:(y2+1), x1:(x2+1)]
            scaled_mask = cv2.resize(
                cropped_mask, (self.mask_size, self.mask_size))

            new_masks[i, :] = scaled_mask.reshape(1, -1)

        # Binarize
        new_masks[new_masks > 0.5] = 1
        new_masks[new_masks <= 0.5] = 0

        return image, new_masks, boxes, labels


class BackboneTransform(object):
    """
    Transforms a BRG image made of floats in the range [0, 255] to whatever
    input the current backbone network needs.
    transform is a transform config object (see config.py).
    in_channel_order is probably 'BGR' but you do you, kid.
    """

    def __init__(self, transform, mean, std, in_channel_order):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std,  dtype=np.float32)
        self.transform = transform

        # Here I use "Algorithms and Coding" to convert string permutations to numbers
        self.channel_map = {c: idx for idx, c in enumerate(in_channel_order)}
        self.channel_permutation = [self.channel_map[c]
                                    for c in transform.channel_order]

    def __call__(self, img, masks=None, boxes=None, labels=None):

        img = img.astype(np.float32)

        if self.transform.normalize:
            img = (img - self.mean) / self.std
        elif self.transform.subtract_means:
            img = (img - self.mean)
        elif self.transform.to_float:
            img = img / 255

        img = img[:, :, self.channel_permutation]

        return img.astype(np.float32), masks, boxes, labels


class BaseTransform(object):
    """ Transorm to be used when evaluating. """

    def __init__(self, cfg):
        mean = cfg.data.means
        std = cfg.data.std
        self.augment = Compose([
            ConvertFromInts(),
            Resize(cfg, resize_gt=False),
            BackboneTransform(cfg.data.transform, mean, std, 'BGR')
        ])

    def __call__(self, img, masks=None, boxes=None, labels=None):
        return self.augment(img, masks, boxes, labels)


class FastBaseTransform(torch.nn.Module):
    """
    Transform that does all operations on the GPU for super speed.
    This doesn't suppport a lot of config settings and should only be used for production.
    Maintain this as necessary.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.mean = torch.Tensor(cfg.data.means).float().cuda()[
            None, :, None, None]
        self.std = torch.Tensor(cfg.data.std).float().cuda()[
            None, :, None, None]
        self.transform = cfg.backbone.transform

    def forward(self, img):
        self.mean = self.mean.to(img.device)
        self.std = self.std.to(img.device)

        # img assumed to be a pytorch BGR image with channel order [n, h, w, c]
        if self.cfg.data.preserve_aspect_ratio:
            _, h, w, _ = img.size()
            img_size = Resize.calc_size_preserve_ar(
                w, h, self.cfg.data.input_size)
            img_size = (img_size[1], img_size[0])  # Pytorch needs h, w
        else:
            img_size = (self.cfg.data.input_size, self.cfg.data.input_size)

        img = img.permute(0, 3, 1, 2).contiguous()
        img = F.interpolate(img, img_size, mode='bilinear',
                            align_corners=False)

        if self.transform.normalize:
            img = (img - self.mean) / self.std
        elif self.transform.subtract_means:
            img = (img - self.mean)
        elif self.transform.to_float:
            img = img / 255

        if self.transform.channel_order != 'RGB':
            raise NotImplementedError

        img = img[:, (2, 1, 0), :, :].contiguous()

        # Return value is in channel order [n, c, h, w] and RGB
        return img


def do_nothing(img=None, masks=None, boxes=None, labels=None):
    return img, masks, boxes, labels


def enable_if(condition, obj):
    return obj if condition else do_nothing


class SSDAugmentation(object):
    """ Transform to be used when training. """

    def __init__(self, cfg):
        mean = cfg.data.means
        # std = cfg.data.std
        self.augment = Compose([
            ConvertFromInts(),
            # ToAbsoluteCoords(),
            enable_if(cfg.data.augment_photometric_distort,
                      PhotometricDistort()),
            # enable_if(cfg.data.augment_expand, Expand(mean)),
            enable_if(cfg.data.augment_random_sample_crop, RandomSampleCrop()),
            enable_if(cfg.data.augment_random_mirror, RandomMirror()),
            enable_if(cfg.data.augment_random_flip, RandomFlip()),
            enable_if(cfg.data.augment_random_flip, RandomRot90()),
            Resize(cfg),
            enable_if(not cfg.data.preserve_aspect_ratio,
                      Pad(cfg.data.input_size[-1], cfg.data.input_size[-1], mean)),
            # ToPercentCoords(),
            # PrepareMasks(cfg.data.mask_size, cfg.data.use_gt_bboxes),
            # BackboneTransform(cfg.data.transform, mean, std, 'BGR')
        ])

    def __call__(self, img, masks, boxes, labels):
        return self.augment(img, masks, boxes, labels)


class ValAugmentation(object):
    """ Transform to be used when training. """

    def __init__(self, cfg):
        self.augment = Compose([
            ConvertFromInts(),
            Resize(cfg, val=True),
        ])

    def __call__(self, img, masks, boxes, labels):
        return self.augment(img, masks, boxes, labels)


class CustomCutout(DualTransform):
    """
    Custom Cutout augmentation with handling of bounding boxes
    Note: (only supports square cutout regions)

    Author: Kaushal28
    Reference: https://arxiv.org/pdf/1708.04552.pdf
    """

    def __init__(
        self,
        fill_value=0,
        num_holes=4,
        bbox_removal_threshold=0.50,
        min_cutout_size=40,
        max_cutout_size=100,
        always_apply=False,
        p=0.5
    ):
        """
        Class construstor

        :param fill_value: Value to be filled in cutout (default is 0 or black color)
        :param bbox_removal_threshold: Bboxes having content cut by cutout path more than this threshold will be removed
        :param min_cutout_size: minimum size of cutout (192 x 192)
        :param max_cutout_size: maximum size of cutout (512 x 512)
        """
        super(CustomCutout, self).__init__(always_apply, p)  # Initialize parent class
        self.fill_value = fill_value
        self.num_holes = num_holes
        self.bbox_removal_threshold = bbox_removal_threshold
        self.min_cutout_size = min_cutout_size
        self.max_cutout_size = max_cutout_size

    def _get_cutout_position(self, img_height, img_width, cutout_size):
        """
        Randomly generates cutout position as a named tuple

        :param img_height: height of the original image
        :param img_width: width of the original image
        :param cutout_size: size of the cutout patch (square)
        :returns position of cutout patch as a named tuple
        """
        position = namedtuple('Point', 'x y')
        return position(
            np.random.randint(0, img_width - cutout_size + 1),
            np.random.randint(0, img_height - cutout_size + 1)
        )

    def _get_cutout(self, img_height, img_width):
        """
        Creates a cutout pacth with given fill value and determines the position in the original image

        :param img_height: height of the original image
        :param img_width: width of the original image
        :returns (cutout patch, cutout size, cutout position)
        """
        cutout_size = np.random.randint(self.min_cutout_size, self.max_cutout_size + 1)
        cutout_position = self._get_cutout_position(img_height, img_width, cutout_size)
        return np.full((cutout_size, cutout_size, 3), self.fill_value), cutout_size, cutout_position

    def apply(self, image, **params):
        """
        Applies the cutout augmentation on the given image

        :param image: The image to be augmented
        :returns augmented image
        """
        image = image.copy()  # Don't change the original image
        self.img_height, self.img_width, _ = image.shape
        for _n in range(self.num_holes):
            cutout_arr, cutout_size, cutout_pos = self._get_cutout(self.img_height, self.img_width)

            # Set to instance variables to use this later
            self.image = image
            self.cutout_pos = cutout_pos
            self.cutout_size = cutout_size

            image[cutout_pos.y:cutout_pos.y+cutout_size, cutout_pos.x:cutout_size+cutout_pos.x, :] = cutout_arr

        return image

    def apply_to_bbox(self, bbox, **params):
        """
        Removes the bounding boxes which are covered by the applied cutout

        :param bbox: A single bounding box coordinates in pascal_voc format
        :returns transformed bbox's coordinates
        """

        # Denormalize the bbox coordinates
        bbox = denormalize_bbox(bbox, self.img_height, self.img_width)
        x_min, y_min, x_max, y_max = tuple(map(int, bbox))

        bbox_size = (x_max - x_min) * (y_max - y_min)  # width * height
        overlapping_size = np.sum(
            (self.image[y_min:y_max, x_min:x_max, 0] == self.fill_value) &
            (self.image[y_min:y_max, x_min:x_max, 1] == self.fill_value) &
            (self.image[y_min:y_max, x_min:x_max, 2] == self.fill_value)
        )

        # Remove the bbox if it has more than some threshold of content is inside the cutout patch
        if overlapping_size / bbox_size > self.bbox_removal_threshold:
            return normalize_bbox((0, 0, 0, 0), self.img_height, self.img_width)

        return normalize_bbox(bbox, self.img_height, self.img_width)

    def get_transform_init_args_names(self):
        """
        Fetches the parameter(s) of __init__ method
        :returns: tuple of parameter(s) of __init__ method
        """
        return ('fill_value', 'bbox_removal_threshold', 'min_cutout_size', 'max_cutout_size', 'always_apply', 'p')


albu_augmentation = A.Compose([
            CustomCutout(p=0.5),
            A.ImageCompression(always_apply=False, p=.45, quality_lower=69, quality_upper=100, compression_type=1),
            A.RandomBrightnessContrast(p=0.3),
            A.Flip(always_apply=True),  # Either Horizontal, Vertical or both flips
            A.OneOf([
                A.GaussNoise(var_limit=5.0 / 255.0, p=0.50),
                A.CLAHE(always_apply=False, p=0.5, clip_limit=(1, 4), tile_grid_size=(8, 8)),
                A.Blur(always_apply=False, p=0.5, blur_limit=(3, 7)),
            ], p=0.4)
        ], bbox_params=A.BboxParams(format='pascal_voc'))


def visualize_bbox(img, bbox, thickness=2, **kwargs):
    color = (255, 0, 0) # Red
    x_min, y_min, x_max, y_max = bbox
    x_min, x_max, y_min, y_max = int(x_min), int(x_max), int(y_min), int(y_max)
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)
    return img
