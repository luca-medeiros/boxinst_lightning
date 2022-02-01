import cv2
import numpy as np
import random
import os.path as osp
import torch
import pytorch_lightning as pl
import torch.utils.data as data

from pycocotools.coco import COCO
from data.augmentation import SSDAugmentation, ValAugmentation, albu_augmentation
from structures import Instances, Boxes, BitMasks


def get_label_map(cfg):
    if cfg.data.label_map is None:
        return {x: x for x in range(len(cfg.data.class_names))}
    else:
        return cfg.data.label_map


class COCOAnnotationTransform(object):
    """Transforms a COCO annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes
    """

    def __init__(self, cfg):
        self.label_map = get_label_map(cfg)

    def __call__(self, target, width, height):
        """
        Args:
            target (dict): COCO target json annotation as a python dict
            height (int): height
            width (int): width
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class idx]
        """
        scale = np.array([width, height, width, height])
        res = []
        for obj in target:
            if 'bbox' in obj:
                bbox = obj['bbox']
                label_idx = obj['category_id']
                label_idx = self.label_map[label_idx]
                final_box = list(
                    np.array([bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]]))
                final_box.append(label_idx)
                res += [final_box]  # [xmin, ymin, xmax, ymax, label_idx]
            else:
                print("No bbox found for object ", obj)

        return res


class COCODataset(data.Dataset):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.
    Args:
        root (string): Root directory where images are downloaded to.
        set_name (string): Name of the specific set of COCO images.
        transform (callable, optional): A function/transform that augments the
                                        raw images`
        prep_crowds (bool): Whether or not to prepare crowds for the evaluation step.
    """

    def __init__(self, cfg, image_path, info_file, transform=None,
                 dataset_name='MS COCO', has_gt=True):
        self.cfg = cfg
        self.root = image_path
        self.coco = COCO(info_file)
        self.classes = self.coco.cats

        self.ids = list(self.coco.imgToAnns.keys())
        if len(self.ids) == 0 or not has_gt:
            self.ids = list(self.coco.imgs.keys())
        self.ids = self.ids
        self.transform = transform
        self.target_transform = COCOAnnotationTransform(cfg)

        self.name = dataset_name
        self.has_gt = has_gt

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, (target, masks, num_crowds)).
                   target is the object returned by ``coco.loadAnns``.
        """
        img_tensor, boxes, masks, labels, img_id, file_name, shape = self.pull_item(index)
        return img_tensor, boxes, masks, labels, img_id, file_name, shape

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target, masks, height, width, crowd).
                   target is the object returned by ``coco.loadAnns``.
        """
        img_id = self.ids[index]

        if self.has_gt:
            ann_ids = self.coco.getAnnIds(imgIds=img_id)

            target = [x for x in self.coco.loadAnns(
                ann_ids) if x['image_id'] == img_id]
        else:
            target = []

        file_name = self.coco.loadImgs(img_id)[0]['file_name']

        path = osp.join(self.root, file_name)
        assert osp.exists(path), 'Image path does not exist: {}'.format(path)

        img = cv2.imread(path)
        height, width, _ = img.shape

        if len(target) > 0:
            # Pool all the masks for this image into one [num_objects,height,width] matrix
            masks = [self.coco.annToMask(obj).reshape(-1) for obj in target]
            masks = np.vstack(masks)
            masks = masks.reshape(-1, height, width)

        if self.target_transform is not None and len(target) > 0:
            target = self.target_transform(target, width, height)

        if self.transform is not None:
            if len(target) > 0:
                target = np.array(target)
                if self.cfg.data.augment_albumentation:
                    transformed = albu_augmentation(image=img, bboxes=target.copy())
                    if len(transformed['bboxes']) > 0:
                        img = transformed['image']
                        target = np.array(transformed['bboxes'])

                img, masks, boxes, labels = self.transform(img,
                                                           masks,
                                                           target[:, :4],
                                                           {'labels': target[:, 4]})
                labels = labels['labels']

            else:
                img, _, _, _ = self.transform(img, np.zeros((1, height, width), dtype=np.float), np.array(
                    [[0, 0, 1, 1]]), {'labels': np.array([0])})
                masks = None
                boxes = None
                labels = None

        if target.shape[0] == 0:
            print(
                'Warning: Augmentation output an example with no ground truth. Resampling...')

            return self.pull_item(random.randint(0, len(self.ids)-1))
        img_tensor = torch.from_numpy(img).permute(2, 0, 1)

        return img_tensor, boxes, masks, labels, img_id, file_name, (height, width)

    def pull_image(self, index):
        '''Returns the original image object at index in PIL form
        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.
        Argument:
            index (int): index of img to show
        Return:
            cv2 img
        '''
        img_id = self.ids[index]
        path = self.coco.loadImgs(img_id)[0]['file_name']
        return cv2.imread(osp.join(self.root, path), cv2.IMREAD_COLOR)

    def pull_anno(self, index):
        '''Returns the original annotation of image at index
        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.
        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        return self.coco.loadAnns(ann_ids)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(
            tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(
            tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).
    Arguments:
        batch: (tuple) A tuple of tensor images and (lists of annotations, masks)
    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list<tensor>, list<tensor>, list<int>) annotations for a given image are stacked
                on 0 dim. The output gt is a tuple of annotations and masks.
    """
    output = []
    for img, box, mask, label,  img_id, file_name, shape in batch:
        instance = Instances(shape)
        instance.gt_boxes = Boxes(box)
        instance.gt_classes = torch.Tensor(label)
        # instance.gt_bitmasks = BitMasks(mask)
        output.append({'image': torch.FloatTensor(img),
                       'instances': instance,
                       'image_id': img_id,
                       'file_name': file_name,
                       'height': shape[0],
                       'width': shape[1]})

    return output


class DataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.batch_size = cfg.data.batch_size

        transforms = None
        self.transforms_train = transforms
        self.transforms_val = None

    def setup(self):
        # Called on every GPU
        self.train = COCODataset(cfg=self.cfg,
                                 image_path=self.cfg.data.train_path,
                                 info_file=self.cfg.data.train_json,
                                 transform=SSDAugmentation(self.cfg))
        self.classes = self.train.classes
        self.ndata = self.train.__len__()

        self.val = COCODataset(cfg=self.cfg,
                               image_path=self.cfg.data.val_path,
                               info_file=self.cfg.data.val_json,
                               transform=ValAugmentation(self.cfg)
                               )

        assert len(self.classes) == len(self.val.classes), 'Train {len(self.classes)} and val {len(self.val.classes)} classes doesnt match.'

    def train_dataloader(self):
        print('Trainset length: ', len(self.train))
        return torch.utils.data.DataLoader(self.train,
                                           batch_size=self.batch_size,
                                           shuffle=True,
                                           num_workers=self.cfg.data.num_workers,
                                           collate_fn=detection_collate)

    def val_dataloader(self, shuffle=False):
        print('Testset length: ', len(self.val))
        return torch.utils.data.DataLoader(self.val,
                                           batch_size=self.batch_size * 4,
                                           shuffle=shuffle,
                                           num_workers=self.cfg.data.num_workers,
                                           collate_fn=detection_collate)

    def test_dataloader(self):
        ...

