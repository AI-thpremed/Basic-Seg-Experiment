# Originally written by Kazuto Nakashima 
# https://github.com/kazuto1011/deeplab-pytorch

from base import BaseDataSet, BaseDataLoader
from utils import palette
import numpy as np
import os
import scipy
import torch
from PIL import Image
import cv2
from torch.utils.data import Dataset
from torchvision import transforms

class DetachmentDataset(BaseDataSet):
    def __init__(self, **kwargs):
        self.num_classes = 2
        self.palette = palette.get_voc_palette(self.num_classes)
        super(DetachmentDataset, self).__init__(**kwargs)

    def _set_files(self):
        self.root = os.path.join(self.root, 'Detachment')
        self.image_dir = os.path.join(self.root, 'JPEGImages')
        self.label_dir = os.path.join(self.root, 'SegmentationClass')

        file_list = os.path.join(self.root, "ImageSets/Segmentation", self.split + ".txt")
        self.files = [line.rstrip() for line in tuple(open(file_list, "r"))]
    
    def _load_data(self, index):
        image_id = self.files[index]
        image_path = os.path.join(self.image_dir, image_id + '.jpg')
        label_path = os.path.join(self.label_dir, image_id + '.png')
        # image = np.asarray(Image.open(image_path), dtype=np.float32)
        # label = np.asarray(Image.open(label_path), dtype=np.int32)
        image = np.asarray(Image.open(image_path))
        label = np.asarray(Image.open(label_path))
        image_id = self.files[index].split("/")[-1].split(".")[0]
        return image, label, image_id


class Detachment(BaseDataLoader):
    def __init__(self, data_dir, batch_size, split, square_resize=None, crop_size=None, base_size=None, scale=True, num_workers=1, val=False,
                    shuffle=False, flip=False, rotate=False, augment=False, val_split= None, return_id=False,
                    clahe=False, randombrightness=False, randomblur=False, randomcontrast=False):
        
        self.MEAN = [0.485, 0.456, 0.406]
        self.STD  = [0.229, 0.224, 0.225]

        kwargs = {
            'root': data_dir,
            'split': split,
            'mean': self.MEAN,
            'std': self.STD,
            'square_resize': square_resize,
            'augment': augment,
            'crop_size': crop_size,
            'base_size': base_size,
            'scale': scale,
            'flip': flip,
            'rotate': rotate,
            'return_id': return_id,
            'val': val,
            'clahe': clahe,
            'randombrightness': randombrightness,
            'randomblur': randomblur,
            'randomcontrast': randomcontrast,
        }

        if split in ["train", "trainval", "val", "test", "train4", "train5", 'train6']:
            self.dataset = DetachmentDataset(**kwargs)
        else: raise ValueError(f"Invalid split name {split}")
        super(Detachment, self).__init__(self.dataset, batch_size, shuffle, num_workers, val_split)

