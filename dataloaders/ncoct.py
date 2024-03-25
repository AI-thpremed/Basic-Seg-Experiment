# Originally written by Kazuto Nakashima 
# https://github.com/kazuto1011/deeplab-pytorch
import cv2

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

class OCTDataset(BaseDataSet):
    """
    Pascal Voc dataset
    http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
    """
    def __init__(self, **kwargs):
        self.num_classes =2
        self.palette = palette.get_voc_palette(self.num_classes)
        super(OCTDataset, self).__init__(**kwargs)

    def _set_files(self):
        self.root = os.path.join(self.root, '/data/gaowh/data/files/drive')
        self.image_dir = os.path.join(self.root,'images')
        self.label_dir = os.path.join(self.root, 'masks')

        file_list = os.path.join(self.root, "fold_1/", self.split + ".txt")
        self.files = [line.rstrip() for line in tuple(open(file_list, "r"))]
    
    def _load_data(self, index):
        image_id = self.files[index]
        image_path = os.path.join(self.image_dir, image_id )
        label_path = os.path.join(self.label_dir, image_id )
        # image = np.asarray(Image.open(image_path), dtype=np.float32)
        # label = np.asarray(Image.open(label_path), dtype=np.int32)
        image = np.asarray(Image.open(image_path))
        image = np.asarray(Image.open(image_path).convert('RGB'))

        label = np.asarray(Image.open(label_path))
        image_id = self.files[index].split("/")[-1].split(".")[0]
        return image, label, image_id


class OCT(BaseDataLoader):
    def __init__(self, data_dir, batch_size, split, square_resize=None, crop_size=None, base_size=None, scale=True, num_workers=1, val=False,
                    shuffle=False, flip=False, rotate=False, augment=False, val_split= None, return_id=False,
                    clahe=False, randombrightness=False, randomblur=False, randomcontrast=False):
        
        self.MEAN = [0.45734706, 0.43338275, 0.40058118]
        self.STD = [0.23965294, 0.23532275, 0.2398498]

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
        # if split in ["train", "trainval", "val", "test"]:
    #
        if split in ["training", "trainval", "validation", "test"]:
            self.dataset = OCTDataset(**kwargs)
        else: raise ValueError(f"Invalid split name {split}")
        super(OCT, self).__init__(self.dataset, batch_size, shuffle, num_workers, val_split)

