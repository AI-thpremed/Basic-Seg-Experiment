import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from scipy import ndimage
import albumentations as albu

class BaseDataSet(Dataset):
    def __init__(self, root, split, mean, std, square_resize=None, base_size=None, augment=True, val=False,
                crop_size=512, scale=True, flip=True, rotate=False, return_id=False,
                 clahe=False, randombrightness=False, randomblur=False, randomcontrast=False):
        self.root = root
        self.split = split
        self.mean = mean
        self.std = std
        self.square_resize = square_resize
        self.augment = augment
        self.crop_size = crop_size
        if self.augment:
            self.base_size = base_size
            self.scale = scale
            self.flip = flip
            self.rotate = rotate
            self.randombrightness = randombrightness
            self.randomblur = randomblur
            self.randomcontrast = randomcontrast
            if self.randombrightness:
                self.transform_randombrightness = albu.Compose(
                    [albu.OneOf([albu.RandomBrightness(p=1), albu.RandomGamma(p=1)], p=0.5)])
            if self.randomblur:
                self.transform_randomblur = albu.Compose([albu.OneOf(
                    [albu.Sharpen(p=1), albu.Blur(blur_limit=3, p=1), albu.MotionBlur(blur_limit=3, p=1)], p=0.5)])
            if self.randomcontrast:
                self.transform_randomcontrast = albu.Compose(
                    [albu.OneOf([albu.RandomContrast(p=1), albu.HueSaturationValue(p=1)], p=0.5)])
            # if self.randombrightness or self.randomcontrast:
            #     self.transform_randombrightnesscontrast = albu.Compose(
            #         [albu.RandomBrightnessContrast(p=1, brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2))])
            # if self.randomblur:
            #     self.transform_randomblur = albu.Compose([albu.OneOf(
            #         [albu.Sharpen(p=1), albu.Blur(blur_limit=3, p=1), albu.MotionBlur(blur_limit=3, p=1)], p=0.5)])
        self.clahe = clahe
        if self.clahe:
            self.transform_clahe = albu.Compose([albu.CLAHE(p=1.0)])
        self.val = val
        self.files = []
        self._set_files()
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean, std)
        self.return_id = return_id

        cv2.setNumThreads(0)

    def _set_files(self):
        raise NotImplementedError
    
    def _load_data(self, index):
        raise NotImplementedError

    def _val_augmentation(self, image, label):
        if self.square_resize: # 直接缩放
            # self.square_resize = 512
            image = cv2.resize(image, (self.square_resize, self.square_resize), interpolation=cv2.INTER_LINEAR)
            label = Image.fromarray(label).resize((self.square_resize, self.square_resize), resample=Image.NEAREST)
            label = np.asarray(label, dtype=np.int32)
        elif self.crop_size: # 短边缩放+中心裁剪
            h, w = label.shape
            # Scale the smaller side to crop size
            if h < w:
                h, w = (self.crop_size, int(self.crop_size * w / h))
            else:
                h, w = (int(self.crop_size * h / w), self.crop_size)

            image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
            label = Image.fromarray(label).resize((w, h), resample=Image.NEAREST)
            label = np.asarray(label, dtype=np.int32)

            # Center Crop
            h, w = label.shape
            start_h = (h - self.crop_size )// 2
            start_w = (w - self.crop_size )// 2
            end_h = start_h + self.crop_size
            end_w = start_w + self.crop_size
            image = image[start_h:end_h, start_w:end_w]
            label = label[start_h:end_h, start_w:end_w]

        if self.clahe: # 均衡化
            image = self.transform_clahe(image=image)['image']

        return image, label

    def _augmentation(self, image, label):
        h, w, _ = image.shape
        # Scaling, we set the bigger to base size, and the smaller 
        # one is rescaled to maintain the same ratio, if we don't have any obj in the image, re-do the processing
        if self.square_resize:
            image = cv2.resize(image, (self.square_resize, self.square_resize), interpolation=cv2.INTER_LINEAR)
            label = cv2.resize(label, (self.square_resize, self.square_resize), interpolation=cv2.INTER_NEAREST)
        elif self.base_size:
            if self.scale:
                longside = random.randint(int(self.base_size*0.5), int(self.base_size*2.0))
            else:
                longside = self.base_size
            h, w = (longside, int(1.0 * longside * w / h + 0.5)) if h > w else (int(1.0 * longside * h / w + 0.5), longside)
            image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
            label = cv2.resize(label, (w, h), interpolation=cv2.INTER_NEAREST)
    
        h, w, _ = image.shape
        # Rotate the image with an angle between -10 and 10
        if self.rotate:
            angle = random.randint(-10, 10)
            center = (w / 2, h / 2)
            rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, rot_matrix, (w, h), flags=cv2.INTER_LINEAR)#, borderMode=cv2.BORDER_REFLECT)
            label = cv2.warpAffine(label, rot_matrix, (w, h), flags=cv2.INTER_NEAREST)#,  borderMode=cv2.BORDER_REFLECT)

        # Padding to return the correct crop size
        if self.square_resize:
            pass
        elif self.crop_size: # 随机裁剪
            pad_h = max(self.crop_size - h, 0)
            pad_w = max(self.crop_size - w, 0)
            pad_kwargs = {
                "top": 0,
                "bottom": pad_h,
                "left": 0,
                "right": pad_w,
                "borderType": cv2.BORDER_CONSTANT,}
            if pad_h > 0 or pad_w > 0:
                image = cv2.copyMakeBorder(image, value=0, **pad_kwargs)
                label = cv2.copyMakeBorder(label, value=0, **pad_kwargs)
            
            # Cropping 
            h, w, _ = image.shape
            start_h = random.randint(0, h - self.crop_size)
            start_w = random.randint(0, w - self.crop_size)
            end_h = start_h + self.crop_size
            end_w = start_w + self.crop_size
            image = image[start_h:end_h, start_w:end_w]
            label = label[start_h:end_h, start_w:end_w]

        # Random hflip
        if self.flip:
            if random.random() > 0.5:
                image = np.fliplr(image).copy()
                label = np.fliplr(label).copy()
        # clahe
        if self.clahe:
            if random.random() > 0.5:
                image = self.transform_clahe(image=image)['image']
        # randombrightness or randomgamma
        if self.randombrightness:
            image = self.transform_randombrightness(image=image)['image']
        # randomblur
        if self.randomblur:
            image = self.transform_randomblur(image=image)['image']
        # randomcontrast
        if self.randomcontrast:
            image = self.transform_randomcontrast(image=image)['image']

        return image, label
        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        image, label, image_id = self._load_data(index)
        if self.val:
            image, label = self._val_augmentation(image, label)
        elif self.augment:
            image, label = self._augmentation(image, label)

        label = torch.from_numpy(np.array(label, dtype=np.int32)).long()
        image = Image.fromarray(np.uint8(image))
        if self.return_id:
            return  self.normalize(self.to_tensor(image)), label, image_id
        return self.normalize(self.to_tensor(image)), label

    def __repr__(self):
        fmt_str = "Dataset: " + self.__class__.__name__ + "\n"
        fmt_str += "    # data: {}\n".format(self.__len__())
        fmt_str += "    Split: {}\n".format(self.split)
        fmt_str += "    Root: {}".format(self.root)
        return fmt_str

