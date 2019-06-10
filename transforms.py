import random

import torch
import torchvision
from PIL import Image


class GroupRandomCrop(object):

    def __init__(self, size):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img_group):
        w, h = img_group[0].size
        th, tw = self.size

        out_images = list()

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)

        for img in img_group:
            assert(img.size[0] == w and img.size[1] == h)
            if w == tw and h == th:
                out_images.append(img)
            else:
                out_images.append(img.crop((x1, y1, x1 + tw, y1 + th)))

        return out_images


class GroupFullyConv(object):

    def __init__(self, size):
        self.worker = GroupRandomCrop(size)

    def __call__(self, img_group):
        crops = [self.worker(img_group) for _ in range(3)]
        return [item for sublist in crops for item in sublist]


class GroupCenterCrop(object):

    def __init__(self, size):
        self.worker = torchvision.transforms.CenterCrop(size)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]


class ConditionedGroupCenterCrop(object):

    def __init__(self, max_size=808, sec_size=256):
        self.max_size = max_size
        self.sec_size = sec_size
        self.w_worker = torchvision.transforms.CenterCrop((sec_size, self.max_size))
        self.h_worker = torchvision.transforms.CenterCrop((self.max_size, sec_size))

    def __call__(self, img_group):
        w, h = img_group[0].size

        if w > self.max_size:
            print('\nNeed to crop width dimention! Image size from: {}x{} to {}x{}\n'.format(
                w, h, self.max_size, self.sec_size))
            return [self.w_worker(img) for img in img_group]
        elif h > self.max_size:
            print('\nNeed to crop heigth dimention! Image size from: {}x{} to {}x{}\n'.format(
                w, h, self.sec_size, self.max_size))
            return [self.h_worker(img) for img in img_group]
        else:
            return img_group


class GroupTenCrop(object):

    def __init__(self, size):
        self.worker = torchvision.transforms.TenCrop(size)

    def __call__(self, img_group):
        cropped_imgs = [self.worker(img) for img in img_group]
        reordered_imgs = [[group[i] for group in cropped_imgs] for i in range(10)]
        return [img for sublist in reordered_imgs for img in sublist]


class GroupRandomHorizontalFlip(object):

    def __call__(self, img_group, is_flow=False):
        v = random.random()
        if v < 0.5:
            ret = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in img_group]
            return ret
        else:
            return img_group


class GroupNormalize(object):
    def __init__(self, mean=None, std=None, num_channels=3):
        self.mean = mean if mean is not None else [0] * num_channels
        self.std = std if std is not None else [1] * num_channels

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.sub_(m).div_(s)

        return tensor


class GroupResize(object):

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.worker = torchvision.transforms.Resize(size, interpolation)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]


class GroupToTensorStack(object):

    def __call__(self, img_group):
        return torch.stack([torchvision.transforms.ToTensor()(img)*255 for img in img_group], dim=1)


def get_default_transforms(mode):
    input_mean = [114.75, 114.75, 114.75]  # [0.485, 0.456, 0.406] -> 114.75 / 255
    input_std = [57.375, 57.375, 57.375]  # [0.229, 0.224, 0.225] --> 57.375 / 255

    if mode == 'val':
        cropping = torchvision.transforms.Compose([
            GroupResize(256),
            GroupCenterCrop(224)
        ])
    elif mode == 'test':
        cropping = torchvision.transforms.Compose([
            GroupResize(256),
            GroupFullyConv(256)
            # ConditionedGroupCenterCrop(max_size=808)
        ])
    elif mode == 'train':
        cropping = torchvision.transforms.Compose([
            GroupResize(256),
            GroupRandomCrop(224)
        ])
    else:
        raise ValueError('Mode {} does not exist. Choose between: val, test or train.'.format(mode))

    transforms = torchvision.transforms.Compose([
            cropping,
            GroupToTensorStack(),
            GroupNormalize(mean=input_mean, std=input_std)
        ])

    return transforms
