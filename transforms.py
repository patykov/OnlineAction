import random

import torch
import torchvision
from PIL import Image


class GroupRandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img_group):

        w, h = img_group[0].size
        th, tw = self.size

        out_images = list()

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)

        for img in img_group:
            assert(img.size == w and img.size == h)
            if w == tw and h == th:
                out_images.append(img)
            else:
                out_images.append(img.crop((x1, y1, x1 + tw, y1 + th)))

        return out_images


class GroupCenterCrop(object):

    def __init__(self, size):
        self.worker = torchvision.transforms.CenterCrop(size)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]


class GroupTenCrop(object):

    def __init__(self, size):
        self.worker = torchvision.transforms.TenCrop(size)

    def __call__(self, img_group):
        cropped_imgs = [self.worker(img) for img in img_group]
        reordered_imgs = [[group[i] for group in cropped_imgs] for i in range(10)]
        new_img_group = [img for sublist in reordered_imgs for img in sublist]
        return new_img_group


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
        return 255 * torch.stack(
            [torchvision.transforms.ToTensor()(img) for img in img_group], dim=1)
