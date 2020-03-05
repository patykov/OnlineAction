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

    def __repr__(self):
        return '{} (Size: {})'.format(self.__class__.__name__, self.size)


class GroupFullyConv(object):

    def __init__(self, size):
        self.size = size
        self.worker = GroupRandomCrop(size)

    def __call__(self, img_group):
        crops = [self.worker(img_group) for _ in range(3)]
        return [item for sublist in crops for item in sublist]

    def __repr__(self):
        return '{} (Size: {})'.format(self.__class__.__name__, self.size)


class GroupCenterCrop(object):

    def __init__(self, size):
        self.size = size
        self.worker = torchvision.transforms.CenterCrop(size)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]

    def __repr__(self):
        return '{} (Size: {})'.format(self.__class__.__name__, self.size)


class GroupRandomHorizontalFlip(object):

    def __call__(self, img_group):
        v = random.random()
        if v < 0.5:
            ret = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in img_group]
            return ret
        else:
            return img_group

    def __repr__(self):
        return self.__class__.__name__


class GroupNormalize(object):
    def __init__(self, mean, std, num_channels=3):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.sub_(m).div_(s)

        return tensor

    def __repr__(self):
        return '{} (Mean: {}, Std: {})'.format(self.__class__.__name__, self.mean, self.std)


class GroupResize(object):

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.worker = torchvision.transforms.Resize(size, interpolation)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]

    def __repr__(self):
        return '{} (Size: {})'.format(self.__class__.__name__, self.size)


class GroupToTensorStack(object):

    def __call__(self, img_group):
        return torch.stack([torchvision.transforms.ToTensor()(img) for img in img_group], dim=1)

    def __repr__(self):
        return self.__class__.__name__


class GroupRandomResize(object):

    def __init__(self, min_size, max_size):
        self.min = min_size
        self.max = max_size

    def __call__(self, img_group):

        size = random.randint(self.min, self.max)
        worker = torchvision.transforms.Resize(size)

        return [worker(img) for img in img_group]

    def __repr__(self):
        return '{} (Min Size: {}, Max Size: {})'.format(self.__class__.__name__, self.min, self.max)
