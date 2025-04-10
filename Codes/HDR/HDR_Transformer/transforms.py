import logging
import random

import numpy as np


_logger = logging.getLogger(__name__)



class RandomCrop(object):
    def __init__(self, size=64):
        self.size = size

    def __call__(self, img):
        h, w, _ = img.shape
        th, tw = self.size, self.size
        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        img = img[y1:y1 + th, x1:x1 + tw]
        return img


class RandomHorizontalFlip(object):
    def __call__(self, img):
        if random.random() < 0.5:
            img = np.copy(np.fliplr(img))
        return img


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, input):
        for t in self.transforms:
            input = t.apply(input)
        return input


def fetch_transform(params):
    if params.transform_type == "hdr_transformer":
        train_transforms = []
        test_transforms = []

    #_logger.info("Train transforms: {}".format(", ".join([type(t).__name__ for t in train_transforms])))
    #_logger.info("Val and Test transforms: {}".format(", ".join([type(t).__name__ for t in test_transforms])))
    train_transforms = Compose(train_transforms)
    test_transforms = Compose(test_transforms)
    return train_transforms, test_transforms