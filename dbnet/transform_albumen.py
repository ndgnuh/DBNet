from itertools import product
from typing import List

import albumentations as A
import cv2
import numpy as np
from PIL import Image

# from megane.augment.aug_bloom import BloomFilter
# from megane.augment.aug_chromatic_aberration import ChromaticAberration
# from megane.augment.aug_fakelight import FakeLight


def rgb_range(step=10):
    r = range(0, 255, step)
    g = range(0, 255, step)
    b = range(0, 255, step)
    return product(r, g, b)


def CoarseDropout(*a, **kw):
    orig = A.CoarseDropout(*a, **kw)
    patched = A.Lambda(name="CoarseDropoutImg", image=orig.apply)
    return patched


def idendity(**kw):
    return kw


def default_transform(
    prob,
    rotate: bool = False,
    flip: bool = False,
):
    """Returns albumetation transform.

    Args:
        rotate (bool):
            Whether to use rotation augmentation. Default: False.
        flip (bool):
            Whether to use flip augmentation. Default: False.
    """
    transformations = [
        # Cropping related
        A.OneOf(
            [
                A.RandomCropFromBorders(),
                A.CropAndPad(percent=(0.025, 0.25)),
            ],
            p=prob,
        ),
        # Color effects
        A.OneOf(
            [
                A.RandomBrightnessContrast(),
                A.ToGray(),
                A.Equalize(),
                A.ChannelDropout(),
                A.ChannelShuffle(),
                A.FancyPCA(),
                A.ToSepia(),
                A.ColorJitter(),
                A.RandomGamma(),
                A.RGBShift(),
            ],
            p=prob,
        ),
        # Degrade
        A.OneOf(
            [
                A.PixelDropout(),
                A.OneOf(
                    [
                        CoarseDropout(fill_value=color, max_width=32, max_height=32)
                        for color in range(0, 255)
                    ]
                ),
                A.Downscale(interpolation=cv2.INTER_LINEAR),
                A.Blur(),
                A.MedianBlur(),
                A.Posterize(),
                A.Spatter(),
                A.ISONoise(),
                A.MultiplicativeNoise(),
                A.ImageCompression(quality_lower=50),
                A.GaussNoise(),
            ],
            p=prob,
        ),
    ]

    if flip:
        # group: Flipping around
        flip_transform = A.OneOf(
            [
                A.RandomRotate90(p=prob),
                A.VerticalFlip(p=prob),
                A.HorizontalFlip(p=prob),
            ],
            p=prob,
        )
        transformations.append(flip_transform)
        # group: Geometric transform

    if rotate:
        rotate_transform = A.OneOf(
            [
                *[
                    A.Perspective(fit_output=True, pad_val=(r, g, b))
                    for (r, g, b) in rgb_range(10)
                ],
                *[
                    A.Affine(
                        scale=(0.3, 1),
                        rotate=(-180, 180),
                        translate_percent=(0.2, 0.2),
                        shear=(-30, 30),
                        fit_output=True,
                        cval=(r, g, b),
                    )
                    for (r, g, b) in rgb_range(10)
                ],
            ],
            p=prob,
        )
        transformations.append(rotate_transform)

    return A.Compose(transformations)


def get_augment(p=0.3, **kwargs):
    aug = default_transform(prob=p, **kwargs)

    def augment(image, proba_maps, thresh_maps):
        masks = np.concatenate([proba_maps, thresh_maps], axis=0)
        n = len(proba_maps)
        result = aug(image=image, masks=masks)
        image = result["image"]
        proba_maps = result["masks"][:n]
        thresh_maps = result["masks"][n:]
        return image, proba_maps, thresh_maps

    return augment
