# Copyright 2025 Jurrian Doornbos
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import numpy as np
import random
import torch
from torchvision import transforms
import cv2

# Import safe transforms from seg_transforms
from .seg_transforms import (
    SafeRandomHorizontalFlip,
    SafeRandomVerticalFlip,
    SafeRandomResizedCrop,
    SafeUIntToFloat,
    SafeGaussianBlur,
    SafeGaussianNoise,
    CustomChannelDropout,
    ToTensorSafe,
    RandomBrightness,
    RandomSpectralShift
)

class Transpose:
    """ Convert NumPy array (H, W, C) to format suitable for PyTorch (C, H, W) """
    def __call__(self, img):
        if isinstance(img, np.ndarray):
            return img.transpose(2, 0, 1)
        elif isinstance(img, torch.Tensor) and img.dim() == 3 and img.shape[-1] <= 4:
            # Assuming HWC format, convert to CHW
            return img.permute(2, 0, 1)
        return img

def get_transform(args = None, img_size = None, ks =3,std_noise = 0.01, brightness_factor = 0.1, max_shift = 0.2):
    """
    Builds a transform pipeline for FastSiam using safer transforms
    and adding channel dropout for more robust augmentation.
    """
    if args:
        img_size = args.input_size
    elif not args:
        img_size = img_size
    else:
        raise ValueError("Please provide either args or img_size")
    base = [
        SafeRandomResizedCrop(size=img_size, scale=(0.3, 1)),
        SafeRandomHorizontalFlip(p=0.5),
        SafeRandomVerticalFlip(p=0.5),
    ]
    
    # Add Gaussian blur
    base.append(SafeGaussianBlur(kernel_size=ks))
    
    # Add channel dropout for improved robustness
    base.append(CustomChannelDropout(drop_prob=0.2, channels_to_drop=1))
    

    
    # Convert to float in 0-1 range
    base.append(SafeUIntToFloat())
    # Add Gaussian noise
    base.append(SafeGaussianNoise(std=std_noise))  # Adjusted for 0-1 range
    # Add brightness variation
    base.append(RandomBrightness(brightness_factor = brightness_factor))
    
    # Add spectral shift
    base.append(RandomSpectralShift(max_shift = max_shift))
    
    # Set correct CHW format for tensor
    base.append(Transpose())
    
    # Convert to torch tensor
    base.append(ToTensorSafe())
    
    pipeline = transforms.Compose(base)
    
    return pipeline