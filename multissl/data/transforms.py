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
    ToTensorSafe
)

class RandomSpectralShift:
    """Randomly shift spectral bands slightly to simulate sensor noise."""
    def __init__(self, max_shift=0.1):
        self.max_shift = max_shift

    def __call__(self, img):
        if isinstance(img, np.ndarray):
            shift = np.random.uniform(-self.max_shift, self.max_shift, size=(img.shape[2],))
            return np.clip(img + shift, 0, 1).astype(np.float32)
        elif isinstance(img, torch.Tensor):
            # For tensors in CHW format
            if img.dim() == 3 and img.shape[0] <= 4:
                num_channels = img.shape[0]
                shift = torch.FloatTensor(num_channels).uniform_(-self.max_shift, self.max_shift)
                # Reshape shift to match the tensor dimensions for broadcasting
                shift = shift.reshape(-1, 1, 1).to(img.device)
                return torch.clamp(img + shift, 0, 1)
            else:
                # For tensors in HWC format
                shift = torch.FloatTensor(img.shape[-1]).uniform_(-self.max_shift, self.max_shift)
                return torch.clamp(img + shift, 0, 1)
        return img

class RandomBrightness:
    """Randomly adjust the brightness of the image."""
    def __init__(self, brightness_factor=0.1, p=0.5):
        self.brightness_factor = brightness_factor
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            if isinstance(img, np.ndarray):
                factor = 1 + np.random.uniform(-self.brightness_factor, self.brightness_factor)
                return np.clip(img * factor, 0, 1).astype(np.float32)
            elif isinstance(img, torch.Tensor):
                factor = 1 + torch.FloatTensor(1).uniform_(-self.brightness_factor, self.brightness_factor).item()
                return torch.clamp(img * factor, 0, 1)
        return img

class Transpose:
    """ Convert NumPy array (H, W, C) to format suitable for PyTorch (C, H, W) """
    def __call__(self, img):
        if isinstance(img, np.ndarray):
            return img.transpose(2, 0, 1)
        elif isinstance(img, torch.Tensor) and img.dim() == 3 and img.shape[-1] <= 4:
            # Assuming HWC format, convert to CHW
            return img.permute(2, 0, 1)
        return img

def get_transform(args):
    """
    Builds a transform pipeline for FastSiam using safer transforms
    and adding channel dropout for more robust augmentation.
    """
    img_size = args.input_size
    base = [
        SafeRandomResizedCrop(size=img_size, scale=(0.2, 1)),
        SafeRandomHorizontalFlip(p=0.5),
        SafeRandomVerticalFlip(p=0.5),
    ]
    
    # Add Gaussian blur
    base.append(SafeGaussianBlur(kernel_size=5))
    
    # Add channel dropout for improved robustness
    base.append(CustomChannelDropout(drop_prob=0.2, channels_to_drop=1))
    
    # Add Gaussian noise
    base.append(SafeGaussianNoise(std=0.05))  # Adjusted for 0-1 range
    
    # Convert to float in 0-1 range
    base.append(SafeUIntToFloat())
    
    # Add brightness variation
    base.append(RandomBrightness())
    
    # Add spectral shift
    base.append(RandomSpectralShift())
    
    # Set correct CHW format for tensor
    base.append(Transpose())
    
    # Convert to torch tensor
    base.append(ToTensorSafe())
    
    pipeline = transforms.Compose(base)
    
    return pipeline