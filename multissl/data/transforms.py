# Copyright 2025 Jurrian Doornbos
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

# Compose the custom augmentations for 4 channel stuff.
import numpy as np
import random
import torch
from torchvision import transforms
import cv2

class GaussianNoise:
    """Add Gaussian noise to a NumPy array."""
    def __init__(self, mean=0, std=30):  # Adjusted for 0-255 range
        self.mean = mean
        self.std = std

    def __call__(self, img):
        noise = np.random.normal(self.mean, self.std, img.shape).astype(np.float32)
        img = img + noise
        img = np.clip(img, 0, 255)  # Ensure values remain in valid range
        return img

class UIntToFloat:
    """Convert NumPy array (H, W, C) to PyTorch tensor (C, H, W) and normalize to [0,1]."""
    def __call__(self, img):
        img = img.astype(np.float32) / 255.0  # Normalize to [0,1]
        return img.transpose(2,0,1)

class RandomResizedCrop:
    """Crop and resize a random region of the image."""
    def __init__(self, size, scale=(0.8, 1.0)):
        self.size = size
        self.scale = scale

    def __call__(self, img):
        h, w, _ = img.shape
        scale_factor = random.uniform(self.scale[0], self.scale[1])
        new_h, new_w = int(h * scale_factor), int(w * scale_factor)
        top = random.randint(0, h - new_h)
        left = random.randint(0, w - new_w)
        img_cropped = img[top:top + new_h, left:left + new_w]
        img_resized = cv2.resize(img_cropped, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        return img_resized

class RandomHorizontalFlip:
    """Randomly flip the image horizontally."""
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return np.flip(img, axis=1).copy()  # Horizontal flip
        return img

class RandomVerticalFlip:
    """Randomly flip the image vertically."""
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return np.flip(img, axis=0).copy()  # Vertical flip
        return img

class GaussianBlur:
    """Apply Gaussian blur to a NumPy image."""
    def __init__(self, kernel_size=21):
        self.kernel_size = kernel_size

    def __call__(self, img):
        return np.stack([cv2.GaussianBlur(img[..., i], (self.kernel_size, self.kernel_size), 0) for i in range(img.shape[-1])], axis=-1)

class ToTensor(object):
  def __call__(self, np_data):
    return torch.from_numpy(np_data)

def get_transform(args):
    img_size = args.input_size
    base =  [
            RandomResizedCrop(size=img_size, scale=(0.2, 1.0)),  # Modify input_size as needed
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.5),
        ]
    
    if args.gaussian_blur:
        base.append(GaussianBlur(kernel_size=21))

    base.append(GaussianNoise(std=5))  # Adjusted for 0-255 range
    base.append(UIntToFloat()) # Convert to PyTorch tensor & normalize to [0,1])
    base.append(ToTensor())
    pipeline = transforms.Compose(base)
    
    return pipeline