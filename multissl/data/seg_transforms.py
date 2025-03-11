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

# Simple transforms that handle both tensor and numpy inputs
class SafeRandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p
        
    def __call__(self, img):
        if random.random() < self.p:
            if isinstance(img, np.ndarray):
                return np.flip(img, axis=1).copy()
            elif isinstance(img, torch.Tensor):
                return img.flip(dims=[1])
        return img

class SafeRandomVerticalFlip:
    def __init__(self, p=0.5):
        self.p = p
        
    def __call__(self, img):
        if random.random() < self.p:
            if isinstance(img, np.ndarray):
                return np.flip(img, axis=0).copy()
            elif isinstance(img, torch.Tensor):
                return img.flip(dims=[0])
        return img

class SafeRandomResizedCrop:
    def __init__(self, size, scale=(0.5, 1.0)):
        self.size = size
        self.scale = scale
        
    def __call__(self, img):
        # Handle tensor input
        if isinstance(img, torch.Tensor):
            if img.dim() == 3 and img.shape[0] <= 4:  # CxHxW format
                h, w = img.shape[1], img.shape[2]
                is_chw = True
            else:  # HxWxC format
                h, w = img.shape[0], img.shape[1]
                is_chw = False
                
            # Convert to numpy for processing
            img_np = img.cpu().numpy()
            if is_chw:
                img_np = np.transpose(img_np, (1, 2, 0))  # Convert to HxWxC
                
            # Apply crop and resize
            scale_factor = random.uniform(self.scale[0], self.scale[1])
            new_h, new_w = max(1, int(h * scale_factor)), max(1, int(w * scale_factor))
            top = random.randint(0, max(0, h - new_h))
            left = random.randint(0, max(0, w - new_w))
            
            img_cropped = img_np[top:top + new_h, left:left + new_w]
            img_resized = cv2.resize(img_cropped, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
            
            # Convert back to tensor
            result = torch.from_numpy(img_resized)
            if is_chw:
                result = result.permute(2, 0, 1)  # Back to CxHxW
            return result
            
        # Handle numpy input
        elif isinstance(img, np.ndarray):
            h, w = img.shape[0], img.shape[1]
            scale_factor = random.uniform(self.scale[0], self.scale[1])
            new_h, new_w = max(1, int(h * scale_factor)), max(1, int(w * scale_factor))
            top = random.randint(0, max(0, h - new_h))
            left = random.randint(0, max(0, w - new_w))
            
            img_cropped = img[top:top + new_h, left:left + new_w]
            img_resized = cv2.resize(img_cropped, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
            return img_resized
            
        else:
            raise TypeError(f"Unsupported image type: {type(img)}")

class SafeUIntToFloat:
    def __call__(self, img):
        if isinstance(img, np.ndarray):
            if img.dtype == np.uint8 or img.max() > 1.0:
                return img.astype(np.float32) / 255.0
        elif isinstance(img, torch.Tensor):
            if img.max() > 1.0:
                return img.float() / 255.0
        return img

class SafeNormalize:
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std
        
    def __call__(self, img):
        if isinstance(img, np.ndarray):
            return (img - np.array(self.mean)) / np.array(self.std)
        elif isinstance(img, torch.Tensor):
            if img.dim() == 3:
                for t, m, s in zip(img, self.mean, self.std):
                    t.sub_(m).div_(s)
            else:
                img = img.sub(torch.tensor(self.mean)).div(torch.tensor(self.std))
            return img
        return img

class SafeGaussianBlur:
    def __init__(self, kernel_size=5):
        self.kernel_size = kernel_size
        
    def __call__(self, img):
        if isinstance(img, np.ndarray):
            return cv2.GaussianBlur(img, (self.kernel_size, self.kernel_size), 0)
        elif isinstance(img, torch.Tensor):
            # Convert to numpy, apply blur, convert back
            is_chw = (img.dim() == 3 and img.shape[0] <= 4)
            img_np = img.cpu().numpy()
            
            if is_chw:
                img_np = np.transpose(img_np, (1, 2, 0))
                
            blurred = cv2.GaussianBlur(img_np, (self.kernel_size, self.kernel_size), 0)
            
            if is_chw:
                blurred = np.transpose(blurred, (2, 0, 1))
                
            return torch.from_numpy(blurred)
        return img

class SafeGaussianNoise:
    def __init__(self, mean=0, std=0.1):
        self.mean = mean
        self.std = std
        
    def __call__(self, img):
        if isinstance(img, np.ndarray):
            noise = np.random.normal(self.mean, self.std, img.shape).astype(img.dtype)
            noisy_img = img + noise
            return np.clip(noisy_img, 0, 1)
        elif isinstance(img, torch.Tensor):
            noise = torch.randn_like(img) * self.std + self.mean
            noisy_img = img + noise
            return torch.clamp(noisy_img, 0, 1)
        return img

class CustomChannelDropout:
    def __init__(self, drop_prob=0.2, channels_to_drop=1):
        self.drop_prob = drop_prob
        self.channels_to_drop = channels_to_drop
        
    def __call__(self, img):
        if random.random() >= self.drop_prob:
            return img
            
        if isinstance(img, np.ndarray):
            # Determine if it's a multi-channel image
            if img.ndim < 3 or img.shape[2] <= 1:
                return img
                
            # Get number of channels
            num_channels = img.shape[2]
            # Ensure we don't try to drop more channels than available
            n_drop = min(self.channels_to_drop, num_channels - 1)
            
            # Select channels to drop
            channels = random.sample(range(num_channels), n_drop)
            
            # Make a copy to avoid modifying the original
            result = img.copy()
            
            # Apply dropout with random intensity
            for ch in channels:
                dropout_factor = random.uniform(0.3, 1.0)
                result[:, :, ch] = result[:, :, ch] * (1 - dropout_factor)
                
            return result
            
        elif isinstance(img, torch.Tensor):
            # Handle different tensor formats
            if img.dim() == 2:  # Single channel
                return img
                
            if img.dim() == 3:
                if img.shape[0] <= 4:  # Assume CxHxW format
                    num_channels = img.shape[0]
                    n_drop = min(self.channels_to_drop, num_channels - 1)
                    channels = random.sample(range(num_channels), n_drop)
                    
                    result = img.clone()
                    for ch in channels:
                        dropout_factor = random.uniform(0.3, 1.0)
                        result[ch] = result[ch] * (1 - dropout_factor)
                        
                else:  # Assume HxWxC format
                    num_channels = img.shape[2]
                    n_drop = min(self.channels_to_drop, num_channels - 1)
                    channels = random.sample(range(num_channels), n_drop)
                    
                    result = img.clone()
                    for ch in channels:
                        dropout_factor = random.uniform(0.3, 1.0)
                        result[:, :, ch] = result[:, :, ch] * (1 - dropout_factor)
                        
                return result
                
        return img

class ToTensorSafe:
    def __call__(self, img):
        if isinstance(img, np.ndarray):
            # Handle different numpy array formats
            if img.ndim == 3 and img.shape[2] <= 4:  # HxWxC format
                img = np.transpose(img, (2, 0, 1))  # Convert to CxHxW
            return torch.from_numpy(img).float()
        elif isinstance(img, torch.Tensor):
            return img.float()
        else:
            raise TypeError(f"Unsupported image type: {type(img)}")

class WeakAugmentation:
    """Weak augmentation pipeline for the teacher model"""
    def __init__(self, img_size=224, in_channels=4):
        self.img_size = img_size
        self.in_channels = in_channels
        
        self.transform = transforms.Compose([
            SafeUIntToFloat(),
            SafeRandomResizedCrop(size=img_size, scale=(0.8, 1.0)),
            SafeRandomHorizontalFlip(p=0.5),
            SafeRandomVerticalFlip(p=0.5),
            ToTensorSafe()
        ])
        
    def __call__(self, img):
        return self.transform(img)

class StrongAugmentation:
    """Strong augmentation pipeline for the student model"""
    def __init__(self, img_size=224, in_channels=4):
        self.img_size = img_size
        self.in_channels = in_channels
        
        self.transform = transforms.Compose([
            SafeUIntToFloat(),
            SafeRandomResizedCrop(size=img_size, scale=(0.8, 1.0)),
            SafeRandomHorizontalFlip(p=0.5),
            SafeRandomVerticalFlip(p=0.5),
            SafeGaussianBlur(kernel_size=5),
            SafeGaussianNoise(std=0.1),
            CustomChannelDropout(drop_prob=0.1, channels_to_drop=1),
            ToTensorSafe()
        ])
        
    def __call__(self, img):
        return self.transform(img)

class MaskTransform:
    """Basic transform for segmentation masks"""
    def __call__(self, mask):
        if isinstance(mask, np.ndarray):
            return torch.from_numpy(mask).long()
        elif isinstance(mask, torch.Tensor):
            return mask.long()
        else:
            raise TypeError(f"Unsupported mask type: {type(mask)}")

class WeakStrongAugmentation:
    """Combined weak and strong augmentation for semi-supervised learning"""
    def __init__(self, img_size=224, in_channels=4, strong_aug_p=0.8):
        self.img_size = img_size
        self.in_channels = in_channels
        self.strong_aug_p = strong_aug_p
        
        self.weak_transform = WeakAugmentation(img_size, in_channels)
        self.strong_transform = StrongAugmentation(img_size, in_channels)
        self.mask_transform = MaskTransform()
    
    def __call__(self, img, mask=None):
        # Apply weak augmentation
        weak_aug = self.weak_transform(img)
        
        # Apply strong augmentation with probability
        if random.random() < self.strong_aug_p:
            strong_aug = self.strong_transform(img)
        else:
            strong_aug = weak_aug.clone()
        
        # Handle mask if provided
        if mask is not None:
            aug_mask = self.mask_transform(mask)
            return (weak_aug, aug_mask), strong_aug
        
        return weak_aug, strong_aug

class SupervisedAugmentation:
    """Augmentation for supervised training with image and mask"""
    def __init__(self, img_size=224, in_channels=4, strong_aug_p=0.8):
        self.img_size = img_size
        self.in_channels = in_channels
        
        self.weak_transforms = WeakAugmentation(img_size, in_channels)
        self.mask_transforms = MaskTransform()
        
        # Keep a reference to strong transforms (not directly used)
        self.strong_transforms = StrongAugmentation(img_size, in_channels)
    
    def __call__(self, img, mask=None):
        aug_img = self.weak_transforms(img)
        
        if mask is not None:
            aug_mask = self.mask_transforms(mask)
            return aug_img, aug_mask
        
        return aug_img

def create_mean_teacher_augmentations(img_size=224, in_channels=4, strong_aug_p=0.8, supervised=False):
    """Factory function to create appropriate augmentation pipeline"""
    if supervised:
        return SupervisedAugmentation(img_size, in_channels, strong_aug_p)
    else:
        return WeakStrongAugmentation(img_size, in_channels, strong_aug_p)