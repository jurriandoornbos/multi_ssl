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


class SafeRandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p
        
    def __call__(self, img):
        if random.random() < self.p:
            if isinstance(img, np.ndarray):
                return np.flip(img, axis=1).copy()
            elif isinstance(img, torch.Tensor):
                # Handle tensor properly
                return img.flip(dims=[1])
            else:
                # For unknown types, return as is
                return img
        return img

class SafeRandomVerticalFlip:
    def __init__(self, p=0.5):
        self.p = p
        
    def __call__(self, img):
        if random.random() < self.p:
            if isinstance(img, np.ndarray):
                return np.flip(img, axis=2).copy()
            elif isinstance(img, torch.Tensor):
                # Handle tensor properly
                return img.flip(dims=[2])
            else:
                # For unknown types, return as is
                return img
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
            
            # Use numpy interpolation instead of cv2
            img_resized = self._resize_numpy(img_cropped, (self.size, self.size))
            
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
            img_resized = self._resize_numpy(img_cropped, (self.size, self.size))
            return img_resized
            
        else:
            raise TypeError(f"Unsupported image type: {type(img)}")
            
    def _resize_numpy(self, img, size):
        """
        Fast image resize using NumPy vectorized operations.
        
        Args:
            img: numpy array of shape (H, W, C)
            size: tuple of (target_height, target_width)
            
        Returns:
            Resized image as numpy array
        """
        src_h, src_w = img.shape[:2]
        target_h, target_w = size
        
        # Calculate ratios
        h_ratio = src_h / target_h
        w_ratio = src_w / target_w
        
        # Create coordinate grids for the target image
        y_coords = np.arange(target_h).reshape(-1, 1) * h_ratio
        x_coords = np.arange(target_w).reshape(1, -1) * w_ratio
        
        # Floor and ceiling coordinates
        y0 = np.floor(y_coords).astype(np.int32)
        y1 = np.minimum(y0 + 1, src_h - 1)
        x0 = np.floor(x_coords).astype(np.int32)
        x1 = np.minimum(x0 + 1, src_w - 1)
        
        # Calculate interpolation weights
        y_weights = (y_coords - y0).astype(np.float32)
        x_weights = (x_coords - x0).astype(np.float32)
        
        # Reshape for broadcasting
        y_weights = y_weights.reshape(target_h, 1, 1)
        x_weights = x_weights.reshape(1, target_w, 1)
        
        # Get the four corner values for bilinear interpolation
        # Access all pixels at once using advanced indexing
        img_flat = img.reshape(-1, img.shape[2])
        
        # Flattened indices for the four corners
        top_left_idx = y0 * src_w + x0
        top_right_idx = y0 * src_w + x1
        bottom_left_idx = y1 * src_w + x0
        bottom_right_idx = y1 * src_w + x1
        
        # Get values for the four corners
        top_left = img_flat[top_left_idx.flatten()].reshape(target_h, target_w, -1)
        top_right = img_flat[top_right_idx.flatten()].reshape(target_h, target_w, -1)
        bottom_left = img_flat[bottom_left_idx.flatten()].reshape(target_h, target_w, -1)
        bottom_right = img_flat[bottom_right_idx.flatten()].reshape(target_h, target_w, -1)
        
        # Bilinear interpolation
        top = top_left * (1 - x_weights) + top_right * x_weights
        bottom = bottom_left * (1 - x_weights) + bottom_right * x_weights
        result = top * (1 - y_weights) + bottom * y_weights
        
        return result


class SafeUIntToFloat:
    def __call__(self, img):
        if isinstance(img, np.ndarray):
            if img.dtype == np.uint8 or img.max() > 1.0:
                return img.astype(np.float32) / 255.0
        elif isinstance(img, torch.Tensor):
            if img.max() > 1.0:
                return img.float() / 255.0
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

class RandomSpectralShift:
    """Randomly shift spectral bands slightly to simulate sensor noise."""
    def __init__(self, max_shift=0.1, p=0.5):
        self.max_shift = max_shift
        self.p = p

    def __call__(self, img):
        if random.random() >= self.p:
            return img
            
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
        if random.random() >= self.p:
            return img
            
        if isinstance(img, np.ndarray):
            factor = 1 + np.random.uniform(-self.brightness_factor, self.brightness_factor)
            return np.clip(img * factor, 0, 1).astype(np.float32)
        elif isinstance(img, torch.Tensor):
            factor = 1 + torch.FloatTensor(1).uniform_(-self.brightness_factor, self.brightness_factor).item()
            return torch.clamp(img * factor, 0, 1)
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

class MaskTransform:
    """Basic transform for segmentation masks"""
    def __call__(self, mask):
        if isinstance(mask, np.ndarray):
            return torch.from_numpy(mask).long()
        elif isinstance(mask, torch.Tensor):
            return mask.long()
        else:
            raise TypeError(f"Unsupported mask type: {type(mask)}")

class Transpose:
    """ Convert NumPy array (H, W, C) to format suitable for PyTorch (C, H, W) """
    def __call__(self, img):
        if isinstance(img, np.ndarray):
            return img.transpose(2, 0, 1)
        elif isinstance(img, torch.Tensor) and img.dim() == 3 and img.shape[-1] <= 4:
            # Assuming HWC format, convert to CHW
            return img.permute(2, 0, 1)
        return img

class SafeGaussianNoise:
    """
    Fast implementation of Gaussian noise addition that follows the exact pattern:
    noise = np.random.normal(0, self.noise_std, img_np.shape).astype(np.float32)
    img_np = np.clip(img_np + noise, 0, 1)
    
    Adapted to handle both NumPy arrays and PyTorch tensors efficiently.
    """
    def __init__(self, mean=0, std=0.1, p=0.5):
        self.mean = mean
        self.std = std
        self.p = p
    
    def __call__(self, img):
        # Early exit based on probability
        if random.random() >= self.p:
            return img
        
        # Handle NumPy arrays
        if isinstance(img, np.ndarray):
            # Generate noise following the exact pattern requested
            noise = np.random.normal(self.mean, self.std, img.shape).astype(np.float32)
            return np.clip(img + noise, 0, 1)
            
        # Handle PyTorch tensors
        elif isinstance(img, torch.Tensor):
            # Generate equivalent noise for PyTorch tensors
            noise = torch.randn_like(img, dtype=torch.float32) * self.std + self.mean
            return torch.clamp(img + noise, 0, 1)
        
        # Return unchanged for any other type
        return img
    
class JointTransform:
    """
    Applies the same spatial transformations to both image and mask.
    This ensures that spatial correspondence is maintained between them.
    Implements weak or strong augmentation based on input parameter.
    """
    def __init__(self, img_size=224, strong=False):
        self.img_size = img_size
        self.strong = strong
        
        # Configure parameters based on augmentation strength
        if not strong:  # weak
            self.crop_scale = (0.5, 1.0)
            self.flip_p = 0.5
            self.noise_p = 0.0
            self.noise_std = 0.05
            self.channel_dropout_p = 0.0
            self.brightness_factor = 0.1
            self.spectral_shift_p = 0.1
        else:  # strong
            self.crop_scale = (0.2, 1.0)
            self.flip_p = 0.5
            self.noise_p = 0.5
            self.noise_std = 0.1
            self.channel_dropout_p = 0.3
            self.brightness_factor = 0.5
            self.spectral_shift_p = 0.5
    
    def __call__(self, img, mask=None):
        # Convert to numpy for processing if they're tensors
        is_tensor = isinstance(img, torch.Tensor)
        is_chw_format = False
        
        if is_tensor:
            img_np = img.cpu().numpy()
            if img.dim() == 3 and img.shape[0] <= 4:  # CxHxW format
                img_np = np.transpose(img_np, (1, 2, 0))  # Convert to HxWxC
                is_chw_format = True
        else:
            img_np = img
            # Check if numpy array is in CHW format
            if img_np.ndim == 3 and img_np.shape[0] <= 4:
                img_np = np.transpose(img_np, (1, 2, 0))
                is_chw_format = True
            
        if mask is not None and isinstance(mask, torch.Tensor):
            mask_np = mask.cpu().numpy()
        else:
            mask_np = mask
        
        # Ensure image is in float format (0-1)
        if img_np.dtype != np.float32 or img_np.max() > 1.0:
            img_np = img_np.astype(np.float32)
            if img_np.max() > 1.0:
                img_np = img_np / 255.0
        
        # Debug print image shape and values
        h, w = img_np.shape[:2]
        c = img_np.shape[2] if img_np.ndim > 2 else 1
        
        # Random crop with resize
        if mask_np is not None:
            img_np, mask_np = self._joint_random_crop_resize(img_np, mask_np)
        else:
            img_np = self._random_crop_resize(img_np)
        
        # Apply same random flips to both image and mask (if provided)
        if random.random() < self.flip_p:
            # Explicitly checking axes for horizontal flip (axis 1 for HWC format)
            img_np = np.flip(img_np, axis=1).copy()  # Horizontal flip
            if mask_np is not None:
                mask_np = np.flip(mask_np, axis=1).copy()
            
        if random.random() < self.flip_p:
            # Explicitly checking axes for vertical flip (axis 0 for HWC format)
            img_np = np.flip(img_np, axis=0).copy()  # Vertical flip
            if mask_np is not None:
                mask_np = np.flip(mask_np, axis=0).copy()
        
        # Image-only transformations (no need to apply to mask)
        # Add Gaussian noise
        if random.random() < self.noise_p:
            noise = np.random.normal(0, self.noise_std, img_np.shape).astype(np.float32)
            img_np = np.clip(img_np + noise, 0, 1)
            
        # Apply channel dropout
        if img_np.ndim > 2 and random.random() < self.channel_dropout_p and img_np.shape[2] > 1:
            num_channels = img_np.shape[2]
            n_drop = min(1, num_channels - 1)  # Drop at most 1 channel
            channels = random.sample(range(num_channels), n_drop)
            
            for ch in channels:
                dropout_factor = random.uniform(0.3, 1.0)
                img_np[:, :, ch] = img_np[:, :, ch] * (1 - dropout_factor)
                
        # Apply brightness adjustment
        if random.random() < self.noise_p:
            factor = 1 + np.random.uniform(-self.brightness_factor, self.brightness_factor)
            img_np = np.clip(img_np * factor, 0, 1)
            
        # Apply spectral shift
        if img_np.ndim > 2 and random.random() < self.spectral_shift_p and img_np.shape[2] > 1:
            shift = np.random.uniform(-0.1, 0.1, size=(img_np.shape[2],))
            img_np = np.clip(img_np + shift, 0, 1)
        
        # Convert back to tensor
        img_tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).float()  # HWC -> CHW
        
        if mask_np is not None:
            mask_tensor = torch.from_numpy(mask_np).long()
            return img_tensor, mask_tensor
        else:
            return img_tensor, None
    
    def _joint_random_crop_resize(self, img, mask):
        """Apply the same random crop to both image and mask, then resize"""
        h, w = img.shape[:2]
        scale_factor = random.uniform(self.crop_scale[0], self.crop_scale[1])
        new_h, new_w = max(1, int(h * scale_factor)), max(1, int(w * scale_factor))
        
        # Get random crop coordinates
        top = random.randint(0, max(0, h - new_h))
        left = random.randint(0, max(0, w - new_w))
        
        # Apply same crop to both image and mask
        img_cropped = img[top:top + new_h, left:left + new_w]
        mask_cropped = mask[top:top + new_h, left:left + new_w]
        
        # Resize using numpy (no cv2)
        img_resized = self._resize_img(img_cropped, (self.img_size, self.img_size))
        mask_resized = self._resize_mask(mask_cropped, (self.img_size, self.img_size))
        
        return img_resized, mask_resized
    
    def _random_crop_resize(self, img):
        """Apply random crop and resize to a single image"""
        h, w = img.shape[:2]
        scale_factor = random.uniform(self.crop_scale[0], self.crop_scale[1])
        new_h, new_w = max(1, int(h * scale_factor)), max(1, int(w * scale_factor))
        
        # Get random crop coordinates
        top = random.randint(0, max(0, h - new_h))
        left = random.randint(0, max(0, w - new_w))
        
        # Apply crop
        img_cropped = img[top:top + new_h, left:left + new_w]
        
        # Resize
        img_resized = self._resize_img(img_cropped, (self.img_size, self.img_size))
        
        return img_resized
    
    def _resize_img(self, img, size):
        """
        Fast image resize using NumPy vectorized operations.
        
        Args:
            img: numpy array of shape (H, W, C)
            size: tuple of (target_height, target_width)
            
        Returns:
            Resized image as numpy array
        """
        src_h, src_w = img.shape[:2]
        target_h, target_w = size
        
        # Calculate ratios
        h_ratio = src_h / target_h
        w_ratio = src_w / target_w
        
        # Create coordinate grids for the target image
        y_coords = np.arange(target_h).reshape(-1, 1) * h_ratio
        x_coords = np.arange(target_w).reshape(1, -1) * w_ratio
        
        # Floor and ceiling coordinates
        y0 = np.floor(y_coords).astype(np.int32)
        y1 = np.minimum(y0 + 1, src_h - 1)
        x0 = np.floor(x_coords).astype(np.int32)
        x1 = np.minimum(x0 + 1, src_w - 1)
        
        # Calculate interpolation weights
        y_weights = (y_coords - y0).astype(np.float32)
        x_weights = (x_coords - x0).astype(np.float32)
        
        # Reshape for broadcasting
        y_weights = y_weights.reshape(target_h, 1, 1)
        x_weights = x_weights.reshape(1, target_w, 1)
        
        # Get the four corner values for bilinear interpolation
        # Access all pixels at once using advanced indexing
        img_flat = img.reshape(-1, img.shape[2])
        
        # Flattened indices for the four corners
        top_left_idx = y0 * src_w + x0
        top_right_idx = y0 * src_w + x1
        bottom_left_idx = y1 * src_w + x0
        bottom_right_idx = y1 * src_w + x1
        
        # Get values for the four corners
        top_left = img_flat[top_left_idx.flatten()].reshape(target_h, target_w, -1)
        top_right = img_flat[top_right_idx.flatten()].reshape(target_h, target_w, -1)
        bottom_left = img_flat[bottom_left_idx.flatten()].reshape(target_h, target_w, -1)
        bottom_right = img_flat[bottom_right_idx.flatten()].reshape(target_h, target_w, -1)
        
        # Bilinear interpolation
        top = top_left * (1 - x_weights) + top_right * x_weights
        bottom = bottom_left * (1 - x_weights) + bottom_right * x_weights
        result = top * (1 - y_weights) + bottom * y_weights
        
        return result

    def _resize_mask(self, mask, size):
        """
        Fast nearest-neighbor interpolation for masks using NumPy.
        
        Args:
            mask: numpy array of shape (H, W)
            size: tuple of (target_height, target_width)
            
        Returns:
            Resized mask as numpy array
        """
        src_h, src_w = mask.shape[:2]
        target_h, target_w = size
        
        # Calculate coordinates in the source image
        y_ratio = src_h / target_h
        x_ratio = src_w / target_w
        
        y_src = np.floor(np.arange(target_h) * y_ratio).astype(np.int32)
        x_src = np.floor(np.arange(target_w) * x_ratio).astype(np.int32)
        
        # Ensure we don't go out of bounds
        y_src = np.minimum(y_src, src_h - 1)
        x_src = np.minimum(x_src, src_w - 1)
        
        # Use meshgrid to create 2D coordinate arrays
        y_coords, x_coords = np.meshgrid(y_src, x_src, indexing='ij')
        
        # Index into the source mask
        result = mask[y_coords, x_coords]
        
        return result

class JointWeakAugmentation:
    """Weak augmentation pipeline for teacher models"""
    def __init__(self, img_size=224, in_channels=4):
        self.img_size = img_size
        self.in_channels = in_channels
        self.joint_transform = JointTransform(img_size=img_size, strong=False)
        
    def __call__(self, img, mask=None):
        return self.joint_transform(img, mask)

class JointStrongAugmentation:
    """Strong augmentation pipeline for student models"""
    def __init__(self, img_size=224, in_channels=4):
        self.img_size = img_size
        self.in_channels = in_channels
        self.joint_transform = JointTransform(img_size=img_size, strong=True)
        
    def __call__(self, img, mask=None):
        return self.joint_transform(img, mask)

class JointWeakStrongAugmentation:
    """Combined weak and strong augmentation for semi-supervised learning"""
    def __init__(self, img_size=224, in_channels=4, strong_aug_p=0.8):
        self.img_size = img_size
        self.in_channels = in_channels
        self.strong_aug_p = strong_aug_p
        
        # Create transformers for weak and strong augmentation
        self.weak_transform = JointWeakAugmentation(img_size, in_channels)
        self.strong_transform = JointStrongAugmentation(img_size, in_channels)
        
        # Simple mask transform for cases where we don't need joint transformation
        self.mask_transform = MaskTransform()
    
    def __call__(self, img, mask=None):
        # When a mask is provided, we need coordinated transforms
        if mask is not None:
            # Apply weak augmentation to both image and mask (for teacher model)
            weak_img, weak_mask = self.weak_transform(img, mask)
            
            # Apply strong augmentation to the image with probability
            if random.random() < self.strong_aug_p:
                strong_img, _ = self.strong_transform(img, None)  # Using None to avoid mask transform
            else:
                # Create a different weak augmentation for student
                strong_img, _ = self.weak_transform(img, None)
                
            return (weak_img, weak_mask), strong_img
        
        # For unlabeled data (no mask provided)
        else:
            # Get a weakly augmented version for the teacher
            weak_img, _ = self.weak_transform(img, None)
            
            # Get a strongly augmented version for the student with probability
            if random.random() < self.strong_aug_p:
                strong_img, _ = self.strong_transform(img, None)
            else:
                # Create a different weak augmentation for student
                strong_img, _ = self.weak_transform(img, None)
            
            return weak_img, strong_img

class JointSupervisedAugmentation:
    """Augmentation for supervised training with image and mask pairs"""
    def __init__(self, img_size=224, in_channels=4, strong=False):
        self.img_size = img_size
        self.in_channels = in_channels
        self.strong = strong
        
        # Create joint transform for proper image-mask augmentation
        self.joint_transform = JointTransform(img_size=img_size, strong=strong)
    
    def __call__(self, img, mask=None):
        if mask is not None:
            return self.joint_transform(img, mask)
        else:
            augmented_img, _ = self.joint_transform(img, None)
            return augmented_img

        
def create_mean_teacher_augmentations(img_size=224, in_channels=4, strong_aug_p=0.8, supervised=False, strong=False, joint=True):
    """
    Factory function to create appropriate augmentation pipeline
    
    Args:
        img_size: Target image size
        in_channels: Number of input channels (default 4 for multispectral)
        strong_aug_p: Probability of using strong augmentation for semi-supervised learning
        supervised: Whether to create supervised or semi-supervised pipeline
        strong: Whether to use strong augmentation for supervised learning
        joint: Whether to use joint transforms for image-mask pairs or independent image transforms
    """
    from .transforms import get_transform
    if supervised:
        if joint:
            return JointSupervisedAugmentation(img_size, in_channels, strong=strong)
        else:
            # For image-only transforms without masks
            return get_transform(img_size = img_size)
    else:
        return JointWeakStrongAugmentation(img_size, in_channels, strong_aug_p=strong_aug_p)
