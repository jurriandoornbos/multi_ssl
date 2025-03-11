# Copyright 2025 Jurrian Doornbos
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import os
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
from multissl.data.loader import tifffile_loader
from torchvision import transforms


class SemiSupervisedSegmentationDataset(Dataset):
    """
    Dataset for semi-supervised semantic segmentation that can handle both 
    labeled images (with masks) and unlabeled images (without masks).
    """
    def __init__(self, 
                 img_dir, 
                 mask_dir=None, 
                 unlabeled_dir=None, 
                 img_size=224, 
                 transform=None,
                 mask_transform=None):
        super().__init__()
        self.img_size = img_size
        self.transform = transform
        self.mask_transform = mask_transform
        
        # Initialize containers
        self.labeled_img_paths = []
        self.mask_paths = []
        self.unlabeled_img_paths = []
        
        # Process labeled data if available
        if img_dir and mask_dir:
            img_files = sorted([f for f in os.listdir(img_dir) 
                              if f.endswith(".tif") or f.endswith(".tiff")])
            
            for img_file in img_files:
                img_path = os.path.join(img_dir, img_file)
                
                # Get corresponding mask file name
                mask_file = self._get_mask_filename(img_file)
                mask_path = os.path.join(mask_dir, mask_file)
                
                # Only add if mask exists
                if os.path.exists(mask_path):
                    self.labeled_img_paths.append(img_path)
                    self.mask_paths.append(mask_path)
                else:
                    print(f"Warning: Mask file for {img_file} not found. Skipping.")
        
        # Process unlabeled data if available
        if unlabeled_dir:
            self.unlabeled_img_paths = []
            
            # Handle both single directory and list of directories
            unlabeled_dirs = unlabeled_dir if isinstance(unlabeled_dir, list) else [unlabeled_dir]
            
            for u_dir in unlabeled_dirs:
                if not os.path.exists(u_dir):
                    print(f"Warning: Unlabeled directory {u_dir} does not exist. Skipping.")
                    continue
                    
                unlabeled_files = sorted([f for f in os.listdir(u_dir) 
                                         if f.endswith(".tif") or f.endswith(".tiff")])
                self.unlabeled_img_paths.extend([os.path.join(u_dir, f) for f in unlabeled_files])
        
        # Create default transforms if none provided
        if self.transform is None:
            self.transform = self._get_default_transform()
        
        # Report dataset composition
        print(f"Dataset composition:")
        print(f"  - Labeled images: {len(self.labeled_img_paths)}")
        print(f"  - Unlabeled images: {len(self.unlabeled_img_paths)}")
        print(f"  - Total: {len(self)}")
    
    def _get_default_transform(self):
        """Create a default transform pipeline for TIFF images"""
        return transforms.Compose([
            # Add your default transforms here
        ])
    
    def _get_mask_filename(self, img_filename):
        """Convert image filename to corresponding mask filename"""
        # This function should be adjusted based on your specific naming convention
        return img_filename.replace(".tif", "_mask.tif").replace(".tiff", "_mask.tiff")
    
    def __len__(self):
        """Return total number of samples (labeled + unlabeled)"""
        return len(self.labeled_img_paths) + len(self.unlabeled_img_paths)
    
    def __getitem__(self, idx):
        """Get item with special handling for labeled vs unlabeled data"""
        num_labeled = len(self.labeled_img_paths)
        
        # Check if this is a labeled or unlabeled sample
        if idx < num_labeled:
            # Labeled sample
            img_path = self.labeled_img_paths[idx]
            mask_path = self.mask_paths[idx]
            
            # Load image and mask
            image = tifffile_loader(img_path)
            mask = tifffile_loader(mask_path)
            
            # Convert to tensor and normalize
            image = torch.from_numpy(image).float()
            if image.max() > 1.0:
                image = image / 255.0
                
            # Process mask
            mask = torch.from_numpy(mask).long()
            
            # Verify mask only contains expected values (0 and 1 for binary segmentation)
            unique_values = torch.unique(mask)
            if len(unique_values) > self.num_classes:
                print(f"Warning: Mask at index {idx} has unexpected values: {unique_values}")
                # Convert to binary mask if needed
                mask = (mask > 0).long()
            
            # Apply transforms if provided
            if self.transform is not None:
                image = self.transform(image)
            
            if self.mask_transform is not None:
                mask = self.mask_transform(mask)
            
            # For labeled data, include both image and mask
            return image, mask, True  # True indicates labeled
            
        else:
            # Unlabeled sample
            unlabeled_idx = idx - num_labeled
            img_path = self.unlabeled_img_paths[unlabeled_idx]
            
            # Load image
            image = tifffile_loader(img_path)
            
            # Convert to tensor and normalize
            image = torch.from_numpy(image).float()
            if image.max() > 1.0:
                image = image / 255.0
            
            # Apply transforms if provided
            if self.transform is not None:
                image = self.transform(image)
            
            # For unlabeled data, return only the image and a flag
            return image, None, False  # False indicates unlabeled
    
    @property
    def num_classes(self):
        """Return number of classes (inferred from masks if available)"""
        if not self.mask_paths:
            return 2  # Default to binary segmentation
        
        # Load a sample mask to check number of classes
        sample_mask = tifffile_loader(self.mask_paths[0])
        return len(np.unique(sample_mask))


def semi_supervised_collate_fn(batch):
    """
    Custom collate function for semi-supervised learning that handles 
    both labeled and unlabeled samples in the same batch.
    
    Returns:
        tuple: (labeled_batch, unlabeled_batch)
            - labeled_batch: (images, masks) or (empty_tensor, empty_tensor) if no labeled samples
            - unlabeled_batch: tensor of unlabeled images or empty tensor if no unlabeled samples
    """
    # Separate labeled and unlabeled samples
    labeled_samples = [(img, mask) for img, mask, is_labeled in batch if is_labeled]
    unlabeled_samples = [img for img, mask, is_labeled in batch if not is_labeled]
    
    # Process labeled data
    if labeled_samples:
        labeled_images, labeled_masks = zip(*labeled_samples)
        labeled_images = torch.stack(labeled_images)
        labeled_masks = torch.stack(labeled_masks)
    else:
        # Create empty tensors with the correct dimensionality
        # Use the shape from the first element's image to get dimensions
        sample_img = batch[0][0]
        labeled_images = torch.zeros((0, *sample_img.shape), dtype=sample_img.dtype, device=sample_img.device)
        labeled_masks = torch.zeros((0,), dtype=torch.long, device=sample_img.device)
    
    labeled_batch = (labeled_images, labeled_masks)
    
    # Process unlabeled data
    if unlabeled_samples:
        unlabeled_batch = torch.stack(unlabeled_samples)
    else:
        # Create empty tensor with the correct dimensionality
        sample_img = batch[0][0]
        unlabeled_batch = torch.zeros((0, *sample_img.shape), dtype=sample_img.dtype, device=sample_img.device)
    
    return labeled_batch, unlabeled_batch

