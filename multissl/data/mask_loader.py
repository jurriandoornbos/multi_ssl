# Copyright 2025 Jurrian Doornbos
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0


import torch
import os
import tifffile as tiff
import numpy as np 

from .loader import tifffile_loader

class SegmentationDataset(torch.utils.data.Dataset):
    """
    Dataset for semantic segmentation with multi-channel (e.g., 4-band) imagery.
    """
    def __init__(self, img_dir, mask_dir, transform=None, mask_transform=None, img_size=224):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.mask_transform = mask_transform
        self.img_size = img_size
        
        # List all files in the image directory
        self.img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(".tif") or f.endswith(".tiff")])
        
        # Check if mask files exist
        for img_file in self.img_files:
            mask_file = self._get_mask_filename(img_file)
            mask_path = os.path.join(mask_dir, mask_file)
            if not os.path.exists(mask_path):
                print(f"Warning: Mask file {mask_path} not found!")
        
    def _get_mask_filename(self, img_filename):
        """Convert image filename to corresponding mask filename based on your naming convention."""
        # This function should be adjusted based on your specific naming convention
        # By default, we assume the mask has the same name but with "_mask" suffix
        return img_filename.replace(".tif", "_mask.tif")
    
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        # Get image file name
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        
        # Get corresponding mask file name
        mask_name = self._get_mask_filename(img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)
        
        # Load image and mask
        image = tifffile_loader(img_path)  # Should be [H, W, C] or [H, W]
        mask = tifffile_loader(mask_path)
        
        # Convert to tensor
        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).long()
        
        # Normalize if needed
        if image.max() > 1.0:
            image = image / 255.0
    
        # Verify mask only contains 0 and 1
        if torch.is_tensor(mask):
            unique_values = torch.unique(mask)
        else:
            unique_values = np.unique(mask)
            
        # If needed, binarize the mask to ensure only 0 and 1
        if len(unique_values) > 2 or unique_values.max() > 1:
            print(f"Warning: Mask at index {idx} has values other than 0 and 1")
            # Convert to binary mask
            if torch.is_tensor(mask):
                mask = (mask > 0).long()  # Convert to 0 and 1
            else:
                mask = (mask > 0).astype(np.int64)
                mask = torch.from_numpy(mask).long()
        
        return image, mask