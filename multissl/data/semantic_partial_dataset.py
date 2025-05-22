# Copyright 2025 Your Name
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
from typing import Optional, Dict, List, Tuple, Union
import random
from pathlib import Path
import cv2

from multissl.data.loader import tifffile_loader
from multissl.data.seg_transforms import JointTransform


class MixedSupervisionSegmentationDataset(Dataset):
    """
    Dataset for segmentation with mixed supervision levels:
    - Fully labeled data: Complete segmentation masks for all objects
    - Partially labeled data: Only some objects are labeled, others are unlabeled (but present)
    
    This is useful for semi-supervised learning where you have limited annotation budget.
    """
    def __init__(
        self,
        fully_labeled_dir: Optional[str] = None,
        partially_labeled_dir: Optional[str] = None,
        img_size: int = 224,
        transform=None,
        ignore_index: int = 255,
        num_classes: int = 2,
        image_extensions: Tuple[str] = ('.jpg', '.jpeg', '.png', '.tiff', '.tif'),
        mask_extensions: Tuple[str] = ('.png', '.tiff', '.tif'),
        reduce_dataset: Optional[int] = None,
        balance_supervision: bool = True,
        partial_label_ratio: float = 0.5  # What ratio of partial vs full labels to maintain
    ):
        """
        Args:
            fully_labeled_dir: Path to directory with complete labels
                Structure: fully_labeled_dir/images/*.jpg, fully_labeled_dir/labels/*.png
            partially_labeled_dir: Path to directory with partial labels  
                Structure: partially_labeled_dir/images/*.jpg, partially_labeled_dir/labels/*.png
            img_size: Target image size for resizing
            transform: Optional transform (should handle both image and mask)
            ignore_index: Value to use for unlabeled pixels (typically 255)
            num_classes: Number of classes including background
            image_extensions: Valid image file extensions
            mask_extensions: Valid mask file extensions
            reduce_dataset: Optionally limit dataset size for debugging
            balance_supervision: Whether to balance fully vs partially labeled samples
            partial_label_ratio: Ratio of partial labels to maintain when balancing
        """
        self.img_size = img_size
        self.transform = transform if transform else JointTransform(img_size=img_size, strong=False)
        self.ignore_index = ignore_index
        self.num_classes = num_classes
        self.image_extensions = image_extensions
        self.mask_extensions = mask_extensions
        
        # Initialize sample lists
        self.samples = []
        
        # Load fully labeled samples
        if fully_labeled_dir and os.path.exists(fully_labeled_dir):
            full_samples = self._load_samples(fully_labeled_dir, supervision_type='full')
            print(f"Found {len(full_samples)} fully labeled samples")
            self.samples.extend(full_samples)
        
        # Load partially labeled samples  
        if partially_labeled_dir and os.path.exists(partially_labeled_dir):
            partial_samples = self._load_samples(partially_labeled_dir, supervision_type='partial')
            print(f"Found {len(partial_samples)} partially labeled samples")
            self.samples.extend(partial_samples)
        
        if len(self.samples) == 0:
            raise ValueError("No samples found. Check your directory paths and structure.")
        
        # Balance supervision types if requested
        if balance_supervision and fully_labeled_dir and partially_labeled_dir:
            self.samples = self._balance_supervision_types(partial_label_ratio)
        
        # Reduce dataset size if requested (for debugging)
        if reduce_dataset and reduce_dataset < len(self.samples):
            random.shuffle(self.samples)
            self.samples = self.samples[:reduce_dataset]
        
        print(f"Final dataset size: {len(self.samples)}")
        self._print_supervision_stats()
    
    def _load_samples(self, base_dir: str, supervision_type: str) -> List[Dict]:
        """Load samples from a directory structure"""
        samples = []
        
        images_dir = os.path.join(base_dir, 'images')
        labels_dir = os.path.join(base_dir, 'labels')
        
        if not os.path.exists(images_dir):
            print(f"Warning: Images directory not found: {images_dir}")
            return samples
            
        if not os.path.exists(labels_dir):
            print(f"Warning: Labels directory not found: {labels_dir}")
            return samples
        
        # Get all image files
        image_files = []
        for ext in self.image_extensions:
            image_files.extend(Path(images_dir).glob(f'*{ext}'))
            image_files.extend(Path(images_dir).glob(f'*{ext.upper()}'))
        
        for img_path in sorted(image_files):
            # Find corresponding mask file
            img_stem = img_path.stem
            mask_path = None
            
            for ext in self.mask_extensions:
                potential_mask = Path(labels_dir) / f"{img_stem}{ext}"
                if potential_mask.exists():
                    mask_path = potential_mask
                    break
            
            if mask_path is None or not mask_path.exists():
                print(f"Warning: No mask found for {img_path}")
                continue
            
            samples.append({
                'image_path': str(img_path),
                'mask_path': str(mask_path),
                'supervision_type': supervision_type
            })
        
        return samples
    
    def _balance_supervision_types(self, partial_ratio: float) -> List[Dict]:
        """Balance the dataset to have a specific ratio of partial vs full supervision"""
        full_samples = [s for s in self.samples if s['supervision_type'] == 'full']
        partial_samples = [s for s in self.samples if s['supervision_type'] == 'partial']
        
        if len(full_samples) == 0 or len(partial_samples) == 0:
            return self.samples  # Can't balance if one type is missing
        
        # Calculate target counts
        total_desired = min(len(full_samples) + len(partial_samples), 
                           max(len(full_samples), len(partial_samples)) * 2)
        
        partial_count = int(total_desired * partial_ratio)
        full_count = total_desired - partial_count
        
        # Sample with replacement if needed
        balanced_samples = []
        
        if len(partial_samples) >= partial_count:
            balanced_samples.extend(random.sample(partial_samples, partial_count))
        else:
            # Sample with replacement
            balanced_samples.extend(random.choices(partial_samples, k=partial_count))
        
        if len(full_samples) >= full_count:
            balanced_samples.extend(random.sample(full_samples, full_count))
        else:
            # Sample with replacement
            balanced_samples.extend(random.choices(full_samples, k=full_count))
        
        random.shuffle(balanced_samples)
        return balanced_samples
    
    def _print_supervision_stats(self):
        """Print statistics about supervision types"""
        full_count = sum(1 for s in self.samples if s['supervision_type'] == 'full')
        partial_count = sum(1 for s in self.samples if s['supervision_type'] == 'partial')
        
        print(f"Dataset composition:")
        print(f"  - Fully supervised: {full_count} ({full_count/len(self.samples)*100:.1f}%)")
        print(f"  - Partially supervised: {partial_count} ({partial_count/len(self.samples)*100:.1f}%)")
    
    def _load_image(self, image_path: str) -> np.ndarray:
        """Load image from file path"""
        if image_path.lower().endswith(('.tif', '.tiff')):
            # Use tifffile loader for TIFF images (handles multispectral)
            return tifffile_loader(image_path)
        else:
            # Use PIL for standard image formats
            img = Image.open(image_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            return np.array(img)
    
    def _load_mask(self, mask_path: str) -> np.ndarray:
        """Load mask from file path"""
        if mask_path.lower().endswith(('.tif', '.tiff')):
            mask = tifffile_loader(mask_path)
            # Handle potential multi-channel masks by taking first channel
            if len(mask.shape) > 2:
                mask = mask[:, :, 0] if mask.shape[2] == 1 else mask
        else:
            mask = np.array(Image.open(mask_path))
        
        # Ensure mask is 2D
        if len(mask.shape) > 2:
            mask = mask[:, :, 0]
        
        return mask.astype(np.int64)
    
    def _process_partial_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Use morphological operations to identify uncertain regions.
        
        Regions that are neither clearly background nor clearly foreground
        are marked as ignore regions.
        """
        import cv2
        
        processed_mask = mask.copy()
        
        # Create binary masks for foreground and background
        foreground_mask = (mask > 0)
        unlabeled_mask = (mask == 0)
        
        # Apply morphological operations to identify certain regions
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        
        # Erode foreground to get "definitely foreground" regions
        certain_foreground = cv2.erode(foreground_mask.astype(np.uint8), kernel, iterations=1)
        
        # Dilate foreground to get "possibly foreground" regions
        possible_foreground = cv2.dilate(foreground_mask.astype(np.uint8), kernel, iterations=2)
        
          # Regions that are neither certain foreground nor certain background become ignore
        uncertain_regions = ~(certain_foreground.astype(bool))
                
        # Set uncertain regions to ignore_index
        processed_mask[uncertain_regions] = self.ignore_index
        processed_mask[unlabeled_mask] = self.ignore_index
        
        return processed_mask
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        """Get a sample from the dataset"""
        sample_info = self.samples[idx]
        
        # Load image and mask
        image = self._load_image(sample_info['image_path'])
        mask = self._load_mask(sample_info['mask_path'])
        
        # Process mask based on supervision type
        if sample_info['supervision_type'] == 'partial':
            mask = self._process_partial_mask(mask)
        
        # Apply transforms
        if self.transform:
            image, mask = self.transform(image, mask)
        else:
            # Basic conversion to tensors
            if isinstance(image, np.ndarray):
                image = torch.from_numpy(image.transpose(2, 0, 1)).float()
                if image.max() > 1.0:
                    image = image / 255.0
            
            if isinstance(mask, np.ndarray):
                mask = torch.from_numpy(mask).long()
        
        return {
            'image': image,
            'mask': mask,
            'supervision_type': sample_info['supervision_type'],
            'image_path': sample_info['image_path'],
            'mask_path': sample_info['mask_path']
        }


def mixed_supervision_collate_fn(batch: List[Dict]) -> Dict[str, Union[torch.Tensor, List]]:
    """
    Collate function for mixed supervision dataset.
    
    Groups samples by supervision type for different loss handling.
    
    Args:
        batch: List of samples from MixedSupervisionSegmentationDataset
        
    Returns:
        Dictionary with:
        - 'images': Stacked images tensor [B, C, H, W]
        - 'masks': Stacked masks tensor [B, H, W]  
        - 'supervision_types': List of supervision types
        - 'full_indices': Indices of fully supervised samples in batch
        - 'partial_indices': Indices of partially supervised samples in batch
        - 'image_paths': List of image paths (for debugging)
        - 'mask_paths': List of mask paths (for debugging)
    """
    # Separate components
    images = [item['image'] for item in batch]
    masks = [item['mask'] for item in batch]
    supervision_types = [item['supervision_type'] for item in batch]
    image_paths = [item['image_path'] for item in batch]
    mask_paths = [item['mask_path'] for item in batch]
    
    # Stack tensors
    images_tensor = torch.stack(images, dim=0)
    masks_tensor = torch.stack(masks, dim=0)
    
    # Get indices for different supervision types
    full_indices = [i for i, stype in enumerate(supervision_types) if stype == 'full']
    partial_indices = [i for i, stype in enumerate(supervision_types) if stype == 'partial']
    
    return {
        'images': images_tensor,
        'masks': masks_tensor,
        'supervision_types': supervision_types,
        'full_indices': full_indices,
        'partial_indices': partial_indices,
        'image_paths': image_paths,
        'mask_paths': mask_paths,
        'batch_size': len(batch)
    }

