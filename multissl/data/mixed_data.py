import os
import torch
import numpy as np
from torch.utils.data import Dataset, Sampler
from PIL import Image
from typing import Dict, Optional, List, Any, Iterator, Tuple
from pathlib import Path
from collections import defaultdict
import random
from glob import glob

from .loader import tifffile_loader

class BalancedSampler(Sampler):
    """
    Custom sampler that balances the dataset by:
    1. Undersampling the majority class (rgb_only)
    2. Oversampling the minority classes (ms_only and aligned)
    3. Ensuring each batch has at least 2 samples from each class when possible
    """
    def __init__(
        self, 
        dataset, 
        batch_size=32, 
        oversample_factor_ms=5, 
        oversample_factor_aligned=20, 
        undersample_rgb=True,
        undersample_factor=None,
        target_ratio=None,
        balance_mode="both"  # "oversample", "undersample", or "both"
    ):
        """
        Args:
            dataset: UAV dataset instance
            batch_size: Size of each batch
            oversample_factor_ms: How many times to oversample MS compared to RGB
            oversample_factor_aligned: How many times to oversample aligned compared to RGB
            undersample_rgb: Whether to undersample the RGB class
            undersample_factor: If set, directly determines how much to undersample RGB
                                (e.g., 0.25 means use 25% of RGB data)
            target_ratio: Target ratio of rgb:ms:aligned samples (e.g., [3, 2, 1])
                         If set, overrides the oversample/undersample factors
            balance_mode: How to balance the dataset:
                         - "oversample": Only oversample minority classes
                         - "undersample": Only undersample majority class
                         - "both": Both oversample minorities and undersample majority
        """
        self.dataset = dataset
        self.batch_size = batch_size
        
        # Get indices for each type of sample
        self.rgb_indices = [i for i, sample in enumerate(dataset.samples) if sample['type'] == 'rgb_only']
        self.ms_indices = [i for i, sample in enumerate(dataset.samples) if sample['type'] == 'ms_only']
        self.aligned_indices = [i for i, sample in enumerate(dataset.samples) if sample['type'] == 'aligned']
        
        # Store original counts for dataset size calculation
        self.rgb_count = len(self.rgb_indices)
        self.ms_count = len(self.ms_indices)
        self.aligned_count = len(self.aligned_indices)
        
        # Store balance mode
        self.balance_mode = balance_mode
        
        # Determine how to balance dataset
        if target_ratio:
            # Calculate factors based on target ratio
            self._calculate_factors_from_ratio(target_ratio)
        else:
            # Use provided factors
            self.oversample_factor_ms = oversample_factor_ms
            self.oversample_factor_aligned = oversample_factor_aligned
            
            # Determine undersample factor if needed
            if undersample_rgb and balance_mode in ["undersample", "both"]:
                if undersample_factor:
                    # Use explicitly provided factor
                    self.undersample_factor = undersample_factor
                else:
                    # Calculate a reasonable undersample factor
                    # Aim to make RGB count comparable to oversampled MS and aligned total
                    total_minority = (self.ms_count * self.oversample_factor_ms + 
                                     self.aligned_count * self.oversample_factor_aligned)
                    self.undersample_factor = min(1.0, max(0.05, total_minority / self.rgb_count))
            else:
                # No undersampling
                self.undersample_factor = 1.0
        
        # Create balanced indices
        self.oversampled_indices = self._create_balanced_indices()
        
    def _calculate_factors_from_ratio(self, target_ratio):
        """Calculate over/undersampling factors based on target ratio"""
        rgb_ratio, ms_ratio, aligned_ratio = target_ratio
        
        # Calculate the ideal counts
        total_samples = self.rgb_count + self.ms_count + self.aligned_count
        total_ratio = rgb_ratio + ms_ratio + aligned_ratio
        
        ideal_rgb = (rgb_ratio / total_ratio) * total_samples
        ideal_ms = (ms_ratio / total_ratio) * total_samples
        ideal_aligned = (aligned_ratio / total_ratio) * total_samples
        
        # Calculate factors to reach those ideals
        self.undersample_factor = ideal_rgb / self.rgb_count if self.rgb_count > 0 else 1.0
        self.oversample_factor_ms = ideal_ms / self.ms_count if self.ms_count > 0 else 1.0
        self.oversample_factor_aligned = ideal_aligned / self.aligned_count if self.aligned_count > 0 else 1.0
        
    def _create_balanced_indices(self):
        """Create list of indices with oversampling for minority classes and undersampling for majority class"""
        # Undersample RGB indices if needed
        if self.balance_mode in ["undersample", "both"] and self.undersample_factor < 1.0:
            # Calculate how many RGB samples to keep
            num_rgb_to_keep = max(2, int(self.rgb_count * self.undersample_factor))
            rgb_indices = random.sample(self.rgb_indices, num_rgb_to_keep)
        else:
            rgb_indices = self.rgb_indices.copy()
        
        # Oversample MS indices if needed
        if self.balance_mode in ["oversample", "both"] and self.oversample_factor_ms > 1.0:
            # Create oversampled MS indices
            ms_indices = []
            for _ in range(int(self.oversample_factor_ms)):
                ms_indices.extend(self.ms_indices)
                
            # Add remaining partial set if needed
            remainder = self.oversample_factor_ms - int(self.oversample_factor_ms)
            if remainder > 0:
                num_extra = int(remainder * len(self.ms_indices))
                if num_extra > 0:
                    ms_indices.extend(random.sample(self.ms_indices, num_extra))
        else:
            ms_indices = self.ms_indices.copy()
        
        # Oversample aligned indices if needed
        if self.balance_mode in ["oversample", "both"] and self.oversample_factor_aligned > 1.0:
            # Create oversampled aligned indices
            aligned_indices = []
            for _ in range(int(self.oversample_factor_aligned)):
                aligned_indices.extend(self.aligned_indices)
                
            # Add remaining partial set if needed
            remainder = self.oversample_factor_aligned - int(self.oversample_factor_aligned)
            if remainder > 0:
                num_extra = int(remainder * len(self.aligned_indices))
                if num_extra > 0:
                    aligned_indices.extend(random.sample(self.aligned_indices, num_extra))
        else:
            aligned_indices = self.aligned_indices.copy()
        
        # Combine all indices
        all_indices = rgb_indices + ms_indices + aligned_indices
        
        # Shuffle indices
        random.shuffle(all_indices)
        
        return all_indices
        
    def _create_oversampled_indices(self):
        """DEPRECATED: Use _create_balanced_indices instead"""
        # Original indices
        rgb_indices = self.rgb_indices.copy()
        
        # Oversample MS indices
        ms_indices = self.ms_indices.copy() * self.oversample_factor_ms
        
        # Oversample aligned indices
        aligned_indices = self.aligned_indices.copy() * self.oversample_factor_aligned
        
        # Combine all indices
        all_indices = rgb_indices + ms_indices + aligned_indices
        
        # Shuffle indices
        random.shuffle(all_indices)
        
        return all_indices
    
    
    def __iter__(self) -> Iterator[int]:
        """Return iterator over indices"""
        # Reshape indices into batches
        batches = []
        
        # Process full batches
        for i in range(0, len(self.oversampled_indices) - self.batch_size + 1, self.batch_size):
            batch = self.oversampled_indices[i:i + self.batch_size]
            
            # Check if batch has at least 2 of each type (if possible)
            types_count = defaultdict(int)
            for idx in batch:
                sample_type = self.dataset.samples[idx]['type']
                types_count[sample_type] += 1
            
            # If batch doesn't meet criteria and we have enough samples of each type,
            # replace some samples to ensure minimum representation
            if ((types_count['rgb_only'] < 2 and len(self.rgb_indices) >= 2) or 
                (types_count['ms_only'] < 2 and len(self.ms_indices) >= 2) or 
                (types_count['aligned'] < 2 and len(self.aligned_indices) >= 2)):
                
                # Make a copy of the batch as a list to modify
                new_batch = list(batch)
                
                # Track how many samples we've added for each underrepresented type
                added_samples = {'rgb_only': 0, 'ms_only': 0, 'aligned': 0}
                
                # First pass: identify how many of each type we need to add
                needed = {
                    'rgb_only': max(0, 2 - types_count['rgb_only']) if len(self.rgb_indices) >= 2 else 0,
                    'ms_only': max(0, 2 - types_count['ms_only']) if len(self.ms_indices) >= 2 else 0,
                    'aligned': max(0, 2 - types_count['aligned']) if len(self.aligned_indices) >= 2 else 0
                }
                
                total_needed = sum(needed.values())
                
                # Only proceed if we need to add samples and the batch isn't already full
                if total_needed > 0:
                    # Calculate how many we need to remove to make room (if any)
                    to_remove = max(0, len(new_batch) + total_needed - self.batch_size)
                    
                    # If we need to remove samples, remove from over-represented classes
                    if to_remove > 0:
                        # Find which type has the most samples
                        most_type = max(types_count, key=types_count.get)
                        
                        # Get indices of this type in the batch
                        indices_to_consider = [i for i, idx in enumerate(new_batch) 
                                              if self.dataset.samples[idx]['type'] == most_type]
                        
                        # Remove samples of the most common type
                        indices_to_remove = random.sample(
                            indices_to_consider, 
                            min(to_remove, len(indices_to_consider))
                        )
                        
                        # Remove the selected indices (in reverse order to avoid index shifting)
                        for i in sorted(indices_to_remove, reverse=True):
                            if len(new_batch) > 2:  # Ensure we don't make batch too small
                                del new_batch[i]
                    
                    # Now add the needed samples for each type
                    for type_name, count in needed.items():
                        if count > 0:
                            if type_name == 'rgb_only' and self.rgb_indices:
                                replacements = random.sample(self.rgb_indices, min(count, len(self.rgb_indices)))
                                new_batch.extend(replacements)
                                added_samples['rgb_only'] += len(replacements)
                                
                            elif type_name == 'ms_only' and self.ms_indices:
                                replacements = random.sample(self.ms_indices, min(count, len(self.ms_indices)))
                                new_batch.extend(replacements)
                                added_samples['ms_only'] += len(replacements)
                                
                            elif type_name == 'aligned' and self.aligned_indices:
                                replacements = random.sample(self.aligned_indices, min(count, len(self.aligned_indices)))
                                new_batch.extend(replacements)
                                added_samples['aligned'] += len(replacements)
                    
                    # Ensure the batch doesn't exceed the batch size
                    if len(new_batch) > self.batch_size:
                        new_batch = new_batch[:self.batch_size]
                    
                    # Update the batch only if we successfully added any samples
                    if sum(added_samples.values()) > 0:
                        batch = new_batch
            
            # Ensure the batch is not empty and has minimum size of 2
            if len(batch) >= 2:
                batches.append(batch)
        
        # Handle any remaining samples if there are enough to form a meaningful batch
        remaining = len(self.oversampled_indices) % self.batch_size
        if remaining >= 2:  # Only add if we have at least 2 samples
            last_batch = self.oversampled_indices[-remaining:]
            batches.append(last_batch)
        
        # Flatten batches and return
        flattened = [idx for batch in batches for idx in batch]
        return iter(flattened)
    
    def __len__(self) -> int:
        """Return length of sampler"""
        return len(self.oversampled_indices)
    
    def get_effective_counts(self) -> Dict[str, int]:
        """
        Get effective counts after over/undersampling
        Returns:
            Dict with counts for each type
        """
        if self.balance_mode in ["undersample", "both"] and self.undersample_factor < 1.0:
            effective_rgb = max(2, int(self.rgb_count * self.undersample_factor))
        else:
            effective_rgb = self.rgb_count
            
        if self.balance_mode in ["oversample", "both"] and self.oversample_factor_ms > 1.0:
            effective_ms = int(self.ms_count * self.oversample_factor_ms)
        else:
            effective_ms = self.ms_count
            
        if self.balance_mode in ["oversample", "both"] and self.oversample_factor_aligned > 1.0:
            effective_aligned = int(self.aligned_count * self.oversample_factor_aligned)
        else:
            effective_aligned = self.aligned_count
            
        return {
            'rgb_only': effective_rgb,
            'ms_only': effective_ms,
            'aligned': effective_aligned,
            'total': effective_rgb + effective_ms + effective_aligned
        }


class MixedUAVDataset(Dataset):
    """
    Minimal dataset for UAV imagery with RGB, MS, and aligned MS-RGB data
    """
    def __init__(
        self,
        root_dir: str,
        transform: Optional[callable] = None,
        mode: str = 'all',  # 'rgb', 'ms', 'aligned', or 'all'
        reduce_by: int = None,
     ):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.mode = mode
        self.reduce_by = reduce_by
        
        # Get file lists based on mode
        self.samples = self._get_samples()
        
        # Calculate class distributions
        self.count_by_type = self.get_class_distribution()
        
    def get_class_distribution(self) -> Dict[str, int]:
        """Get count of samples by type"""
        counts = {
            'rgb_only': 0,
            'ms_only': 0,
            'aligned': 0
        }
        
        for sample in self.samples:
            counts[sample['type']] += 1
            
        return counts
        
    def _get_samples(self) -> List[Dict[str, Path]]:
        """Get list of samples based on mode"""
        samples = []
        
        if self.mode in ['rgb', 'all']:
            rgb_dir = os.path.join(self.root_dir, 'RGB')
            if os.path.exists(rgb_dir):
                if not self.reduce_by:
                    for rgb_file in sorted(glob(os.path.join(rgb_dir, "*.jpg"))):
                        samples.append({'rgb': rgb_file, 'type': 'rgb_only'})
                else: 
                    for rgb_file in sorted(glob(os.path.join(rgb_dir, "*.jpg")))[:self.reduce_by]:
                        samples.append({'rgb': rgb_file, 'type': 'rgb_only'})

        
        if self.mode in ['ms', 'all']:
            ms_dir = os.path.join(self.root_dir,'MS')
            if os.path.exists(ms_dir):
                if not self.reduce_by:
                    for ms_file in sorted(glob(os.path.join(ms_dir, "*.TIF"))):
                        samples.append({'ms': ms_file, 'type': 'ms_only'})
                else:
                    for ms_file in sorted(glob(os.path.join(ms_dir, "*.TIF")))[:self.reduce_by]:
                        samples.append({'ms': ms_file, 'type': 'ms_only'})

        
        if self.mode in ['aligned', 'all']:
            aligned_dir = os.path.join(self.root_dir,"RGBMS")
            
            if os.path.exists(aligned_dir):
                # Get matching pairs based on _RGB.TIF beiong the aligned RGB one
                rgb_files = sorted(glob(os.path.join(aligned_dir, "*_RGB.TIF")))
                all_files = sorted(glob(os.path.join(aligned_dir, "*.TIF")))
                ms_files = list(set(all_files) - set(rgb_files))

                if self.reduce_by:
                    rgb_files = rgb_files[:self.reduce_by]
                    ms_files = ms_files[:self.reduce_by]

                    
                for rgb_file, ms_file in zip(rgb_files, ms_files):
                    samples.append({
                        'rgb': rgb_file,
                        'ms': ms_file,
                        'type': 'aligned'
                    })
        
        return samples
    
    def _load_rgb(self, path: Path) -> torch.Tensor:
        """Load RGB image using PIL"""
        return np.array(Image.open(path))
    
    
    def _load_ms(self, path: Path) -> torch.Tensor:
        """Load MS image using tiffffile loader function"""
        return tifffile_loader(path)  # This function is imported in your original code
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        output = {'type': sample['type']}
        
        # Load RGB if available
        if 'rgb' in sample:
            rgb = self._load_rgb(sample['rgb'])
            output['rgb'] = rgb
            # Apply transforms if any
            if self.transform:
                output["rgb"] = self.transform(output["rgb"])
            
        # Load MS if available
        if 'ms' in sample:
            ms = self._load_ms(sample['ms'])
            h,w,c = ms.shape
            if c==4:
                # Add Green as Blue channel at the beginning
                green = ms[:,:,0]
                # Option 1: Reshape green to be 3D (add a channel dimension)
                green_reshaped = green[:,:,np.newaxis]  # Shape becomes (h, w, 1)
                ms = np.concatenate([green_reshaped, ms], axis=2)  # Concatenate along channel axis

            output['ms'] = ms
            # Apply transforms if any
            if self.transform:
                output["ms"] = self.transform(output["ms"])
            
        return output


def restack_nested(list_of_tensor_lists):
    """
    Restack nested lists of tensors, aggregating by positions
    """
    # Determine the number of tensor positions from the first list
    if not list_of_tensor_lists:
        return []
    
    num_positions = len(list_of_tensor_lists[0])
    
    # Initialize empty lists to collect tensors at each position
    collected_tensors = [[] for _ in range(num_positions)]
    
    # Collect tensors by their position in each inner list
    for tensor_list in list_of_tensor_lists:
        for i, tensor in enumerate(tensor_list):
            collected_tensors[i].append(tensor)
    
    # Stack each collection of tensors
    return [torch.stack(tensor_collection) for tensor_collection in collected_tensors]


def multisensor_views_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, Any]:
    """Collect samples by type for separate processing"""
    result = {
        'rgb_only': {'data': [], 'indices': []},
        'ms_only': {'data': [], 'indices': []},
        'aligned': {'rgb': [], 'ms': [], 'indices': []}
    }
    
    for i, item in enumerate(batch):
        if item['type'] == 'rgb_only':
            result['rgb_only']['data'].append(item['rgb'])
            result['rgb_only']['indices'].append(i)
        elif item['type'] == 'ms_only':
            result['ms_only']['data'].append(item['ms'])
            result['ms_only']['indices'].append(i)
        elif item['type'] == 'aligned':
            result['aligned']['rgb'].append(item['rgb'])
            result['aligned']['ms'].append(item['ms'])
            result['aligned']['indices'].append(i)
    
    # Convert lists to tensors where applicable
    if result['rgb_only']['data']:
        result['rgb_only']['data'] = restack_nested(result['rgb_only']['data'])
    if result['ms_only']['data']:
        result['ms_only']['data'] = restack_nested(result['ms_only']['data'])
    if result['aligned']['rgb']:
        result['aligned']['rgb'] = restack_nested(result['aligned']['rgb'])
        result['aligned']['ms'] = restack_nested(result['aligned']['ms'])
    
    # Add original batch size for reference
    result['batch_size'] = len(batch)
    
    return result


# Example usage
def smote_mixed_dataloader(
    root_dir, 
    batch_size=32, 
    num_workers=4, 
    transform=None,
    balance_mode="both",  # "oversample", "undersample", or "both"
    target_ratio=None,    # e.g., [3, 2, 1] for rgb:ms:aligned
    undersample_factor=None,
    oversample_factor_ms=None,
    oversample_factor_aligned=None
):
    """
    Create a balanced dataloader for the UAV dataset
    
    Args:
        root_dir: Path to dataset root directory
        batch_size: Batch size
        num_workers: Number of worker processes
        transform: Optional transforms to apply
        balance_mode: How to balance the dataset:
                     - "oversample": Only oversample minority classes
                     - "undersample": Only undersample majority class
                     - "both": Both oversample minorities and undersample majority
        target_ratio: Target ratio of rgb:ms:aligned samples (e.g., [3, 2, 1])
                     If set, overrides the oversample/undersample factors
        undersample_factor: If set, directly determines how much to undersample RGB
                            (e.g., 0.25 means use 25% of RGB data)
        oversample_factor_ms: How many times to oversample MS compared to RGB
        oversample_factor_aligned: How many times to oversample aligned compared to RGB
        
    Returns:
        PyTorch DataLoader with balanced sampling and epoch size information
    """
    from torch.utils.data import DataLoader
    
    # Create dataset
    dataset = MixedUAVDataset(root_dir=root_dir, transform=transform)
    
    # Get dataset statistics
    rgb_count = dataset.count_by_type['rgb_only']
    ms_count = dataset.count_by_type['ms_only']
    aligned_count = dataset.count_by_type['aligned']
    
    # Default oversampling factors if not specified
    if oversample_factor_ms is None:
        if rgb_count > 0 and ms_count > 0:
            oversample_factor_ms = max(5, min(20, rgb_count // ms_count))
        else:
            oversample_factor_ms = 5
    
    if oversample_factor_aligned is None:
        if rgb_count > 0 and aligned_count > 0:
            oversample_factor_aligned = max(10, min(50, rgb_count // aligned_count))
        else:
            oversample_factor_aligned = 20
    
    # Create balanced sampler
    sampler = BalancedSampler(
        dataset, 
        batch_size=batch_size,
        oversample_factor_ms=oversample_factor_ms,
        oversample_factor_aligned=oversample_factor_aligned,
        undersample_rgb=(balance_mode in ["undersample", "both"]),
        undersample_factor=undersample_factor,
        target_ratio=target_ratio,
        balance_mode=balance_mode
    )
    
    # Create dataloader with custom collate function
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=multisensor_views_collate_fn,
        pin_memory=True
    )
    
    # Get effective counts after balancing
    effective_counts = sampler.get_effective_counts()
    
    # Calculate dataset size per epoch
    epoch_size = effective_counts['total']
    
    # Ensure epoch size is divisible by batch size
    batches = epoch_size // batch_size
    if batches == 0:
        batches = 1
    epoch_size = batches * batch_size
    
    # Add epoch size information to dataloader
    dataloader.epoch_size = epoch_size
    dataloader.epoch_batches = batches
    dataloader.effective_counts = effective_counts
    
    return dataloader