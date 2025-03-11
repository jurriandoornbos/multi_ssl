import numpy as np
import torch
import random
import cv2
from torchvision import transforms
import os

from .seg_transforms import ToTensorSafe

class JointTransform:
    """
    Applies the same spatial transformations to both image and mask.
    This ensures that spatial correspondence is maintained between them.
    """
    def __init__(self, img_size=224, scale=(0.5, 1.0), flip_p=0.5):
        self.img_size = img_size
        self.scale = scale
        self.flip_p = flip_p
        
        # Non-spatial transforms for images only
        self.img_transforms = transforms.Compose([
            UIntToFloat(),  # Normalize to [0,1]
        ])
    
    def __call__(self, img, mask):
        # Convert to numpy for processing if they're tensors
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
            
        # Make sure image is in HWC format for processing
        if img.ndim == 3 and img.shape[0] <= 4:  # If in CHW format
            img = np.transpose(img, (1, 2, 0))
        
        # Apply random resized crop to both
        img, mask = self.random_resized_crop(img, mask)
        
        # Apply same random flips to both
        if random.random() < self.flip_p:
            img = np.flip(img, axis=1).copy()  # Horizontal flip
            mask = np.flip(mask, axis=1).copy()
            
        if random.random() < self.flip_p:
            img = np.flip(img, axis=0).copy()  # Vertical flip
            mask = np.flip(mask, axis=0).copy()
        
        # Apply non-spatial transforms to image only
        img = self.img_transforms(img)
        
        # Convert to tensor
        img = torch.from_numpy(img.transpose(2, 0, 1)).float()  # HWC -> CHW
        mask = torch.from_numpy(mask).long()
        
        return img, mask
    
    def random_resized_crop(self, img, mask):
        h, w = img.shape[0], img.shape[1]
        scale_factor = random.uniform(self.scale[0], self.scale[1])
        new_h, new_w = max(1, int(h * scale_factor)), max(1, int(w * scale_factor))
        
        # Get random crop coordinates
        top = random.randint(0, max(0, h - new_h))
        left = random.randint(0, max(0, w - new_w))
        
        # Apply same crop to both image and mask
        img_cropped = img[top:top + new_h, left:left + new_w]
        mask_cropped = mask[top:top + new_h, left:left + new_w]
        
        # Resize both to target size
        img_resized = cv2.resize(img_cropped, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        mask_resized = cv2.resize(mask_cropped, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        
        return img_resized, mask_resized


class UIntToFloat:
    """ Normalize to [0,1]."""
    def __call__(self, img):
        if img.dtype == np.uint8 or img.max() > 1.0:
            img = img.astype(np.float32) / 255.0
        return img


# Modified SemiSupervisedSegmentationDataset with corrected transform handling
class SemiSupervisedSegmentationDataset(torch.utils.data.Dataset):
    """
    Dataset for semi-supervised semantic segmentation that handles both 
    labeled images (with masks) and unlabeled images (without masks).
    With corrected joint transform handling for image and mask.
    """
    def __init__(self, 
                 img_dir, 
                 mask_dir=None, 
                 unlabeled_dir=None, 
                 img_size=224, 
                 joint_transform=None,
                 unlabeled_transform=None):
        super().__init__()
        self.img_size = img_size
        
        # Create default transforms if none provided
        self.joint_transform = joint_transform if joint_transform else JointTransform(img_size=img_size)
        self.unlabeled_transform = unlabeled_transform if unlabeled_transform else transforms.Compose([
            UIntToFloat(),
            ToTensorSafe(),
        ])
        
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
        
        # Report dataset composition
        print(f"Dataset composition:")
        print(f"  - Labeled images: {len(self.labeled_img_paths)}")
        print(f"  - Unlabeled images: {len(self.unlabeled_img_paths)}")
        print(f"  - Total: {len(self)}")
    
    def _get_mask_filename(self, img_filename):
        """Convert image filename to corresponding mask filename"""
        # Adjust based on your specific naming convention
        return img_filename.replace(".tif", "_mask.tif").replace(".tiff", "_mask.tiff")
    
    def __len__(self):
        """Return total number of samples (labeled + unlabeled)"""
        return len(self.labeled_img_paths) + len(self.unlabeled_img_paths)
    
    def __getitem__(self, idx):
        """Get item with fixed joint transform for labeled data"""
        num_labeled = len(self.labeled_img_paths)
        
        # Check if this is a labeled or unlabeled sample
        if idx < num_labeled:
            # Labeled sample
            img_path = self.labeled_img_paths[idx]
            mask_path = self.mask_paths[idx]
            
            # Load image and mask
            from multissl.data.loader import tifffile_loader
            image = tifffile_loader(img_path)
            mask = tifffile_loader(mask_path)
            
            # Verify mask only contains expected values (for binary segmentation)
            unique_values = np.unique(mask)
            if len(unique_values) > 2:  # Assuming binary segmentation (0, 1)
                print(f"Warning: Mask at index {idx} has unexpected values: {unique_values}")
                # Convert to binary mask
                mask = (mask > 0).astype(np.int64)
            
            # Apply JOINT transform to both image and mask together
            # This ensures spatial transformations are applied identically
            image, mask = self.joint_transform(image, mask)
            
            # For labeled data, include both image and mask
            return image, mask, True  # True indicates labeled
            
        else:
            # Unlabeled sample
            unlabeled_idx = idx - num_labeled
            img_path = self.unlabeled_img_paths[unlabeled_idx]
            
            # Load image
            from multissl.data.loader import tifffile_loader
            image = tifffile_loader(img_path)
            
            # Apply transforms for unlabeled data
            if isinstance(self.unlabeled_transform, transforms.Compose):
                # For torchvision transform
                image = torch.from_numpy(image).float()
                if image.max() > 1.0:
                    image = image / 255.0
                
                # Apply transforms 
                image = self.unlabeled_transform(image)
            else:
                # For custom transforms
                image = self.unlabeled_transform(image)
            
            # For unlabeled data, return only the image and a flag
            return image, None, False  # False indicates unlabeled
    
    @property
    def num_classes(self):
        """Return number of classes (inferred from masks if available)"""
        if not self.mask_paths:
            return 2  # Default to binary segmentation
        
        # Load a sample mask to check number of classes
        from multissl.data.loader import tifffile_loader
        sample_mask = tifffile_loader(self.mask_paths[0])
        return len(np.unique(sample_mask))

def semi_supervised_collate_fn(batch):
    """
    Custom collate function for semi-supervised learning that handles 
    both labeled and unlabeled samples in the same batch.
    
    Each item in the batch is (image, mask, is_labeled) where:
    - For labeled data: mask is a tensor, is_labeled is True
    - For unlabeled data: mask is None, is_labeled is False
    
    Returns:
        tuple: (labeled_batch, unlabeled_batch)
            - labeled_batch: (images, masks) or (empty_tensor, empty_tensor) if no labeled samples
            - unlabeled_batch: tensor of unlabeled images or empty tensor if no unlabeled samples
    """
    import torch
    
    # Separate labeled and unlabeled samples
    labeled_samples = [(img, mask) for img, mask, is_labeled in batch if is_labeled and mask is not None]
    unlabeled_samples = [img for img, mask, is_labeled in batch if not is_labeled or mask is None]
    
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