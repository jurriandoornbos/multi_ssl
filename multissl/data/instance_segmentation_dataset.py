from typing import Optional, Dict, List, Tuple, Union
import random
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from pycocotools.coco import COCO
from PIL import Image

class COCOInstanceSegmentationDataset(Dataset):
    """
    Dataset for loading a COCO image with its segmentation mask
    Adapted to return bboxes in YOLO-style format (list of tensors) with xyxy coordinates
    """
    def __init__(self, img_dir, coco_json_path, instance_dir, transform=None,device=None):
        """
        Args:
            coco_json_path: Path to COCO JSON annotations file
            img_dir: Directory with the images
            instance_dir: Directory with instance mask images
            transform: Optional transform for the image
        """
        self.coco = COCO(coco_json_path)
        self.instance_dir = instance_dir
        self.img_dir = img_dir
        self.transform = transform
        
        # Get all image ids with segmentation annotations
        self.img_ids = list(self.coco.imgs.keys())
        
        # Filter ids to ensure each image has segmentation annotations
        self.img_ids = [img_id for img_id in self.img_ids 
                        if len(self.coco.getAnnIds(imgIds=img_id, iscrowd=None)) > 0]

        # Get all image ids with segmentation annotations
        self.instance_ids = os.listdir(instance_dir)
        
        # Get category information
        self.categories = self.coco.loadCats(self.coco.getCatIds())
        self.category_names = {cat['id']: cat['name'] for cat in self.categories}
        self.num_classes = len(self.categories) + 1  # +1 for background
        if device == None:
            self.device = 'cpu'
        else:
            self.device = device

        
    def __len__(self):
        return len(self.img_ids)
    
    def __getitem__(self, idx):
        """
        Get image and data for a given index
        
        Args:
            idx: Index to retrieve
                
        Returns:
            dict: Contains RGB image and bounding box data in YOLO-style format
        """
        img_id = self.img_ids[idx]
        
        # Load image information
        img_info = self.coco.loadImgs(img_id)[0]
        image_path = os.path.join(self.img_dir, img_info['file_name'])
        
        # Load RGB image
        image = Image.open(image_path).convert("RGB")
        image = np.array(image)  # Convert to numpy for our transforms

        # Load instance mask if available
        if idx < len(self.instance_ids):
            instance_id = self.instance_ids[idx]
            instance_path = os.path.join(self.instance_dir, instance_id)
            instance = Image.open(instance_path).convert("L")
            instance = np.array(instance)  # Convert to numpy
        else:
            # Create an empty instance mask if none is available
            instance = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)
        
        # Create segmentation mask
        semantic_mask = self.create_mask(img_id, img_info['height'], img_info['width'])
        
        # Create initial sample dictionary
        sample = {
            'rgb': image,
            'mask': semantic_mask,
            'instance': instance,
            'img_id': img_id
        }
        
        # Apply transform if provided
        if self.transform:
            sample = self.transform(sample)
        
        # After transforms, prepare instance data and bboxes
        instance_data = self._prepare_instance_tensors_with_boxes(
            semantic_tensor=sample['mask'] if isinstance(sample['mask'], torch.Tensor) 
                        else torch.from_numpy(sample['mask']).long(),
            instance_tensor=sample['instance'] if isinstance(sample['instance'], torch.Tensor)
                        else torch.from_numpy(sample['instance']).long(),
            num_classes=self.num_classes
        )
        sample.update(instance_data)
        
        # Extract just the information we need for YOLO-style training
        return {
            'rgb': sample['rgb'],
            'boxes': instance_data['boxes_yolo_style'],  # List containing single tensor of boxes c,xywh, normalized
            'instance_masks': instance_data['instance_masks'],
            'instance_classes':  instance_data['instance_classes'],
            'mask': instance_data["mask"],
            'img_id': img_id
        }
    
    def create_mask(self, img_id, height, width):
        """
        Create a segmentation mask for a given image
        
        Args:
            img_id: Image ID
            height, width: Dimensions of the image
            
        Returns:
            mask: Segmentation mask where each pixel value corresponds to a class ID
        """
        # Initialize empty mask with zeros (background)
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Get annotations for this image
        ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
        annotations = self.coco.loadAnns(ann_ids)
        
        # Sort annotations by area (to handle overlapping masks)
        annotations = sorted(annotations, key=lambda x: x['area'], reverse=True)
        
        # Add each annotation to the mask
        for ann in annotations:
            # Convert COCO format to binary mask
            binary_mask = self.coco.annToMask(ann)
            
            # Set pixels in the binary mask to corresponding category ID
            # Add 1 to category ID because 0 is reserved for background
            mask[binary_mask == 1] = ann['category_id']
        
        return mask
    
    def _instance_id_tensor_to_binary_masks(self, instance_tensor, ignore_zero=True):
        """
        Convert an instance ID tensor to separate binary masks for each instance
        
        Args:
            instance_tensor: Tensor of shape [H, W] with integer instance IDs
            ignore_zero: Whether to ignore 0 (typically background)
            
        Returns:
            Binary masks tensor of shape [num_instances, H, W]
        """
        # Check if batch dimension exists
        has_batch_dim = instance_tensor.dim() == 3
        
        if has_batch_dim:
            # Remove batch dimension for processing
            instance_tensor = instance_tensor.squeeze(0)
            
        instance_ids = torch.unique(instance_tensor)
        
        # Remove background ID (0) if needed
        if ignore_zero and 0 in instance_ids:
            instance_ids = instance_ids[instance_ids != 0]
        
        # Create binary mask for each instance
        binary_masks = []
        for instance_id in instance_ids:
            binary_mask = (instance_tensor == instance_id)
            binary_masks.append(binary_mask.float())

        return binary_masks
    
    def _create_boxes_from_masks_tensor_xyxy(self, instance_masks):
        """
        Create bounding boxes from instance mask tensors in xyxy format
        
        Args:
            instance_masks: Tensor of shape [num_instances, H, W] with binary masks
            
        Returns:
            Tensor of boxes with shape [num_instances, 4] in format [x1, y1, x2, y2]
        """
        num_instances = len(instance_masks)
        
        boxes = torch.zeros((num_instances, 4), device=self.device)
        
        for i, mask in enumerate(instance_masks):
            # Find non-zero indices
            y_indices, x_indices = torch.where(mask > 0.5)
            if len(y_indices) == 0:
                # Empty mask, add a dummy box
                boxes[i] = torch.tensor([0, 0, 1, 1], device=self.device)
                continue
                
            # Get bounding box coordinates in xyxy format
            x1, x2 = torch.min(x_indices), torch.max(x_indices)
            y1, y2 = torch.min(y_indices), torch.max(y_indices)
            
            # Add box in [x1, y1, x2, y2] format
            boxes[i] = torch.tensor([x1, y1, x2, y2], device=self.device)
        
        return boxes
        
    def _prepare_instance_tensors_with_boxes(self, semantic_tensor, instance_tensor, class_ids_tensor=None, num_classes=2):
        """
        Prepare tensor data for instance segmentation training with YOLO-style bbox format
        
        Args:
            semantic_tensor: Tensor [H, W] with semantic class indices
            instance_tensor: Tensor [H, W] with instance IDs, or [N, H, W] with binary masks
            class_ids_tensor: Tensor of class IDs for each instance, or None to derive from semantic mask
            num_classes: Number of semantic classes
            
        Returns:
            Dictionary with prepared tensors including YOLO-style bbox format
        """
        instance_masks = self._instance_id_tensor_to_binary_masks(instance_tensor)
        num_instances = len(instance_masks)
        h,w = semantic_tensor.shape

        
        # One-hot encode semantic mask
        semantic_one_hot = self._one_hot_encode_tensor(semantic_tensor, num_classes)
        
        # Determine class IDs if not provided
        if class_ids_tensor is None and num_instances > 0:
            # Derive class IDs from semantic mask and instance masks
            class_ids = []
            for instance_mask in instance_masks:
                # Find most common class in this instance region
                masked_semantic = semantic_tensor[instance_mask > 0.5]
                if masked_semantic.numel() > 0:
                    most_common_class = torch.mode(masked_semantic).values
                    class_ids.append(most_common_class)
                else:
                    class_ids.append(torch.tensor(0, device=self.device))
            
            class_ids_tensor = torch.stack(class_ids)
        elif class_ids_tensor is None:
            # No instances, create empty tensor
            class_ids_tensor = torch.zeros(0, dtype=torch.long, device=self.device)
        
        # Create bounding boxes in xyxy format
        boxes_xyxy = self._create_boxes_from_masks_tensor_xyxy(instance_masks)
        boxes_xywh = self._xyxy_to_xywh(boxes_xyxy)
        boxes_norm = self._normalize_boxes(boxes_xywh, format= "xywh", image_size= (h,w))


        #convert them to yolo: class, x,y,w,h (normalized)
        boxes_yolo_style = []
        for box, cls in zip(boxes_norm, class_ids_tensor):
            yolo_style = []
            yolo_style.append(cls)
            yolo_style.extend(box)
            boxes_yolo_style.append(torch.tensor(yolo_style))

        return {
            'mask': semantic_tensor,
            'semantic_one_hot': semantic_one_hot,
            'instance_masks': instance_masks,
            'instance_classes': class_ids_tensor,
            'boxes_xyxy': boxes_xyxy,
            'boxes_yolo_style': boxes_yolo_style
            }


    def _xyxy_to_xywh(self, boxes: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        """
        Convert bounding boxes from XYXY format to XYWH format.
        
        Args:
            boxes: Bounding boxes in XYXY format [N, 4] where each box is [x1, y1, x2, y2]
                Can be either torch.Tensor or numpy.ndarray
                
        Returns:
            Bounding boxes in XYWH format [N, 4] where each box is [x_center, y_center, width, height]
            Same type as input (torch.Tensor or numpy.ndarray)
            
        Examples:
            >>> # PyTorch tensor example
            >>> boxes_xyxy = torch.tensor([[10, 20, 50, 80], [100, 150, 200, 250]])
            >>> boxes_xywh = xyxy_to_xywh(boxes_xyxy)
            >>> print(boxes_xywh)
            tensor([[ 30.,  50.,  40.,  60.],
                    [150., 200., 100., 100.]])
            
            >>> # NumPy array example
            >>> boxes_xyxy_np = np.array([[10, 20, 50, 80], [100, 150, 200, 250]])
            >>> boxes_xywh_np = xyxy_to_xywh(boxes_xyxy_np)
            >>> print(boxes_xywh_np)
            [[ 30.  50.  40.  60.]
            [150. 200. 100. 100.]]
        """
        # Handle empty input
        if len(boxes) == 0:
            if isinstance(boxes, torch.Tensor):
                return torch.zeros((0, 4), dtype=boxes.dtype, device=self.device)
            else:
                return np.zeros((0, 4), dtype=boxes.dtype)
        
        # Ensure input has correct shape
        assert boxes.shape[-1] == 4, f"Expected boxes to have 4 coordinates, got {boxes.shape[-1]}"
        
        # Handle both single box and batch of boxes
        if boxes.ndim == 1:
            # Single box case
            if isinstance(boxes, torch.Tensor):
                x1, y1, x2, y2 = boxes.unbind(-1)
            else:
                x1, y1, x2, y2 = boxes
            
            # Calculate center coordinates and dimensions
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2
            width = x2 - x1
            height = y2 - y1
            
            # Stack results
            if isinstance(boxes, torch.Tensor):
                return torch.stack([x_center, y_center, width, height])
            else:
                return np.array([x_center, y_center, width, height])
        
        else:
            # Batch of boxes case
            if isinstance(boxes, torch.Tensor):
                x1, y1, x2, y2 = boxes.unbind(-1)
            else:
                x1, y1, x2, y2 = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
            
            # Calculate center coordinates and dimensions
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2
            width = x2 - x1
            height = y2 - y1
            
            # Stack results
            if isinstance(boxes, torch.Tensor):
                return torch.stack([x_center, y_center, width, height], dim=-1)
            else:
                return np.stack([x_center, y_center, width, height], axis=-1)


    def _xywh_to_xyxy(self, boxes: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        """
        Convert bounding boxes from XYWH format to XYXY format.
        
        Args:
            boxes: Bounding boxes in XYWH format [N, 4] where each box is [x_center, y_center, width, height]
                Can be either torch.Tensor or numpy.ndarray
                
        Returns:
            Bounding boxes in XYXY format [N, 4] where each box is [x1, y1, x2, y2]
            Same type as input (torch.Tensor or numpy.ndarray)
            
        Examples:
            >>> # PyTorch tensor example
            >>> boxes_xywh = torch.tensor([[30, 50, 40, 60], [150, 200, 100, 100]])
            >>> boxes_xyxy = xywh_to_xyxy(boxes_xywh)
            >>> print(boxes_xyxy)
            tensor([[ 10.,  20.,  50.,  80.],
                    [100., 150., 200., 250.]])
        """
        # Handle empty input
        if len(boxes) == 0:
            if isinstance(boxes, torch.Tensor):
                return torch.zeros((0, 4), dtype=boxes.dtype, device=boxes.device)
            else:
                return np.zeros((0, 4), dtype=boxes.dtype)
        
        # Ensure input has correct shape
        assert boxes.shape[-1] == 4, f"Expected boxes to have 4 coordinates, got {boxes.shape[-1]}"
        
        # Handle both single box and batch of boxes
        if boxes.ndim == 1:
            # Single box case
            if isinstance(boxes, torch.Tensor):
                x_center, y_center, width, height = boxes.unbind(-1)
            else:
                x_center, y_center, width, height = boxes
            
            # Calculate corner coordinates
            x1 = x_center - width / 2
            y1 = y_center - height / 2
            x2 = x_center + width / 2
            y2 = y_center + height / 2
            
            # Stack results
            if isinstance(boxes, torch.Tensor):
                return torch.stack([x1, y1, x2, y2])
            else:
                return np.array([x1, y1, x2, y2])
        
        else:
            # Batch of boxes case
            if isinstance(boxes, torch.Tensor):
                x_center, y_center, width, height = boxes.unbind(-1)
            else:
                x_center, y_center, width, height = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
            
            # Calculate corner coordinates
            x1 = x_center - width / 2
            y1 = y_center - height / 2
            x2 = x_center + width / 2
            y2 = y_center + height / 2
            
            # Stack results
            if isinstance(boxes, torch.Tensor):
                return torch.stack([x1, y1, x2, y2], dim=-1)
            else:
                return np.stack([x1, y1, x2, y2], axis=-1)


    def _normalize_boxes(self, boxes: Union[torch.Tensor, np.ndarray], 
                    image_size: tuple,
                    format: str = 'xyxy') -> Union[torch.Tensor, np.ndarray]:
        """
        Normalize bounding box coordinates to [0, 1] range.
        
        Args:
            boxes: Bounding boxes [N, 4]
            image_size: (height, width) of the image
            format: Either 'xyxy' or 'xywh' to specify box format
            
        Returns:
            Normalized boxes with coordinates in [0, 1] range
        """
        height, width = image_size
        
        if isinstance(boxes, torch.Tensor):
            boxes_norm = boxes.clone()
        else:
            boxes_norm = boxes.copy()
        
        if format == 'xyxy':
            # x1, y1, x2, y2
            boxes_norm[..., 0] /= width   # x1
            boxes_norm[..., 1] /= height  # y1
            boxes_norm[..., 2] /= width   # x2
            boxes_norm[..., 3] /= height  # y2
        elif format == 'xywh':
            # x_center, y_center, width, height
            boxes_norm[..., 0] /= width   # x_center
            boxes_norm[..., 1] /= height  # y_center
            boxes_norm[..., 2] /= width   # width
            boxes_norm[..., 3] /= height  # height
        else:
            raise ValueError(f"Unknown format: {format}. Expected 'xyxy' or 'xywh'")
        
        return boxes_norm
            
    def _one_hot_encode_tensor(self, mask_tensor, num_classes):
        """
        Convert a single-channel class index tensor to one-hot format
        
        Args:
            mask_tensor: Tensor of shape [H, W] or [B, H, W] with integer class indices
            num_classes: Number of classes to encode
            
        Returns:
            One-hot encoded tensor of shape [num_classes, H, W] or [B, num_classes, H, W]
        """
        # Ensure input is long/int tensor for indexing
        mask_tensor = mask_tensor.long()
        
        # Check if batch dimension exists
        has_batch_dim = mask_tensor.dim() == 3
        
        if not has_batch_dim:
            # Add batch dimension for processing
            mask_tensor = mask_tensor.unsqueeze(0)
        
        # Get shape information
        B, H, W = mask_tensor.shape
        
        # Create one-hot tensor [B, num_classes, H, W]
        one_hot = torch.zeros((B, num_classes, H, W), dtype=torch.float32, device=self.device)
        
        # Scatter to fill in the one-hot representation
        one_hot.scatter_(1, mask_tensor.unsqueeze(1), 1.0)
        
        # Remove batch dimension if it wasn't in the input
        if not has_batch_dim:
            one_hot = one_hot.squeeze(0)
        
        return one_hot


def instance_segmentation_collate_fn(batch):
    """
    Collate function for instance segmentation that handles variable numbers of instances
    Similar to YOLO's approach but for instance segmentation data
    
    Args:
        batch: List of samples from the dataset
        
    Returns:
        Batched data with images stacked and boxes as list of tensors
    """
    # Separate the components
    images = []
    boxes_list = []
    instance_masks_list = []
    img_ids = []
    masks = []
    
    for sample in batch:
        images.append(sample['rgb'])
        boxes_list.append(sample['boxes'])  # Each sample['boxes'] is already a list
        instance_masks_list.append(sample['instance_masks'])
        img_ids.append(sample['img_id'])
        masks.append(sample["mask"])    
    
    # Stack images into a batch 
    images = torch.stack(images, dim=0)
    # stack masks into a batch
    masks = torch.stack(masks, dim = 0)
    
    # Keep boxes as list of tensors (YOLO-style)
    # boxes_list is now a list where each element is a tensor of boxes for one image
    
    # Keep instance masks as list (since each image can have different numbers of instances)
    # instance_masks_list remains as a list of tensors
    
    return {
        'rgb': images,
        'boxes': boxes_list,  # List of tensors, one list per image
        'instance_masks': instance_masks_list,  # List of tensors, one  list per image
        'img_ids': img_ids,
        'mask' : masks,
    }

def create_transforms(target_size=(224, 224)):
    """
    Create transforms for image and mask
    
    Args:
        target_size: Target size for resizing
        
    Returns:
        transform: Transform for images
        target_transform: Transform for masks
    """
    # Image transforms
    transform = transforms.Compose([
        transforms.Resize(target_size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
    ])
    
    # Mask transforms
    target_transform = transforms.Compose([
        transforms.Lambda(lambda x: Image.fromarray(x) if isinstance(x, np.ndarray) else x),
        transforms.Resize(target_size, interpolation=transforms.InterpolationMode.NEAREST),
        transforms.Lambda(lambda x: torch.from_numpy(np.array(x)).long()),
    ])
    
    return transform, target_transform


def get_instance_transforms(img_size=224, augment=True):
    """
    Create transforms for instance segmentation tasks that properly handle both images and masks
    
    Args:
        img_size: Target image size (int or tuple)
        augment: Whether to apply data augmentation
        
    Returns:
        transform: Transform for images
        target_transform: Transform for masks
    """
    # Ensure img_size is a tuple
    if isinstance(img_size, int):
        img_size = (img_size, img_size)
    
    # Base image transforms - without normalization so we can visualize easily
    img_transforms = []
    
    # Add augmentation if requested
    if augment:
        img_transforms.extend([
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1)
        ])
    else:
        img_transforms.append(transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BILINEAR))
    
    # Add ToTensor transform
    img_transforms.append(transforms.ToTensor())
    
    # Compile the transform pipeline
    image_transform = transforms.Compose(img_transforms)
    
    # Mask transforms - handle with safer methods
    mask_transforms = []
    
    # Add augmentation if requested - must match spatial transforms of images exactly
    if augment:
        # For masks we need specialized transforms that apply the same spatial transforms
        # We'll implement this using a JointTransform class below
        pass
    else:
        mask_transforms.append(
            transforms.Lambda(lambda x: transforms.functional.resize(
                transforms.ToPILImage()(x) if isinstance(x, torch.Tensor) else Image.fromarray(x.astype(np.uint8)),
                img_size, interpolation=transforms.InterpolationMode.NEAREST))
        )
    
    # Convert to tensor
    mask_transforms.append(transforms.Lambda(lambda x: torch.from_numpy(np.array(x)).long() if not isinstance(x, torch.Tensor) else x))
    
    # Compile the transform pipeline
    mask_transform = transforms.Compose(mask_transforms)
    
    # Return them separately or create a JointTransform
    return image_transform, mask_transform

from .seg_transforms import JointTransform


class InstanceJointTransform:
    """
    Joint transform for instance segmentation that extends JointTransform
    to handle multiple instance masks and instance-specific data.
    """
    def __init__(self, img_size=224, strong=False):
        """
        Args:
            img_size: Target image size
            strong: Whether to use strong augmentation
        """
        self.img_size = img_size
        # Reuse the JointTransform for basic image-mask transforms
        self.joint_transform = JointTransform(img_size=img_size, strong=strong)
    
    def __call__(self, sample):
        """
        Apply transforms to all components of an instance segmentation sample
        
        Args:
            sample: Dictionary containing:
                - 'rgb': RGB image
                - 'mask': Semantic segmentation mask
                - 'instance_masks': Instance masks [N, H, W]
                - Other instance data like boxes, etc.
                
        Returns:
            Transformed sample dictionary
        """
        result = sample.copy()
        
        # Extract base components
        image = sample['rgb']
        semantic_mask = sample.get('mask')
        instance_tensor = sample.get('instance')
        
        # Apply the joint transform to image and semantic mask
        if semantic_mask is not None:
            # Transform image and semantic mask with consistent spatial transforms
            image_trans, mask_trans = self.joint_transform(image, semantic_mask)
            result['rgb'] = image_trans
            result['mask'] = mask_trans
            
            # If instance tensor is provided, apply same transform
            if instance_tensor is not None:
                # Apply the same spatial transform as was applied to the mask
                _, instance_trans = self.joint_transform(image, instance_tensor)
                result['instance'] = instance_trans
        else:
            # Only transform the image if no mask is provided
            image_trans = self.joint_transform(image, None)
            result['rgb'] = image_trans
        
        # If instance masks tensor is directly provided (binary masks)
        if 'instance_masks' in sample and sample['instance_masks'] is not None:
            instance_masks = sample['instance_masks']
            
            # Apply the same transform to each instance mask
            transformed_masks = []
            for i in range(instance_masks.shape[0]):
                mask = instance_masks[i]
                # Use the joint transform function but only keep the mask result
                _, trans_mask = self.joint_transform(image, mask)
                transformed_masks.append(trans_mask)
            
            # Stack transformed masks along first dimension
            if transformed_masks:
                result['instance_masks'] = torch.stack(transformed_masks)
            else:
                # Handle empty case
                result['instance_masks'] = torch.zeros((0, self.img_size, self.img_size),
                                                     device=self.device)
        
        # Transform centers heatmap if provided
        if 'centers' in sample and sample['centers'] is not None:
            # The centers heatmap should be transformed like a mask
            _, centers_trans = self.joint_transform(image, sample['centers'])
            result['centers'] = centers_trans
            
        # Handle boxes if provided (requires special transform logic)
        if 'boxes' in sample and sample['boxes'] is not None:
            # This would need a custom transformation based on the same 
            # random parameters used in joint_transform
            # For now, we'll skip box transformation as it requires access 
            # to the internal transform parameters
            pass
            
        return result

def get_instance_transforms(img_size=224, augment=True):
    """
    Create transforms for instance segmentation tasks using safe transforms
    
    Args:
        img_size: Target image size (int or tuple)
        augment: Whether to apply data augmentation
        
    Returns:
        transform: Transform for instance segmentation samples
    """
    return InstanceJointTransform(img_size=img_size, strong=augment)

 

class InstanceSegJointTransform:
    """
    Joint transform for instance segmentation that applies the same
    spatial transformations to image, semantic mask, and all instance masks.
    Extends the JointTransform approach with support for multiple masks.
    """
    def __init__(self, img_size=224, strong=False):
        """
        Args:
            img_size: Target image size
            strong: Whether to use strong augmentation
        """
        self.img_size = img_size
        self.strong = strong
        
        # Configure parameters based on augmentation strength
        if not strong:  # weak
            self.crop_scale = (0.5, 1.0)
            self.flip_h_p = 0.5
            self.flip_v_p = 0.0
            self.noise_p = 0.0
            self.noise_std = 0.05
            self.channel_dropout_p = 0.0
            self.brightness_factor = 0.1
            self.spectral_shift_p = 0.1
        else:  # strong
            self.crop_scale = (0.2, 1.0)
            self.flip_h_p = 0.5
            self.flip_v_p = 0.3
            self.noise_p = 0.5
            self.noise_std = 0.1
            self.channel_dropout_p = 0.3
            self.brightness_factor = 0.5
            self.spectral_shift_p = 0.5
        
        # Store random transform parameters for consistent application
        self.random_state = {}
    
    def __call__(self, sample):
        """
        Apply transforms to all components of instance segmentation
        
        Args:
            sample: Dictionary with image, masks, and instance data
            
        Returns:
            Transformed sample dictionary
        """
        # Make a copy to avoid modifying original
        result = sample.copy()
        
        # Extract components
        image = sample['rgb']
        semantic_mask = sample.get('mask')
        instance_tensor = sample.get('instance')
        
        # Generate random transform parameters once
        self._generate_random_state(image)
        
        # Apply transforms to image
        result['rgb'] = self._apply_transforms_to_image(image)
        
        # Apply same transforms to semantic mask
        if semantic_mask is not None:
            result['mask'] = self._apply_transforms_to_mask(semantic_mask)
            
        # Apply same transforms to instance tensor
        if instance_tensor is not None:
            result['instance'] = self._apply_transforms_to_mask(instance_tensor)
        
        # Apply to instance masks if provided
        if 'instance_masks' in sample and sample['instance_masks'] is not None:
            instance_masks = sample['instance_masks']
            
            # Apply transforms to each mask
            transformed_masks = []
            for i in range(instance_masks.shape[0]):
                mask = instance_masks[i]
                transformed_masks.append(self._apply_transforms_to_mask(mask))
            
            if transformed_masks:
                result['instance_masks'] = torch.stack(transformed_masks)
            else:
                # Handle empty case
                result['instance_masks'] = torch.zeros((0, self.img_size, self.img_size), 
                                                     dtype=torch.float32)
        
        # Apply to center heatmap
        if 'centers' in sample and sample['centers'] is not None:
            result['centers'] = self._apply_transforms_to_mask(sample['centers'], is_heatmap=True)
        
        # Transform boxes
        if 'boxes' in sample and sample['boxes'] is not None:
            result['boxes'] = self._transform_boxes(sample['boxes'], 
                                                   image.shape[1:] if isinstance(image, torch.Tensor) 
                                                   else (image.shape[0], image.shape[1]))
        
        return result
    
    def _generate_random_state(self, image):
        """Generate random transformation parameters once for all components"""
        # Reset random state
        self.random_state = {}
        
        # Get image shape - handle both numpy arrays and tensors
        if isinstance(image, torch.Tensor):
            if image.dim() == 3 and image.shape[0] <= 4:  # CHW format
                h, w = image.shape[1], image.shape[2]
            else:  # HWC format
                h, w = image.shape[0], image.shape[1]
        else:
            h, w = image.shape[:2]
        
        # Random crop parameters
        scale_factor = random.uniform(self.crop_scale[0], self.crop_scale[1])
        new_h, new_w = max(1, int(h * scale_factor)), max(1, int(w * scale_factor))
        top = random.randint(0, max(0, h - new_h))
        left = random.randint(0, max(0, w - new_w))
        self.random_state['crop'] = (top, left, new_h, new_w)
        
        # Random flip
        self.random_state['flip_h'] = random.random() < self.flip_h_p
        self.random_state['flip_v'] = random.random() < self.flip_v_p
        
    def _apply_transforms_to_image(self, img):
        """Apply transforms to image with proper handling of tensor/numpy"""
        # Handle different input types
        is_tensor = isinstance(img, torch.Tensor)
        
        if is_tensor:
            # Convert tensor to numpy for processing
            img_np = img.cpu().numpy()
            if img.dim() == 3 and img.shape[0] <= 4:  # CHW format
                img_np = np.transpose(img_np, (1, 2, 0))  # Convert to HWC
            is_chw = img.dim() == 3 and img.shape[0] <= 4
        else:
            img_np = img
            # Check if numpy array is in CHW format
            is_chw = img_np.ndim == 3 and img_np.shape[0] <= 4
            if is_chw:
                img_np = np.transpose(img_np, (1, 2, 0))
            
        # Ensure image is in float format (0-1)
        if img_np.dtype != np.float32 or img_np.max() > 1.0:
            img_np = img_np.astype(np.float32)
            if img_np.max() > 1.0:
                img_np = img_np / 255.0
                img_np = img_np.astype(np.float32)
        
        # Apply random crop
        top, left, new_h, new_w = self.random_state['crop']
        img_np = img_np[top:top + new_h, left:left + new_w]
        
        # Resize to target size
        img_np = self._resize_img(img_np, (self.img_size, self.img_size))
        
        # Apply flips if needed
        if self.random_state['flip_h']:
            img_np = np.flip(img_np, axis=1).copy()
        if self.random_state['flip_v']:
            img_np = np.flip(img_np, axis=0).copy()
        
        # Apply additional image-only transformations
        # Add Gaussian noise
        if random.random() < self.noise_p:
            noise = np.random.normal(0, self.noise_std, img_np.shape).astype(np.float32)
            img_np = np.clip(img_np + noise, 0, 1)
            
        # Apply brightness adjustment
        if random.random() < self.noise_p:
            factor = 1 + np.random.uniform(-self.brightness_factor, self.brightness_factor)
            img_np = np.clip(img_np * factor, 0, 1)
            
        # Apply spectral shift if image has multiple channels
        if img_np.ndim > 2 and random.random() < self.spectral_shift_p and img_np.shape[2] > 1:
            shift = np.random.uniform(-0.1, 0.1, size=(img_np.shape[2],))
            img_np = np.clip(img_np + shift, 0, 1)
        
        # Convert back to tensor if input was tensor
        img_np = np.transpose(img_np, (2,0,1))
        img_tensor = torch.from_numpy(img_np).float()
        return img_tensor


    def _apply_transforms_to_mask(self, mask, is_heatmap=False):
        """Apply same spatial transforms to mask/heatmap with appropriate interpolation"""
        # Handle different input types
        is_tensor = isinstance(mask, torch.Tensor)
        
        if is_tensor:
            mask_np = mask.cpu().numpy()
        else:
            mask_np = mask
            
        # Apply random crop with same parameters as image
        top, left, new_h, new_w = self.random_state['crop']
        mask_np = mask_np[top:top + new_h, left:left + new_w]
        
        # Resize to target size using appropriate method
        if is_heatmap:
            # For heatmaps, use bilinear interpolation to preserve gradients
            mask_np = self._resize_img(mask_np, (self.img_size, self.img_size))
        else:
            # For masks, use nearest neighbor to preserve exact class labels
            mask_np = self._resize_mask(mask_np, (self.img_size, self.img_size))
        
        # Apply flips if needed
        if self.random_state['flip_h']:
            mask_np = np.flip(mask_np, axis=1).copy()
        if self.random_state['flip_v']:
            mask_np = np.flip(mask_np, axis=0).copy()
        
        # Convert back to tensor
        if is_tensor:
            if is_heatmap:
                return torch.from_numpy(mask_np).float()
            else:
                return torch.from_numpy(mask_np).long()
        else:
            return mask_np
    
    def _transform_boxes(self, boxes, original_size):
        """Transform bounding boxes with the same spatial transforms"""
        # Handle tensor input
        is_tensor = isinstance(boxes, torch.Tensor)
        
        if is_tensor:
            boxes_np = boxes.cpu().numpy()
        else:
            boxes_np = boxes
            
        # Get crop parameters and calculate scale factors
        top, left, new_h, new_w = self.random_state['crop']
        h, w = original_size
        
        # Transform each box
        transformed_boxes = []
        for box in boxes_np:
            x1, y1, x2, y2 = box
            
            # Adjust to crop window
            x1 = max(0, x1 - left)
            y1 = max(0, y1 - top) 
            x2 = min(new_w, x2 - left)
            y2 = min(new_h, y2 - top)
            
            # Skip invalid boxes (completely outside crop)
            if x2 <= 0 or y2 <= 0 or x1 >= new_w or y1 >= new_h:
                continue
                
            # Scale to target size
            x1 = x1 * self.img_size / new_w
            y1 = y1 * self.img_size / new_h
            x2 = x2 * self.img_size / new_w
            y2 = y2 * self.img_size / new_h
            
            # Apply horizontal flip if needed
            if self.random_state['flip_h']:
                x1, x2 = self.img_size - x2, self.img_size - x1
                
            # Apply vertical flip if needed
            if self.random_state['flip_v']:
                y1, y2 = self.img_size - y2, self.img_size - y1
            
            transformed_boxes.append([x1, y1, x2, y2])
        
        # Handle empty case
        if not transformed_boxes:
            return torch.zeros((0, 4), dtype=torch.float32) if is_tensor else np.zeros((0, 4))
        
        # Convert back to tensor or numpy
        if is_tensor:
            return torch.tensor(transformed_boxes, dtype=torch.float32, device=self.device)
        else:
            return np.array(transformed_boxes)
    
    def _resize_img(self, img, size):
        """
        Resize image using bilinear interpolation
        Reuse implementation from SafeRandomResizedCrop
        """
        if img.ndim == 2:  # For grayscale images, add channel dimension
            img = img[:, :, np.newaxis]
            
        src_h, src_w = img.shape[:2]
        target_h, target_w = size
        
        # Calculate ratios
        h_ratio = src_h / target_h
        w_ratio = src_w / target_w
        
        # Create coordinate grids
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
        
        # Get values for the four corners
        img_flat = img.reshape(-1, img.shape[2])
        
        # Calculate indices for the four corners
        top_left_idx = y0 * src_w + x0
        top_right_idx = y0 * src_w + x1
        bottom_left_idx = y1 * src_w + x0
        bottom_right_idx = y1 * src_w + x1
        
        # Get values for corners and reshape
        top_left = img_flat[top_left_idx.flatten()].reshape(target_h, target_w, -1)
        top_right = img_flat[top_right_idx.flatten()].reshape(target_h, target_w, -1)
        bottom_left = img_flat[bottom_left_idx.flatten()].reshape(target_h, target_w, -1)
        bottom_right = img_flat[bottom_right_idx.flatten()].reshape(target_h, target_w, -1)
        
        # Bilinear interpolation
        top = top_left * (1 - x_weights) + top_right * x_weights
        bottom = bottom_left * (1 - x_weights) + bottom_right * x_weights
        result = top * (1 - y_weights) + bottom * y_weights
        
        # Handle single channel case
        if result.shape[2] == 1:
            result = result.squeeze(2)
            
        return result
    
    def _resize_mask(self, mask, size):
        """
        Resize mask using nearest neighbor interpolation to preserve class values
        Adaptation of JointTransform._resize_mask
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
    
def get_instance_transforms(img_size=224, augment=True):
    """
    Create transforms for instance segmentation tasks
    
    Args:
        img_size: Target image size (int or tuple)
        augment: Whether to apply data augmentation
        
    Returns:
        transform: Transform for instance segmentation samples
    """
    return InstanceSegJointTransform(img_size=img_size, strong=augment)
