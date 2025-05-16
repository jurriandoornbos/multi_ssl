from typing import Optional, Dict, List, Tuple
import random
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from pycocotools.coco import COCO

class COCOInstanceSegmentationDataset(Dataset):
    """
    Dataset for loading a COCO image with its segmentation mask
    """
    def __init__(self, img_dir, coco_json_path, instance_dir, transform=None):
        """
        Args:
            coco_json_path: Path to COCO JSON annotations file
            img_dir: Directory with the images
            transform: Optional transform for the image
            target_transform: Optional transform for the mask
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
        
    def __len__(self):
        return len(self.img_ids)
    
    def __getitem__(self, idx):
        """
        Get image and mask for a given index
        
        Args:
            idx: Index to retrieve
                
        Returns:
            dict: Contains RGB image and instance segmentation data
        """
        img_id = self.img_ids[idx]
        
        # Load image information
        img_info = self.coco.loadImgs(img_id)[0]
        image_path = os.path.join(self.img_dir, img_info['file_name'])
        
        # Load RGB image
        image = Image.open(image_path).convert("RGB")
        image = np.array(image)  # Convert to numpy for our transforms

        if idx < len(self.instance_ids):
            instance_id = self.instance_ids[idx]
            instance_path = os.path.join(self.instance_dir, instance_id)
            instance = Image.open(instance_path).convert("L")
            instance = np.array(instance)  # Convert to numpy
        else:
            # Create an empty instance mask if none is available
            instance = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)
        
        # Create segmentation mask
        mask = self.create_mask(img_id, img_info['height'], img_info['width'])
        
        # Create initial sample dictionary
        sample = {
            'rgb': image,
            'mask': mask,
            'instance': instance,
            'img_id': img_id
        }
        
        # Apply transform if provided
        if self.transform:
            sample = self.transform(sample)
        
        # After transforms, prepare instance data
        instance_data = self._prepare_instance_tensors(
            semantic_tensor=sample['mask'] if isinstance(sample['mask'], torch.Tensor) 
                        else torch.from_numpy(sample['mask']).long(),
            instance_tensor=sample['instance'] if isinstance(sample['instance'], torch.Tensor)
                        else torch.from_numpy(sample['instance']).long(),
            num_classes=self.num_classes
        )
        sample.update(instance_data)
        
        return sample

    
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

    from matplotlib.colors import ListedColormap
    
    def visualize_instance_segmentation(self,
        image: Optional[torch.Tensor] = None,
        semantic_mask: Optional[torch.Tensor] = None,
        instance_masks: Optional[torch.Tensor] = None,
        class_ids: Optional[torch.Tensor] = None,
        boxes: Optional[torch.Tensor] = None,
        class_names: Optional[List[str]] = None,
        batch: Optional[Dict]= None,
        alpha: float = 0.9,
        figsize: Tuple[int, int] = (16, 12),
        random_colors: bool = True,
        save_path: Optional[str] = None,
        show_boxes: bool = True,
        show_class_labels: bool = True,
        max_instances_to_show: int = 20
    ):
        """
        Visualize instance segmentation results
        
        Args:
            image: Original image tensor [C, H, W] or [H, W, C]
            semantic_mask: Semantic segmentation mask [H, W] or one-hot encoded [C, H, W]
            instance_masks: Instance masks [N, H, W] where N is number of instances
            class_ids: Class IDs for each instance [N]
            boxes: Bounding boxes for instances [N, 4] in format [x1, y1, x2, y2]
            class_names: List of class names for labeling
            alpha: Transparency for masks
            figsize: Figure size for the plot
            random_colors: Whether to use random colors for instances or a fixed colormap
            save_path: Path to save the visualization, if provided
            show_boxes: Whether to show bounding boxes
            show_class_labels: Whether to show class labels
            max_instances_to_show: Maximum number of instances to visualize
        
        Returns:
            Matplotlib figure
        """
        # Create figure and determine the subplot layout
        if batch:
            image = batch['rgb']
            semantic_mask = batch["mask"]
            instance_masks = batch["instance_masks"]
            boxes = batch["boxes"]
        num_rows = 0
        if image is not None:
            num_rows += 1
        if semantic_mask is not None:
            num_rows += 1
        if instance_masks is not None:
            num_rows += 1
        
        if num_rows == 0:
            raise ValueError("At least one of image, semantic_mask, or instance_masks must be provided")
        
        fig, axs = plt.subplots(num_rows, 1, figsize=figsize)
        
        # Convert to single axis if only one row
        if num_rows == 1:
            axs = [axs]
        
        current_row = 0
        
        # Process and display the original image
        if image is not None:
            ax_img = axs[current_row]
            current_row += 1
            
            # Convert torch tensor to numpy and ensure correct shape
            if isinstance(image, torch.Tensor):
                image_np = image.cpu().detach().numpy()
                
                # Check if channels are first dimension
                if image_np.shape[0] == 3 or image_np.shape[0] == 4:  # [C, H, W]
                    image_np = np.transpose(image_np, (1, 2, 0))
                
                # Handle 4 channels (RGBA or RGBD)
                if image_np.shape[2] == 4:
                    image_np = image_np[:, :, :3]  # Take RGB channels
            else:
                image_np = image
                
            # Normalize if needed
            if image_np.max() > 1.0:
                image_np = image_np / 255.0
                
            # Display the image
            ax_img.imshow(image_np)
            ax_img.set_title("Original Image")
            ax_img.axis('off')
        
        # Process and display the semantic segmentation mask
        if semantic_mask is not None:
            ax_sem = axs[current_row]
            current_row += 1
            
            # Convert torch tensor to numpy
            if isinstance(semantic_mask, torch.Tensor):
                semantic_mask_np = semantic_mask.cpu().detach().numpy()
            else:
                semantic_mask_np = semantic_mask
                
            # Check if semantic mask is one-hot encoded
            if semantic_mask_np.ndim == 3:
                if semantic_mask_np.shape[0] > 1:  # [C, H, W] format
                    semantic_mask_np = np.argmax(semantic_mask_np, axis=0)
                else:  # Single channel
                    semantic_mask_np = semantic_mask_np[0]
            
            # Create a colormap for semantic segmentation
            num_classes = int(np.max(semantic_mask_np)) + 1
            colors = plt.cm.get_cmap('tab20', num_classes)
            
            # Display semantic segmentation
            sem_img = ax_sem.imshow(semantic_mask_np, cmap=colors, vmin=0, vmax=num_classes-1)
            ax_sem.set_title("Semantic Segmentation")
            ax_sem.axis('off')
            
            # Add colorbar with class names if provided
            if class_names is not None:
                cbar = plt.colorbar(sem_img, ax=ax_sem, ticks=np.arange(num_classes))
                if len(class_names) >= num_classes:
                    cbar.ax.set_yticklabels(class_names[:num_classes])
        
        # Process and display instance masks
        if instance_masks is not None:
            ax_inst = axs[current_row]
            
            # Convert torch tensor to numpy
            if isinstance(instance_masks, torch.Tensor):
                instance_masks_np = instance_masks.cpu().detach().numpy()
            else:
                instance_masks_np = instance_masks
                
            # Convert class IDs to numpy if provided
            class_ids_np = None
            if class_ids is not None and isinstance(class_ids, torch.Tensor):
                class_ids_np = class_ids.cpu().detach().numpy()
            else:
                class_ids_np = class_ids
                
            # Convert boxes to numpy if provided
            boxes_np = None
            if boxes is not None and isinstance(boxes, torch.Tensor):
                boxes_np = boxes.cpu().detach().numpy()
            else:
                boxes_np = boxes
            
            # Limit number of instances to visualize
            num_instances = min(instance_masks_np.shape[0], max_instances_to_show)
            
            # Set up the plot for instances
            if image is not None:
                # Show instances overlaid on original image
                ax_inst.imshow(image_np)
            else:
                # Create a blank canvas for instances
                blank = np.zeros(instance_masks_np.shape[1:] + (3,), dtype=np.float32)
                ax_inst.imshow(blank)
            
            ax_inst.set_title(f"Instance Segmentation (showing {num_instances} of {instance_masks_np.shape[0]} instances)")
            ax_inst.axis('off')
            
            # Generate random colors for instances
            if random_colors:
                # Generate distinct colors for better visualization
                colors = []
                for i in range(num_instances):
                    # Generate vibrant colors
                    h = random.random()  # Hue
                    s = 0.8 + random.random() * 0.2  # Saturation
                    v = 0.8 + random.random() * 0.2  # Value
                    
                    # Convert HSV to RGB
                    r, g, b = self._hsv_to_rgb(h, s, v)
                    colors.append([r, g, b])
            else:
                # Use a fixed colormap
                cmap = plt.cm.get_cmap('tab20', num_instances)
                colors = [cmap(i)[:3] for i in range(num_instances)]
            
            # Overlay instance masks
            for i in range(num_instances):
                # Get mask and create rgba mask for overlay
                mask = instance_masks_np[i]
                color_mask = np.zeros(mask.shape + (4,), dtype=np.float32)
                
                # Fill the mask with the instance color and set alpha
                color_mask[mask > 0.5, :3] = colors[i]
                color_mask[mask > 0.5, 3] = alpha
                
                # Overlay on the plot
                ax_inst.imshow(color_mask, alpha=alpha)
                
        plt.tight_layout()
        
        # Save the figure if path is provided
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        
        return fig
    
    def visualize_onehot_instances(self,
        image: Optional[torch.Tensor] = None,
        onehot_instances: torch.Tensor = None,
        class_ids: Optional[torch.Tensor] = None,
        class_names: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (16, 12),
        alpha: float = 0.7,
        save_path: Optional[str] = None
    ):
        """
        Visualize one-hot encoded instance masks
        
        Args:
            image: Original image tensor [C, H, W] or [H, W, C]
            onehot_instances: One-hot encoded instance masks [N, H, W]
            class_ids: Class IDs for each instance [N]
            class_names: List of class names for labeling
            figsize: Figure size for the plot
            alpha: Transparency for instance overlays
            save_path: Path to save the visualization
            
        Returns:
            Matplotlib figure
        """
        # Extract the shape of the one-hot encoded instances
        if onehot_instances is None:
            raise ValueError("onehot_instances must be provided")
        
        num_instances, h, w = onehot_instances.shape
        
        # Create the figure
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        # Display the original image if provided, otherwise create a blank canvas
        if image is not None:
            # Convert torch tensor to numpy and ensure correct shape
            if isinstance(image, torch.Tensor):
                image_np = image.cpu().detach().numpy()
                
                # Check if channels are first dimension
                if image_np.shape[0] == 3 or image_np.shape[0] == 4:  # [C, H, W]
                    image_np = np.transpose(image_np, (1, 2, 0))
                
                # Handle 4 channels (RGBA or RGBD)
                if image_np.shape[2] == 4:
                    image_np = image_np[:, :, :3]  # Take RGB channels
            else:
                image_np = image
                
            # Normalize if needed
            if image_np.max() > 1.0:
                image_np = image_np / 255.0
                
            # Display the image
            ax.imshow(image_np)
        else:
            # Create a blank canvas
            blank = np.zeros((h, w, 3), dtype=np.float32)
            ax.imshow(blank)
        
        # Convert onehot instances to numpy
        instances_np = onehot_instances.cpu().detach().numpy()
        
        # Convert class IDs to numpy if provided
        class_ids_np = None
        if class_ids is not None:
            if isinstance(class_ids, torch.Tensor):
                class_ids_np = class_ids.cpu().detach().numpy()
            else:
                class_ids_np = class_ids
        
        # Generate a colormap for instances
        cmap = plt.cm.get_cmap('tab20', num_instances)
        
        # Create legend elements
        legend_elements = []
        
        # Display each instance with a unique color
        for i in range(num_instances):
            # Get instance mask and choose a color
            mask = instances_np[i]
            color = cmap(i)[:3]
            
            # Create RGBA overlay
            mask_color = np.zeros((h, w, 4), dtype=np.float32)
            mask_color[mask > 0.5, :3] = color
            mask_color[mask > 0.5, 3] = alpha
            
            # Overlay instance
            ax.imshow(mask_color)
            
            # Determine class label if available
            label = f"Instance {i+1}"
            if class_ids_np is not None and i < len(class_ids_np):
                class_id = int(class_ids_np[i])
                class_label = str(class_id)
                
                if class_names is not None and class_id < len(class_names):
                    class_label = class_names[class_id]
                    
                label = f"Instance {i+1} ({class_label})"
            
            # Add to legend
            legend_elements.append(plt.Line2D([0], [0], marker='s', color='w', 
                                             markerfacecolor=color, markersize=10, label=label))
            
            # Find the center of mass for the instance
            y_indices, x_indices = np.where(mask > 0.5)
            if len(y_indices) > 0:
                y_center = int(np.mean(y_indices))
                x_center = int(np.mean(x_indices))
                
                # Add instance number at the center
                ax.text(x_center, y_center, str(i+1), color='white', fontsize=10, 
                        ha='center', va='center', weight='bold',
                        bbox=dict(boxstyle="circle,pad=0.3", fc=color, ec='black', alpha=0.8))
        
        # Add legend outside the image
        ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Set title and turn off axis
        ax.set_title(f"Instance Segmentation ({num_instances} instances)")
        ax.axis('off')
        
        plt.tight_layout()
        
        # Save the figure if path is provided
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        
        return fig
        
    def _create_center_heatmap_tensor(self, instance_masks, sigma=2.0):
        """
        Create a center point heatmap from instance mask tensors
        
        Args:
            instance_masks: Tensor of shape [num_instances, H, W] with binary masks
            sigma: Standard deviation for Gaussian peaks
            
        Returns:
            Center heatmap tensor [H, W]
        """
        if instance_masks.shape[0] == 0:
            return torch.zeros((instance_masks.shape[1], instance_masks.shape[2]), 
                            device=instance_masks.device)
        
        num_instances, h, w = instance_masks.shape
        heatmap = torch.zeros((h, w), device=instance_masks.device)
        
        for mask in instance_masks:
            # Find center of mass of the instance
            y_indices, x_indices = torch.where(mask > 0.5)
            if len(y_indices) == 0:
                continue
                
            center_y = torch.mean(y_indices.float()).round().long()
            center_x = torch.mean(x_indices.float()).round().long()
            
            # Create meshgrid for distance calculation
            y = torch.arange(h, device=instance_masks.device)
            x = torch.arange(w, device=instance_masks.device)
            y_grid, x_grid = torch.meshgrid(y, x, indexing='ij')
            
            # Calculate squared distance
            dist_sq = (y_grid - center_y) ** 2 + (x_grid - center_x) ** 2
            
            # Create Gaussian
            gaussian = torch.exp(-dist_sq / (2 * sigma ** 2))
            
            # Add to heatmap (maximum)
            heatmap = torch.maximum(heatmap, gaussian)
        
        return heatmap

    def _create_boxes_from_masks_tensor(self, instance_masks):
        """
        Create bounding boxes from instance mask tensors
        
        Args:
            instance_masks: Tensor of shape [num_instances, H, W] with binary masks
            
        Returns:
            Tensor of boxes with shape [num_instances, 4] in format [x1, y1, x2, y2]
        """
        num_instances = instance_masks.shape[0]
        boxes = torch.zeros((num_instances, 4), device=instance_masks.device)
        
        for i, mask in enumerate(instance_masks):
            # Find non-zero indices
            y_indices, x_indices = torch.where(mask > 0.5)
            if len(y_indices) == 0:
                # Empty mask, add a dummy box
                boxes[i] = torch.tensor([0, 0, 1, 1], device=instance_masks.device)
                continue
                
            # Get bounding box coordinates
            y1, y2 = torch.min(y_indices), torch.max(y_indices)
            x1, x2 = torch.min(x_indices), torch.max(x_indices)
            
            # Add box in [x1, y1, x2, y2] format
            boxes[i] = torch.tensor([x1, y1, x2, y2], device=instance_masks.device)
        
        return boxes
        
        
    def _instance_id_tensor_to_binary_masks(self, instance_tensor, ignore_zero=True):
        """
        Convert an instance ID tensor to separate binary masks for each instance
        
        Args:
            instance_tensor: Tensor of shape [H, W] with integer instance IDs
            ignore_zero: Whether to ignore 0 (typically background)
            
        Returns:
            Binary masks tensor of shape [num_instances, H, W]
        """
        # Get unique instance IDs
        # Check if batch dimension exists
        has_batch_dim = instance_tensor.dim() == 3
        
        if has_batch_dim:
            # Add batch dimension for processing
            instance_tensor = instance_tensor.squeeze(0)
            
        instance_ids = torch.unique(instance_tensor)
        
        # Remove background ID (0) if needed
        if ignore_zero and 0 in instance_ids:
            instance_ids = instance_ids[instance_ids != 0]
        
        # Create binary mask for each instance
        binary_masks = []
        for instance_id in instance_ids:
            binary_mask = (instance_tensor == instance_id).float()
            binary_masks.append(binary_mask)
        
        # Stack masks along first dimension
        if binary_masks:
            return torch.stack(binary_masks, dim=0)
        else:
            # Return empty tensor with correct shape if no instances
            return torch.zeros((0, *instance_tensor.shape), device=instance_tensor.device)

    def _prepare_instance_tensors(self, semantic_tensor, instance_tensor, class_ids_tensor=None, num_classes=2):
        """
        Prepare tensor data for instance segmentation training
        
        Args:
            semantic_tensor: Tensor [H, W] with semantic class indices
            instance_tensor: Tensor [H, W] with instance IDs, or [N, H, W] with binary masks
            class_ids_tensor: Tensor of class IDs for each instance, or None to derive from semantic mask
            num_classes: Number of semantic classes
            
        Returns:
            Dictionary with prepared tensors
        """
        instance_masks = self._instance_id_tensor_to_binary_masks(instance_tensor)

        num_instances = instance_masks.shape[0]
        
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
                    class_ids.append(torch.tensor(0, device=semantic_tensor.device))
            
            class_ids_tensor = torch.stack(class_ids)
        
        # Create center points heatmap
        centers_heatmap = self._create_center_heatmap_tensor(instance_masks)
        
        # Create bounding boxes
        boxes = self._create_boxes_from_masks_tensor(instance_masks)


        
        return {
            'mask': semantic_tensor,
            'semantic_one_hot': semantic_one_hot,
            'instance_masks': instance_masks,
            'instance_classes': class_ids_tensor,
            'centers': centers_heatmap,
            'boxes': boxes
        }
        
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
        one_hot = torch.zeros((B, num_classes, H, W), dtype=torch.float32, device=mask_tensor.device)
        
        # Scatter to fill in the one-hot representation
        one_hot.scatter_(1, mask_tensor.unsqueeze(1), 1.0)
        
        # Remove batch dimension if it wasn't in the input
        if not has_batch_dim:
            one_hot = one_hot.squeeze(0)
        
        return one_hot
        
    def _hsv_to_rgb(self, h, s, v):
        """Convert HSV color to RGB color"""
        if s == 0.0:
            return v, v, v
        
        i = int(h * 6)
        f = (h * 6) - i
        p = v * (1 - s)
        q = v * (1 - s * f)
        t = v * (1 - s * (1 - f))
        i %= 6
        
        if i == 0:
            return v, t, p
        elif i == 1:
            return q, v, p
        elif i == 2:
            return p, v, t
        elif i == 3:
            return p, q, v
        elif i == 4:
            return t, p, v
        else:
            return v, p, q

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
                                                     device=image_trans.device)
        
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
            return torch.tensor(transformed_boxes, dtype=torch.float32, device=boxes.device)
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

def augmented_duplicates_collate_fn(batch_size=16, img_size=224, strong_augment=True):
    """
    Creates a collate function that duplicates a single sample and applies
    different random augmentations to create a diverse batch
    
    Args:
        batch_size: Size of the target batch
        img_size: Size of the output image
        strong_augment: Whether to use strong augmentation
        
    Returns:
        collate_fn: Function that creates a batch of augmented duplicates
    """
    # Create the augmentation transform
    augment_transform = InstanceSegJointTransform(img_size=img_size, strong=strong_augment)
    
    def collate_fn(batch):
        """
        Duplicate a single sample with different augmentations to fill a batch
        
        Args:
            batch: List containing a single sample
            
        Returns:
            Batch dictionary with the sample augmented batch_size times
        """
        # Make sure we have at least one sample
        if not batch or len(batch) == 0:
            raise ValueError("Batch must contain at least one sample")
        
        # Take the first sample as our base
        base_sample = batch[0]
        
        # Create a list to hold augmented samples
        augmented_samples = []
        
        # Apply different random augmentations batch_size times
        for _ in range(batch_size):
            # Create a copy of the base sample
            sample_copy = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in base_sample.items()}
            
            # Apply a new random augmentation
            augmented_sample = augment_transform(sample_copy)
            
            # Ensure instance data is properly prepared
            if 'instance' in augmented_sample:
                # Access dataset's prepare_instance_tensors method if the dataset is available
                # This part depends on how your dataset class is structured
                instance_data = None
                
                try:
                    # Try to use _prepare_instance_tensors from the dataset
                    dataset = batch[0].get('_dataset', None)
                    if dataset and hasattr(dataset, '_prepare_instance_tensors'):
                        instance_data = dataset._prepare_instance_tensors(
                            semantic_tensor=augmented_sample['mask'],
                            instance_tensor=augmented_sample['instance'],
                            num_classes=dataset.num_classes
                        )
                except:
                    # Fallback - implement inline if needed
                    from multissl.data.instance_segmentation_dataset import COCOInstanceSegmentationDataset
                    dummy_dataset = COCOInstanceSegmentationDataset.__new__(COCOInstanceSegmentationDataset)
                    dummy_dataset.num_classes = 2  # Default to binary segmentation
                    
                    # Add the _prepare_instance_tensors method
                    instance_data = dummy_dataset._prepare_instance_tensors(
                        semantic_tensor=augmented_sample['mask'],
                        instance_tensor=augmented_sample['instance'],
                        num_classes=dummy_dataset.num_classes
                    )
                
                if instance_data:
                    augmented_sample.update(instance_data)
            
            augmented_samples.append(augmented_sample)
        
        # Combine all augmented samples into a batch
        batch_dict = {}
        for key in augmented_samples[0].keys():
            if isinstance(augmented_samples[0][key], torch.Tensor):
                # For tensors, stack along batch dimension
                if key in ['img_id']:
                    # For scalar tensors, collect as list
                    batch_dict[key] = [sample[key] for sample in augmented_samples]
                else:
                    # Stack tensors along batch dimension
                    batch_dict[key] = torch.stack([sample[key] for sample in augmented_samples], dim=0)
            elif isinstance(augmented_samples[0][key], (int, float, str)):
                # For scalar values, collect in a list
                batch_dict[key] = [sample[key] for sample in augmented_samples]
            elif augmented_samples[0][key] is None:
                # Keep None values
                batch_dict[key] = None
            else:
                # For other types, just collect in a list
                batch_dict[key] = [sample[key] for sample in augmented_samples]
        
        return batch_dict
    
    return collate_fn
