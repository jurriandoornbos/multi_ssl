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
    def __init__(self, img_dir, coco_json_path, instance_dir, transform=None, target_transform=None):
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
        self.target_transform = target_transform
        
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
            dict: Contains RGB image and segmentation mask
        """
        img_id = self.img_ids[idx]
        
        # Load image information
        img_info = self.coco.loadImgs(img_id)[0]
        image_path = os.path.join(self.img_dir, img_info['file_name'])
        
        # Load RGB image
        image = Image.open(image_path).convert("RGB")


        instance_id = self.instance_ids[idx]
        instance_path = os.path.join(self.instance_dir, instance_id)

        instance = Image.open(instance_path).convert()
        
        # Create segmentation mask
        mask = self.create_mask(img_id, img_info['height'], img_info['width'])
        
        # Apply transforms
        if self.transform is not None:
            image = self.transform(image)
            
        if self.target_transform is not None:
            mask = self.target_transform(mask)
            instance = self.target_transform(self.transform(instance))
        else:
            # Convert mask to tensor if no transform is specified
            mask = torch.from_numpy(mask).long()
            instance =torch.from_numpy(instance).long()
        instance_onehot = self._prepare_instance_tensors(semantic_tensor = mask, 
                                                   instance_tensor = instance,
                                                   num_classes= self.num_classes)
        
        instance_onehot["rgb"] = image
        instance_onehot["img_id"] = img_id
        return instance_onehot
    
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
            semantic_mask = batch["semantic_mask"]
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