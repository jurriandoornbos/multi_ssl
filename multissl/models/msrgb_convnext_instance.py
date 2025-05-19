import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union, Dict
from .msrgb_convnext_upernet import PPM, MSRGBConvNeXtFeatureExtractor
import pytorch_lightning as pl
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment



class InstanceUPerNet(nn.Module):
    """
    UPerNet with instance segmentation capabilities
    Combines semantic segmentation with instance awareness
    """
    def __init__(
        self,
        in_channels: List[int],
        num_classes: int,
        fpn_dim: int = 256,
        ppm_bins: Tuple[int] = (1, 2, 3, 6),
        mask_dim: int = 28,  # Dimension of mask predictions
        num_instances: int = 100  # Max number of instances to detect
    ):
        super(InstanceUPerNet, self).__init__()
        self.in_channels = in_channels
        self.num_instances = num_instances
        
        # Reuse UPerNet components for semantic features
        ppm_reduction_dim = fpn_dim // len(ppm_bins)
        self.ppm = PPM(in_channels[-1], ppm_reduction_dim, ppm_bins)
        ppm_out_dim = in_channels[-1] + ppm_reduction_dim * len(ppm_bins)
        
        # FPN lateral connections
        self.fpn_in = nn.ModuleList()
        for in_c in in_channels[:-1]:
            self.fpn_in.append(nn.Sequential(
                nn.Conv2d(in_c, fpn_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(fpn_dim),
                nn.ReLU(inplace=True)
            ))
            
        # FPN output connections
        self.fpn_out = nn.ModuleList()
        for _ in range(len(in_channels) - 1):
            self.fpn_out.append(nn.Sequential(
                nn.Conv2d(fpn_dim, fpn_dim, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(fpn_dim),
                nn.ReLU(inplace=True)
            ))
            
        # PPM bottleneck
        self.fpn_bottleneck = nn.Sequential(
            nn.Conv2d(ppm_out_dim, fpn_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fpn_dim),
            nn.ReLU(inplace=True)
        )
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(len(in_channels) * fpn_dim, fpn_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(fpn_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(fpn_dim, fpn_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fpn_dim),
            nn.ReLU(inplace=True)
        )
        
        # Semantic segmentation head (class prediction)
        self.sem_seg_head = nn.Conv2d(fpn_dim, num_classes, kernel_size=1)
        
        # Instance-specific heads
        # Center prediction for instance localization
        self.center_head = nn.Conv2d(fpn_dim, 1, kernel_size=1)
        
        # Mask prediction head (for instance masks)
        self.mask_head = nn.Sequential(
            nn.Conv2d(fpn_dim, fpn_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(fpn_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(fpn_dim, mask_dim * num_classes, kernel_size=1)
        )
        
        # Instance embedding for differentiating instances
        self.embedding_head = nn.Sequential(
            nn.Conv2d(fpn_dim, fpn_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(fpn_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(fpn_dim, fpn_dim // 2, kernel_size=1)
        )
        
        # Region of Interest (RoI) pooling for instance-specific features
        self.roi_pool = nn.AdaptiveAvgPool2d((7, 7))
        
        # Instance classification head
        self.instance_classifier = nn.Sequential(
            nn.Linear(fpn_dim * 7 * 7, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes)
        )
        
        # Instance box regression head (for bounding boxes)
        self.box_head = nn.Sequential(
            nn.Linear(fpn_dim * 7 * 7, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 4)  # x, y, w, h
        )
    
    def forward(self, features: List[torch.Tensor]):
        """
        Forward pass for instance segmentation
        
        Args:
            features: List of feature maps from the backbone [feat1, feat2, ..., featN]
                     ordered from lowest resolution to highest
            
        Returns:
            Dictionary containing:
            - sem_logits: Semantic segmentation logits
            - center_heatmap: Center point heatmap for instance detection
            - instance_embeddings: Embeddings to differentiate instances
            - mask_coeffs: Coefficients for generating instance masks
            - cls_scores: Class scores for each detected instance
            - boxes: Bounding box predictions for instances
        """
        # Copy features to avoid modifying input
        feats = features.copy()
        
        # Apply PPM to the last feature map
        ppm_out = self.ppm(feats[-1])
        feats[-1] = self.fpn_bottleneck(ppm_out)
        
        # Build FPN from bottom to top
        fpn_features = [feats[-1]]
        for i in reversed(range(len(self.in_channels) - 1)):
            feat = feats[i]
            lateral = self.fpn_in[i](feat)
            
            # Top-down pathway
            top_down = F.interpolate(
                fpn_features[0], size=lateral.shape[2:], 
                mode='bilinear', align_corners=True
            )
            
            # Add lateral connection
            fpn_feat = lateral + top_down
            fpn_feat = self.fpn_out[i](fpn_feat)
            fpn_features.insert(0, fpn_feat)
        
        # Upsample all feature maps to the highest resolution
        output_size = fpn_features[0].shape[2:]
        aligned_features = []
        
        for i, feat in enumerate(fpn_features):
            if i == 0:
                aligned_features.append(feat)
            else:
                upsampled = F.interpolate(
                    feat, size=output_size, mode='bilinear', align_corners=True
                )
                aligned_features.append(upsampled)
        
        # Fuse all FPN levels
        fused = torch.cat(aligned_features, dim=1)
        fused = self.fusion(fused)
        
        # Semantic segmentation prediction
        sem_logits = self.sem_seg_head(fused)
        
        # Center point heatmap prediction
        center_heatmap = torch.sigmoid(self.center_head(fused))
        
        # Instance embedding
        instance_embeddings = self.embedding_head(fused)
        
        # Mask coefficients
        mask_coeffs = self.mask_head(fused)
        
        # We'd normally use these features to detect instance centers, then run ROI pooling
        # For simplicity, we'll use a mock implementation to demonstrate the flow
        
        # Detect potential instance centers from the heatmap (in a real implementation)
        # For demonstration, we'll just create dummy instance features
        batch_size = fused.shape[0]
        fake_instances = 10  # Pretend we detected 10 instances
        
        # In a real implementation, you'd:
        # 1. Find center points from center_heatmap
        # 2. Extract features around these centers
        # 3. Pool features to fixed size with RoI pooling
        # 4. Predict class and box for each instance
        
        # Mock instance feature extraction (this would be ROI pooling in practice)
        instance_features = torch.zeros(
            (batch_size, fake_instances, fused.shape[1], 7, 7), 
            device=fused.device
        )
        
        # Flatten for FC layers
        instance_features_flat = instance_features.view(
            batch_size * fake_instances, -1
        )
        
        # Instance classification
        cls_scores = self.instance_classifier(instance_features_flat)
        cls_scores = cls_scores.view(batch_size, fake_instances, -1)
        
        # Box regression
        boxes = self.box_head(instance_features_flat)
        boxes = boxes.view(batch_size, fake_instances, 4)
        
        return {
            'sem_logits': sem_logits,
            'center_heatmap': center_heatmap,
            'instance_embeddings': instance_embeddings,
            'mask_coeffs': mask_coeffs,
            'cls_scores': cls_scores,
            'boxes': boxes
        }
class MSRGBConvNeXtInstanceSegmentation(nn.Module):
    """
    ConvNeXt with UPerNet for instance segmentation
    """
    def __init__(
        self,
        num_classes: int,
        rgb_in_channels: int = 3,
        ms_in_channels: int = 5,
        model_size: str = 'tiny',
        fusion_strategy: str = 'hierarchical',
        fusion_type: str = 'attention',
        fpn_dim: int = 256,
        mask_dim: int = 28,
        num_instances: int = 100,
        drop_path_rate: float = 0.1,
        pretrained_backbone: Optional[str] = None,
        freeze_backbone: bool = False,
    ):
        super(MSRGBConvNeXtInstanceSegmentation, self).__init__()
        
        # Create backbone
        self.backbone = MSRGBConvNeXtFeatureExtractor(
            model_name=model_size,
            rgb_in_channels=rgb_in_channels,
            ms_in_channels=ms_in_channels,
            fusion_strategy=fusion_strategy,
            fusion_type=fusion_type,
            drop_path_rate=drop_path_rate
        )
        
        # Get feature dimensions from backbone
        feature_dims = [
            self.backbone.feature_dims[f'layer{i+1}'] 
            for i in range(len(self.backbone.feature_dims) if 'flat' not in self.backbone.feature_dims 
                           else len(self.backbone.feature_dims) - 1)
        ]
        
        # Create instance segmentation head
        self.decode_head = InstanceUPerNet(
            in_channels=feature_dims,
            num_classes=num_classes,
            fpn_dim=fpn_dim,
            mask_dim=mask_dim,
            num_instances=num_instances
        )
        
        # Initialize from pretrained weights if provided
        if pretrained_backbone:
            self._load_pretrained_backbone(pretrained_backbone)
         # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
    
    def forward(self, rgb=None, ms=None):
        """
        Forward pass through backbone and instance segmentation head
        """
        # Get input size for upsampling
        if rgb is not None:
            input_size = rgb.shape[2:]
        elif ms is not None:
            input_size = ms.shape[2:]
        else:
            raise ValueError("At least one of RGB or MS input must be provided")
        
        # Extract features from backbone
        feat_dict = self.backbone(rgb=rgb, ms=ms)
        
        # Convert to list ordered from lowest to highest resolution
        features = []
        for i in range(len(feat_dict) if 'flat' not in feat_dict else len(feat_dict) - 1):
            key = f'layer{i+1}'
            if key in feat_dict:
                features.append(feat_dict[key])
        
        # Apply instance segmentation head
        outputs = self.decode_head(features)
        
        # Upsample outputs to match input size
        outputs['sem_logits'] = F.interpolate(
            outputs['sem_logits'], 
            size=input_size, 
            mode='bilinear', 
            align_corners=True
        )
        
        outputs['center_heatmap'] = F.interpolate(
            outputs['center_heatmap'], 
            size=input_size, 
            mode='bilinear', 
            align_corners=True
        )
        
        outputs['instance_embeddings'] = F.interpolate(
            outputs['instance_embeddings'], 
            size=input_size, 
            mode='bilinear', 
            align_corners=True
        )
        outputs['mask_coeffs'] = F.interpolate(
            outputs['mask_coeffs'], 
            size=input_size, 
            mode='bilinear', 
            align_corners=True
        )
        
        return outputs
    
    def _load_pretrained_backbone(self, checkpoint_path):
        """Load weights from a checkpoint file"""
        print(f"Loading checkpoint from {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('backbone.'):
                    new_key = key[len('backbone.'):]
                    new_state_dict[new_key] = value
                elif key.startswith('feature_extractor.'):
                    new_key = key[len('feature_extractor.'):]
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value
            state_dict = new_state_dict
        else:
            state_dict = checkpoint
        
        missing_keys, unexpected_keys = self.backbone.load_state_dict(state_dict, strict=False)
        
        if len(missing_keys) > 0:
            print(f"Missing keys: {missing_keys}")
        if len(unexpected_keys) > 0:
            print(f"Unexpected keys: {unexpected_keys}")

class MSRGBConvNeXtInstanceSegmentationModule(pl.LightningModule):
    """
    PyTorch Lightning module for MSRGBConvNeXt Instance Segmentation
    """
    def __init__(
        self,
        num_classes: int,
        rgb_in_channels: int = 3,
        ms_in_channels: int = 5,
        model_size: str = 'tiny',
        fusion_strategy: str = 'hierarchical',
        fusion_type: str = 'attention',
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        class_weights: Optional[List[float]] = None,
        pretrained_backbone: Optional[str] = None,
        freeze_backbone: bool = False,
    ):
        super(MSRGBConvNeXtInstanceSegmentationModule, self).__init__()
        
        # Create the model
        self.model = MSRGBConvNeXtInstanceSegmentation(
            num_classes=num_classes,
            rgb_in_channels=rgb_in_channels,
            ms_in_channels=ms_in_channels,
            model_size=model_size,
            fusion_strategy=fusion_strategy,
            fusion_type=fusion_type,
            pretrained_backbone=pretrained_backbone,
            freeze_backbone=freeze_backbone
        )

        
        # Training parameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Setup loss functions
        if class_weights is not None:
            self.register_buffer('class_weights', torch.tensor(class_weights))
            self.semantic_criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        else:
            self.semantic_criterion = nn.CrossEntropyLoss()
        
        # Additional loss functions for instance segmentation
        self.center_criterion = nn.BCELoss()
        self.box_criterion = self.iou_loss
        self.instance_cls_criterion = nn.CrossEntropyLoss()
        self.mask_criterion = nn.BCEWithLogitsLoss()
        
        # Save hyperparameters for logging
        self.save_hyperparameters(ignore=['model'])
        
    def forward(self, rgb=None, ms=None):
        return self.model(rgb=rgb, ms=ms)
    
    def _compute_box_cost_matrix(self, pred_boxes, gt_boxes):
        """
        Compute cost matrix between predicted and ground truth boxes using IoU
        
        Args:
            pred_boxes: Predicted boxes tensor [N, 4] in format [x1, y1, x2, y2]
            gt_boxes: Ground truth boxes tensor [M, 4] in format [x1, y1, x2, y2]
            
        Returns:
            Cost matrix of shape [N, M]
        """
        # Ensure boxes are at least 1 pixel in size
        pred_boxes = torch.clamp(pred_boxes, min=0)
        gt_boxes = torch.clamp(gt_boxes, min=0)
        
        # Ensure width and height are positive
        pred_boxes = pred_boxes.clone()
        gt_boxes = gt_boxes.clone()
        
        # Make sure width and height are positive
        pred_boxes[:, 2] = torch.max(pred_boxes[:, 2], pred_boxes[:, 0] + 1)
        pred_boxes[:, 3] = torch.max(pred_boxes[:, 3], pred_boxes[:, 1] + 1)
        gt_boxes[:, 2] = torch.max(gt_boxes[:, 2], gt_boxes[:, 0] + 1)
        gt_boxes[:, 3] = torch.max(gt_boxes[:, 3], gt_boxes[:, 1] + 1)
        
        num_pred = pred_boxes.size(0)
        num_gt = gt_boxes.size(0)
        
        # Create cost matrix of appropriate size
        cost_matrix = torch.zeros((num_pred, num_gt), device=pred_boxes.device)
        
        # If either set of boxes is empty, return the empty cost matrix
        if num_pred == 0 or num_gt == 0:
            return cost_matrix
        
        # Calculate areas for both sets of boxes
        pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
        gt_area = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
        
        # Compute IoU for each pred-gt pair
        for p in range(num_pred):
            # Get the predicted box
            p_box = pred_boxes[p]
            
            # Calculate intersection with all ground truth boxes
            left = torch.max(p_box[0], gt_boxes[:, 0])
            top = torch.max(p_box[1], gt_boxes[:, 1])
            right = torch.min(p_box[2], gt_boxes[:, 2])
            bottom = torch.min(p_box[3], gt_boxes[:, 3])
            
            # Width and height of intersection
            inter_w = torch.clamp(right - left, min=0)
            inter_h = torch.clamp(bottom - top, min=0)
            
            # Area of intersection
            intersection = inter_w * inter_h
            
            # Union area (using broadcasting)
            union = pred_area[p] + gt_area - intersection
            
            # IoU
            iou = intersection / (union + 1e-6)
            
            # Use 1-IoU as cost (lower is better)
            cost_matrix[p] = 1 - iou
        
        return cost_matrix

    def training_step(self, batch, batch_idx):
        # Extract inputs and targets
        rgb = batch.get('rgb')
        ms = batch.get('ms')
        semantic_target = batch['mask']
        
        # Instance segmentation targets
        instance_masks = batch.get('instance_masks', None)
        instance_classes = batch.get('instance_classes', None)
        boxes = batch.get('boxes', None)
        
        # Forward pass
        outputs = self(rgb=rgb, ms=ms)
        
        # Calculate semantic segmentation loss
        semantic_loss = self.semantic_criterion(outputs['sem_logits'], semantic_target)
        
        # Initialize center loss (if centers are provided)
        center_loss = torch.tensor(0.0, device=self.device)
        if 'centers' in batch and batch['centers'] is not None:
            center_target = batch['centers']
            center_loss = self.center_criterion(outputs['center_heatmap'], center_target)
        
        # Initialize instance losses
        box_loss = torch.tensor(0.0, device=self.device)
        instance_cls_loss = torch.tensor(0.0, device=self.device)
        mask_loss = torch.tensor(0.0, device=self.device)
        
        # Calculate instance losses if we have instance targets
        if instance_masks is not None and instance_classes is not None and boxes is not None:
            # Process each sample in batch separately
            batch_size = semantic_target.shape[0]
            for b in range(batch_size):
                # Get predictions for this sample
                pred_cls_scores = outputs['cls_scores'][b]
                pred_boxes = outputs['boxes'][b]
                pred_mask_coeffs = outputs['mask_coeffs'][b]
                
                # Get ground truth for this sample
                batch_instance_masks = instance_masks[b]
                batch_instance_classes = instance_classes[b]
                batch_instance_boxes = boxes[b]
                
                # Skip if no instances in this sample
                if batch_instance_masks.shape[0] == 0 or pred_boxes.shape[0] == 0:
                    continue
                
                # Compute cost matrix for matching
                cost_matrix = self._compute_box_cost_matrix(pred_boxes, batch_instance_boxes)
                
                # Use Hungarian algorithm for matching
                from scipy.optimize import linear_sum_assignment
                pred_indices, gt_indices = linear_sum_assignment(cost_matrix.detach().cpu().numpy())
                
                # Convert to tensors
                pred_indices = torch.tensor(pred_indices, device=self.device)
                gt_indices = torch.tensor(gt_indices, device=self.device)
                
                # Skip if no matches
                if len(pred_indices) == 0:
                    continue
                
                # Calculate box regression loss for matched pairs
                matched_pred_boxes = pred_boxes[pred_indices]
                matched_gt_boxes = batch_instance_boxes[gt_indices]
                box_loss += self.box_criterion(matched_pred_boxes, matched_gt_boxes)
                
                # Calculate classification loss for matched pairs
                matched_pred_cls = pred_cls_scores[pred_indices]
                matched_gt_cls = batch_instance_classes[gt_indices]
                instance_cls_loss += self.instance_cls_criterion(matched_pred_cls, matched_gt_cls)
                
                # Calculate mask loss if applicable
                if hasattr(self, 'mask_criterion') and pred_mask_coeffs is not None:
                    # This would need to be implemented based on your specific mask representation
                    pass
        
        # Normalize instance losses by batch size
        batch_size = semantic_target.shape[0]
        if batch_size > 0:
            box_loss = box_loss / batch_size
            instance_cls_loss = instance_cls_loss / batch_size
            mask_loss = mask_loss / batch_size
        
        # Combine all losses with appropriate weights
        total_loss = semantic_loss + 0.1 * center_loss + 0.1 * box_loss + 0.1 * instance_cls_loss + 0.1 * mask_loss
        
        # Log components
        self.log('train_semantic_loss', semantic_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_center_loss', center_loss, on_step=True, on_epoch=True)
        self.log('train_box_loss', box_loss, on_step=True, on_epoch=True)
        self.log('train_instance_cls_loss', instance_cls_loss, on_step=True, on_epoch=True)
        self.log('train_mask_loss', mask_loss, on_step=True, on_epoch=True)
        self.log('train_total_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return total_loss

    def normalize_boxes(self, boxes, img_size):
        """
        Normalize box coordinates to 0-1 range.
        
        Args:
            boxes: Tensor of shape [N, 4] with coordinates [x1, y1, x2, y2]
            img_size: Tuple (height, width) of the original image
            
        Returns:
            Normalized boxes with coordinates between 0 and 1
        """
        height, width = img_size
        
        # Clone to avoid modifying the original tensor
        normalized = boxes.clone()
        
        # Convert to 0-1 range
        normalized[:, 0] /= width  # x1
        normalized[:, 1] /= height  # y1
        normalized[:, 2] /= width  # x2
        normalized[:, 3] /= height  # y2
        
        return normalized
        
    def validation_step(self, batch, batch_idx):
        # Similar to training_step but with evaluation metrics
        rgb = batch.get('rgb')
        ms = batch.get('ms')
        semantic_target = batch['mask']
        
        # Forward pass
        outputs = self(rgb=rgb, ms=ms)
        
        # Calculate semantic segmentation metrics
        semantic_loss = self.semantic_criterion(outputs['sem_logits'], semantic_target)
        preds = torch.argmax(outputs['sem_logits'], dim=1)
        accuracy = (preds == semantic_target).float().mean()
        
        # Calculate IoU for each class
        iou_per_class = []
        for cls in range(outputs['sem_logits'].shape[1]):
            intersection = ((preds == cls) & (semantic_target == cls)).float().sum()
            union = ((preds == cls) | (semantic_target == cls)).float().sum()
            iou = intersection / (union + 1e-6)
            iou_per_class.append(iou)
            self.log(f'val_iou_class{cls}', iou, on_step=False, on_epoch=True)
        
        # Mean IoU
        mean_iou = torch.stack(iou_per_class).mean()
        
        # Log metrics
        self.log('val_semantic_loss', semantic_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_accuracy', accuracy, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_mean_iou', mean_iou, on_step=False, on_epoch=True, prog_bar=True)
        
        # In a complete implementation, you'd also evaluate instance segmentation metrics
        # such as Average Precision (AP) at different IoU thresholds
        
        return semantic_loss
        
    def configure_optimizers(self):
        # Set up separate parameter groups with different learning rates
        backbone_params = []
        head_params = []
        
        # Separate backbone and head parameters
        for name, param in self.named_parameters():
            if 'backbone' in name:
                backbone_params.append(param)
            else:
                head_params.append(param)
        
        # Create optimizer with parameter groups
        optimizer = torch.optim.AdamW([
            {'params': backbone_params, 'lr': self.learning_rate * 0.1},  # Lower LR for backbone
            {'params': head_params, 'lr': self.learning_rate}
        ], weight_decay=self.weight_decay)
        
        # Create learning rate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=self.trainer.max_epochs
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_semantic_loss',
                'interval': 'epoch'
            }
        }
        
    def calculate_mask_loss_with_matching(self, mask_coeffs, target_masks):
        """
        Calculate mask loss with Hungarian matching between predictions and targets.
        
        Args:
            mask_coeffs: Model prediction for masks [B, M, H, W]
            target_masks: Target masks [B, N, H, W]
            
        Returns:
            Mask loss value
        """
        batch_size = mask_coeffs.shape[0]
        total_loss = 0.0
        
        # Process each batch item separately
        for b in range(batch_size):
            pred_masks = mask_coeffs[b]  # [M, H, W]
            gt_masks = target_masks[b]   # [N, H, W]
            
            # Skip if either prediction or target has no instances
            if pred_masks.shape[0] == 0 or gt_masks.shape[0] == 0:
                continue
            
            # Compute pairwise IoU cost matrix
            cost_matrix = self._compute_mask_cost_matrix(pred_masks, gt_masks)
            
            # Use Hungarian algorithm to find optimal matching
            pred_indices, gt_indices = linear_sum_assignment(cost_matrix.cpu().numpy())
            
            # Extract matched pairs
            matched_pred_masks = pred_masks[pred_indices]
            matched_gt_masks = gt_masks[gt_indices]
            
            # Compute loss on matched pairs
            batch_loss = self.mask_criterion(matched_pred_masks, matched_gt_masks)
            total_loss += batch_loss
        
        # Average over batch size
        return total_loss / batch_size

    def _compute_mask_cost_matrix(self, pred_masks, gt_masks):
        """
        Compute cost matrix for Hungarian matching based on IoU.
        
        Args:
            pred_masks: Predicted masks [M, H, W]
            gt_masks: Ground truth masks [N, H, W]
            
        Returns:
            Cost matrix of shape [M, N]
        """
        M, H, W = pred_masks.shape
        N = gt_masks.shape[0]
        
        # Reshape for broadcasting
        pred_flat = pred_masks.view(M, 1, H*W)  # [M, 1, H*W]
        gt_flat = gt_masks.view(1, N, H*W)      # [1, N, H*W]
        
        # Compute intersection
        intersection = torch.sum((pred_flat > 0.5) & (gt_flat > 0.5), dim=2)  # [M, N]
        
        # Compute union
        pred_areas = torch.sum(pred_flat > 0.5, dim=2)  # [M, 1]
        gt_areas = torch.sum(gt_flat > 0.5, dim=2)      # [1, N]
        union = pred_areas + gt_areas - intersection    # [M, N]
        
        # Compute IoU
        iou = intersection / (union + 1e-6)  # [M, N]
        
        # Convert to cost matrix (1 - IoU)
        cost_matrix = 1.0 - iou  # [M, N]
        
        return cost_matrix
    def iou_loss(self, pred_boxes, target_boxes):
        """
        Compute IoU loss between prediction and target boxes.
        
        Args:
            pred_boxes: Predicted boxes [B, N, 4] in format [x1, y1, x2, y2]
            target_boxes: Target boxes [B, N, 4] in format [x1, y1, x2, y2]
            
        Returns:
            IoU loss (1 - IoU)
        """
        # Extract coordinates
        pred_x1, pred_y1, pred_x2, pred_y2 = pred_boxes.unbind(-1)
        target_x1, target_y1, target_x2, target_y2 = target_boxes.unbind(-1)
        
        # Calculate areas
        pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
        target_area = (target_x2 - target_x1) * (target_y2 - target_y1)
        
        # Intersection coordinates
        inter_x1 = torch.max(pred_x1, target_x1)
        inter_y1 = torch.max(pred_y1, target_y1)
        inter_x2 = torch.min(pred_x2, target_x2)
        inter_y2 = torch.min(pred_y2, target_y2)
        
        # Intersection area
        inter_w = (inter_x2 - inter_x1).clamp(min=0)
        inter_h = (inter_y2 - inter_y1).clamp(min=0)
        intersection = inter_w * inter_h
        
        # Union area
        union = pred_area + target_area - intersection
        
        # IoU
        iou = intersection / (union + 1e-6)
        
        # Loss is 1 - IoU
        loss = 1 - iou
        
        return loss.mean()
    
    def predict_instances(self, outputs, confidence_threshold=0.5, nms_threshold=0.5, max_instances=100):
        """
        Extract instance predictions from model outputs
        
        Args:
            outputs: Dictionary of model outputs
            confidence_threshold: Minimum confidence score for instances
            nms_threshold: IoU threshold for non-maximum suppression
            max_instances: Maximum number of instances to return
            
        Returns:
            Dictionary with instance predictions
        """
        # Extract relevant outputs
        center_heatmap = outputs['center_heatmap']  # [B, 1, H, W]
        mask_coeffs = outputs['mask_coeffs']  # [B, mask_dim*num_classes, H, W]
        cls_scores = outputs['cls_scores']  # [B, N, num_classes]
        boxes = outputs['boxes']  # [B, N, 4]
        
        batch_size = center_heatmap.shape[0]
        
        # Process each sample in the batch
        all_instance_masks = []
        all_center_scores = []
        all_instance_boxes = []
        all_instance_classes = []
        
        for b in range(batch_size):
            # Extract peak points from center heatmap (these are instance centers)
            centers_b = center_heatmap[b, 0]  # [H, W]
            
            # Apply threshold to center heatmap
            centers_binary = (centers_b > confidence_threshold).float()
            
            # Find connected components in thresholded heatmap
            from skimage.measure import label as skimage_label
            import numpy as np
            
            # Convert to numpy for connected component analysis
            centers_np = centers_binary.cpu().numpy()
            instance_labels = skimage_label(centers_np)
            
            # Get instance centers (one per connected component)
            instance_ids = np.unique(instance_labels)
            
            # Remove background (0)
            instance_ids = instance_ids[instance_ids != 0]
            
            # Limit number of instances
            if len(instance_ids) > max_instances:
                instance_ids = instance_ids[:max_instances]
                
            # Process each detected instance
            instance_masks = []
            center_scores = []
            instance_boxes = []
            instance_classes = []
            
            for instance_id in instance_ids:
                # Get instance mask from connected component
                instance_mask = (instance_labels == instance_id)
                
                # Find center coordinates
                ys, xs = np.where(instance_mask)
                if len(ys) == 0:
                    continue
                    
                cy = np.mean(ys)
                cx = np.mean(xs)
                
                # Convert to integer coordinates
                cy_int = int(np.round(cy))
                cx_int = int(np.round(cx))
                
                # Get confidence score at center
                center_score = centers_b[cy_int, cx_int].item()  # This is a float, not a tensor
                
                # Skip low confidence instances
                if center_score < confidence_threshold:
                    continue
                    
                # Find nearest predicted box
                # This assumes boxes are ordered by confidence
                box_dists = []
                for i in range(len(boxes[b])):
                    box = boxes[b][i]
                    # Calculate center of box
                    box_cx = (box[0] + box[2]) / 2
                    box_cy = (box[1] + box[3]) / 2
                    
                    # Calculate distance to center
                    dist = torch.sqrt((box_cx - cx_int) ** 2 + (box_cy - cy_int) ** 2)
                    box_dists.append((i, dist.item()))
                
                # Sort by distance
                box_dists.sort(key=lambda x: x[1])
                
                # Get nearest box and its class
                if box_dists:
                    nearest_box_idx = box_dists[0][0]
                    box = boxes[b][nearest_box_idx]
                    cls_score = cls_scores[b][nearest_box_idx]
                    cls_id = torch.argmax(cls_score).item()
                    
                    # Convert instance mask to tensor
                    mask_tensor = torch.from_numpy(instance_mask).float().to(box.device)
                    
                    # Add to predictions
                    instance_masks.append(mask_tensor)
                    center_scores.append(center_score)  # This is a float, not a tensor
                    instance_boxes.append(box)
                    instance_classes.append(cls_id)
            
            # Convert to tensors carefully
            if instance_masks:
                instance_masks = torch.stack(instance_masks)
                # Convert center_scores from floats to tensor
                center_scores = torch.tensor(center_scores, device=instance_masks.device)
                instance_boxes = torch.stack(instance_boxes)
                instance_classes = torch.tensor(instance_classes, device=instance_masks.device)
                
                # Apply non-maximum suppression
                keep_indices = self._nms(instance_boxes, center_scores, nms_threshold)
                
                instance_masks = instance_masks[keep_indices]
                center_scores = center_scores[keep_indices]
                instance_boxes = instance_boxes[keep_indices]
                instance_classes = instance_classes[keep_indices]
            else:
                # Create empty tensors with the right shape
                h, w = center_heatmap.shape[2:]
                instance_masks = torch.zeros((0, h, w), device=center_heatmap.device)
                center_scores = torch.zeros(0, device=center_heatmap.device)
                instance_boxes = torch.zeros((0, 4), device=center_heatmap.device)
                instance_classes = torch.zeros(0, dtype=torch.long, device=center_heatmap.device)
            
            all_instance_masks.append(instance_masks)
            all_center_scores.append(center_scores)
            all_instance_boxes.append(instance_boxes)
            all_instance_classes.append(instance_classes)
        
        return {
            'instance_masks': all_instance_masks,
            'center_scores': all_center_scores,
            'instance_boxes': all_instance_boxes,
            'instance_classes': all_instance_classes
        }

    def _nms(self, boxes, scores, iou_threshold):
        """
        Non-maximum suppression to remove overlapping boxes
        
        Args:
            boxes: Bounding boxes [N, 4]
            scores: Confidence scores [N]
            iou_threshold: IoU threshold for suppression
            
        Returns:
            Indices of kept boxes
        """
        # If no boxes, return empty tensor
        if boxes.shape[0] == 0:
            return torch.zeros(0, dtype=torch.long, device=boxes.device)
        
        # Convert to (x1, y1, x2, y2) format if necessary
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        
        # Calculate areas
        areas = (x2 - x1) * (y2 - y1)
        
        # Sort boxes by score
        order = torch.argsort(scores, descending=True)
        
        keep = []
        while order.shape[0] > 0:
            # Pick the box with highest score
            i = order[0].item()
            keep.append(i)
            
            # If only one box left, break
            if order.shape[0] == 1:
                break
            
            # Calculate IoU with remaining boxes
            xx1 = torch.max(x1[i], x1[order[1:]])
            yy1 = torch.max(y1[i], y1[order[1:]])
            xx2 = torch.min(x2[i], x2[order[1:]])
            yy2 = torch.min(y2[i], y2[order[1:]])
            
            # Calculate intersection area
            w = torch.clamp(xx2 - xx1, min=0)
            h = torch.clamp(yy2 - yy1, min=0)
            intersection = w * h
            
            # Calculate IoU
            union = areas[i] + areas[order[1:]] - intersection
            iou = intersection / (union + 1e-6)
            
            # Keep boxes with IoU below threshold
            inds = torch.where(iou <= iou_threshold)[0]
            order = order[inds + 1]  # +1 because we skipped the first element
        
        return torch.tensor(keep, dtype=torch.long, device=boxes.device)

    def _find_center_points(self, heatmap, threshold=0.5, nms_kernel_size=7, max_centers=100):
        """
        Find center points from heatmap with non-maximum suppression.
        
        Args:
            heatmap: Center point heatmap tensor [H, W]
            threshold: Confidence threshold
            nms_kernel_size: Kernel size for non-maximum suppression
            max_centers: Maximum number of centers to return
            
        Returns:
            List of tuples (y, x, score) for center points
        """
        # Threshold heatmap
        binary = (heatmap > threshold).float()
        
        # Apply non-maximum suppression
        # First create a max pooled version of the heatmap
        padded = F.pad(
            heatmap.unsqueeze(0).unsqueeze(0),
            [(nms_kernel_size - 1) // 2] * 4,
            mode='constant'
        )
        
        max_pooled = F.max_pool2d(
            padded,
            kernel_size=nms_kernel_size,
            stride=1,
            padding=0
        ).squeeze()
        
        # Keep only pixels that are the local maximum
        is_max = (heatmap == max_pooled) & (binary > 0)
        
        # Get coordinates and scores of center points
        y_indices, x_indices = torch.where(is_max)
        
        # Get confidence scores at these points
        scores = heatmap[y_indices, x_indices]
        
        # Sort by confidence score (descending)
        sorted_indices = torch.argsort(scores, descending=True)
        
        # Take only top max_centers
        if len(sorted_indices) > max_centers:
            sorted_indices = sorted_indices[:max_centers]
        
        # Return as list of tuples (y, x, score)
        centers = [(y_indices[i].item(), x_indices[i].item(), scores[i].item()) 
                for i in sorted_indices]
        
        return centers

    def _compute_embedding_similarity(self, center_embedding, all_embeddings):
        """
        Compute similarity between center embedding and all embeddings.
        
        Args:
            center_embedding: Embedding vector at center [E]
            all_embeddings: Embedding tensor for all pixels [E, H, W]
            
        Returns:
            Similarity map [H, W]
        """
        # Normalize embeddings for cosine similarity
        center_norm = F.normalize(center_embedding, p=2, dim=0)
        
        # Reshape and normalize all embeddings
        e, h, w = all_embeddings.shape
        all_embeddings_flat = all_embeddings.reshape(e, -1)
        all_embeddings_norm = F.normalize(all_embeddings_flat, p=2, dim=0)
        
        # Compute similarity (dot product of normalized vectors = cosine similarity)
        similarity = torch.sum(center_norm.unsqueeze(1) * all_embeddings_norm, dim=0)
        
        # Reshape back to spatial dimensions
        similarity = similarity.reshape(h, w)
        
        return similarity

    def _refine_mask(self, mask, semantic_pred, min_size=10):
        """
        Refine instance mask using semantic prediction and morphological operations.
        
        Args:
            mask: Binary instance mask [H, W]
            semantic_pred: Semantic segmentation prediction [H, W]
            min_size: Minimum size of instance
            
        Returns:
            Refined mask [H, W]
        """
        # Convert to CPU numpy for morphological operations
        mask_np = mask.cpu().numpy()
        
        # Apply morphological operations (can be done in PyTorch but simpler in OpenCV)
        try:
            import cv2
            # Close small gaps
            kernel = np.ones((3, 3), np.uint8)
            mask_np = cv2.morphologyEx(mask_np.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
            
            # Remove small isolated regions
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_np, connectivity=8)
            
            # Keep only the largest component if it's above min_size
            if num_labels > 1:
                # Skip background (0)
                areas = stats[1:, cv2.CC_STAT_AREA]
                if np.max(areas) < min_size:
                    # Instance too small, remove it
                    mask_np = np.zeros_like(mask_np)
                else:
                    # Keep only largest component
                    largest_label = np.argmax(areas) + 1  # +1 because we skipped background
                    mask_np = (labels == largest_label).astype(np.uint8)
        except ImportError:
            # If OpenCV is not available, use simple thresholding
            mask_np = (mask_np > 0.5).astype(np.uint8)
        
        # Convert back to tensor
        refined_mask = torch.from_numpy(mask_np).to(mask.device).float()
        
        return refined_mask

    def _mask_to_box(self, mask):
        """
        Convert instance mask to bounding box.
        
        Args:
            mask: Binary instance mask [H, W]
            
        Returns:
            Bounding box tensor [4] in format [x1, y1, x2, y2]
        """
        # Find non-zero indices
        y_indices, x_indices = torch.where(mask > 0.5)
        
        if len(y_indices) == 0:
            # Empty mask, return dummy box
            return torch.tensor([0, 0, 1, 1], device=mask.device)
        
        # Get bounding box coordinates
        x1 = torch.min(x_indices).float()
        y1 = torch.min(y_indices).float()
        x2 = torch.max(x_indices).float()
        y2 = torch.max(y_indices).float()
        
        return torch.tensor([x1, y1, x2, y2], device=mask.device)

    def _non_maximum_suppression(self, boxes, scores, threshold=0.5):
        """
        Apply non-maximum suppression to remove overlapping boxes.
        
        Args:
            boxes: Bounding boxes [N, 4]
            scores: Confidence scores [N]
            threshold: IoU threshold for overlap
            
        Returns:
            Tensor of indices to keep
        """
        # If no boxes, return empty tensor
        if boxes.shape[0] == 0:
            return torch.zeros(0, dtype=torch.long, device=boxes.device)
        
        # Get coordinates for IoU calculation
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        
        # Calculate areas
        areas = (x2 - x1) * (y2 - y1)
        
        # Sort by score
        order = torch.argsort(scores, descending=True)
        
        keep = []
        while order.shape[0] > 0:
            # Pick the box with highest score
            i = order[0].item()
            keep.append(i)
            
            # If only one box left, break
            if order.shape[0] == 1:
                break
            
            # Calculate IoU with remaining boxes
            xx1 = torch.max(x1[i], x1[order[1:]])
            yy1 = torch.max(y1[i], y1[order[1:]])
            xx2 = torch.min(x2[i], x2[order[1:]])
            yy2 = torch.min(y2[i], y2[order[1:]])
            
            # Calculate intersection area
            w = torch.clamp(xx2 - xx1, min=0)
            h = torch.clamp(yy2 - yy1, min=0)
            intersection = w * h
            
            # Calculate IoU
            iou = intersection / (areas[i] + areas[order[1:]] - intersection)
            
            # Keep boxes with IoU less than threshold
            inds = torch.where(iou <= threshold)[0]
            order = order[inds + 1]
        
        return torch.tensor(keep, dtype=torch.long, device=boxes.device)