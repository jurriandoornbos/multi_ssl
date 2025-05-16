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
    
    def training_step(self, batch, batch_idx):
        # Extract inputs and targets
        rgb = batch.get('rgb')
        ms = batch.get('ms')
        semantic_target = batch['mask']
        
        # Center heatmap target and other instance data
        center_target = batch.get('centers', torch.zeros_like(semantic_target, dtype=torch.float32))
        instance_masks = batch.get('instance_masks', None)
        instance_classes = batch.get('instance_classes', None)
        boxes_target = batch.get('boxes', None)
        
        # Get input dimensions for normalization
        img_size = (semantic_target.shape[-2], semantic_target.shape[-1])
        
        # Forward pass
        outputs = self(rgb=rgb, ms=ms)
        
        # Calculate semantic segmentation loss
        semantic_loss = self.semantic_criterion(outputs['sem_logits'], semantic_target)
        
        # Ensure center_target has a channel dimension [B, H, W] -> [B, 1, H, W]
        if center_target.dim() == 3 and outputs['center_heatmap'].dim() == 4:
            center_target = center_target.unsqueeze(1)
        
        # Calculate center point prediction loss
        center_loss = self.center_criterion(outputs['center_heatmap'], center_target)
        
        # Initialize instance losses
        box_loss = torch.tensor(0.0, device=self.device)
        instance_cls_loss = torch.tensor(0.0, device=self.device)
        mask_loss = torch.tensor(0.0, device=self.device)
        
        # Process instance masks and associated targets
        if instance_masks is not None:
            # First detect center points from ground truth masks
            # We'll use these to associate ground truth instances with predictions
            batch_size = instance_masks.shape[0]
            instance_loss_components = []
            
            for b in range(batch_size):
                # Get target instances for this batch item
                batch_instance_masks = instance_masks[b]  # [N, H, W]
                batch_instance_boxes = boxes_target[b] if boxes_target is not None else None  # [N, 4]
                batch_instance_classes = instance_classes[b] if instance_classes is not None else None  # [N]
                
                # Skip if no instances in this sample
                if batch_instance_masks.shape[0] == 0:
                    continue
                
                # Extract center points from ground truth masks
                gt_centers = []
                for i in range(batch_instance_masks.shape[0]):
                    mask = batch_instance_masks[i]
                    # Find center of mass
                    y_indices, x_indices = torch.where(mask > 0.5)
                    if len(y_indices) == 0:
                        continue
                        
                    center_y = torch.mean(y_indices.float()).round().long()
                    center_x = torch.mean(x_indices.float()).round().long()
                    gt_centers.append((center_y, center_x, i))  # (y, x, instance_idx)
                
                # Skip if no valid centers found
                if not gt_centers:
                    continue
                
                # Calculate embedding loss if we have instance embeddings
                if 'instance_embeddings' in outputs:
                    embeddings = outputs['instance_embeddings'][b]  # [E, H, W]
                    embedding_loss = 0.0
                    num_centers = len(gt_centers)
                    
                    if num_centers > 0:
                        # Extract embeddings at center points
                        center_embeddings = []
                        for center_y, center_x, inst_idx in gt_centers:
                            center_embedding = embeddings[:, center_y, center_x]  # [E]
                            center_embeddings.append(center_embedding)
                        
                        # Calculate embedding loss (pull loss - centers of same instance should be close)
                        if len(center_embeddings) > 1:
                            center_embeddings = torch.stack(center_embeddings)  # [N, E]
                            embedding_norm = F.normalize(center_embeddings, p=2, dim=1)
                            
                            # Calculate pairwise distances
                            distances = torch.cdist(embedding_norm, embedding_norm, p=2)
                            
                            # Get instance indices
                            instance_indices = torch.tensor([idx for _, _, idx in gt_centers], 
                                                        device=embeddings.device)
                            
                            # Create mask for same/different instances
                            same_instance = instance_indices.unsqueeze(1) == instance_indices.unsqueeze(0)
                            
                            # Pull loss - same instance embeddings should be close
                            pull_loss = distances[same_instance].mean()
                            
                            # Push loss - different instance embeddings should be far
                            # Using hinge loss with margin
                            margin = 1.0
                            push_loss = torch.clamp(margin - distances[~same_instance], min=0).mean()
                            
                            embedding_loss = pull_loss + push_loss
                        
                    instance_loss_components.append(embedding_loss)
                
                # Match predictions with ground truth for box and mask losses
                # Method 1: Simple matching by order
                if outputs['boxes'].shape[1] == batch_instance_masks.shape[0]:
                    # If number of predictions matches ground truth, use direct matching
                    pred_boxes = outputs['boxes'][b]
                    if batch_instance_boxes is not None:
                        # Normalize boxes to 0-1 range for loss calculation
                        norm_pred_boxes = self.normalize_boxes(pred_boxes, img_size)
                        norm_gt_boxes = self.normalize_boxes(batch_instance_boxes, img_size)
                        
                        # Box loss
                        box_loss_batch = self.box_criterion(norm_pred_boxes, norm_gt_boxes)
                        instance_loss_components.append(box_loss_batch)
                    
                    # Class loss if we have class predictions
                    if 'cls_scores' in outputs and batch_instance_classes is not None:
                        cls_scores = outputs['cls_scores'][b]
                        cls_loss_batch = self.instance_cls_criterion(
                            cls_scores, batch_instance_classes
                        )
                        instance_loss_components.append(cls_loss_batch)
                    
                    # Mask loss if we have mask predictions
                    if 'mask_coeffs' in outputs:
                        mask_pred = outputs['mask_coeffs'][b]
                        
                        # If sigmoid is needed
                        mask_pred = torch.sigmoid(mask_pred)
                        
                        mask_loss_batch = self.mask_criterion(mask_pred, batch_instance_masks)
                        instance_loss_components.append(mask_loss_batch)
                
                # Method 2: Hungarian matching (more sophisticated)
                else:
                    # Use more sophisticated matching when prediction count doesn't match ground truth
                    if 'mask_coeffs' in outputs:
                        mask_loss_batch = self.calculate_mask_loss_with_matching(
                            outputs['mask_coeffs'][b:b+1],
                            batch_instance_masks.unsqueeze(0)
                        )
                        instance_loss_components.append(mask_loss_batch)
                    
                    # For box loss with Hungarian matching
                    if 'boxes' in outputs and batch_instance_boxes is not None:
                        pred_boxes = outputs['boxes'][b]
                        
                        # Compute IoU cost matrix
                        cost_matrix = self._compute_box_cost_matrix(pred_boxes, batch_instance_boxes)
                        
                        # Use Hungarian algorithm for matching
                        pred_indices, gt_indices = linear_sum_assignment(cost_matrix.cpu().numpy())
                        
                        # Match boxes
                        matched_pred_boxes = pred_boxes[pred_indices]
                        matched_gt_boxes = batch_instance_boxes[gt_indices]
                        
                        # Normalize
                        norm_pred_boxes = self.normalize_boxes(matched_pred_boxes, img_size)
                        norm_gt_boxes = self.normalize_boxes(matched_gt_boxes, img_size)
                        
                        # Compute loss
                        box_loss_batch = self.box_criterion(norm_pred_boxes, norm_gt_boxes)
                        instance_loss_components.append(box_loss_batch)
                        
                        # Also match classes if available
                        if 'cls_scores' in outputs and batch_instance_classes is not None:
                            cls_scores = outputs['cls_scores'][b][pred_indices]
                            matched_classes = batch_instance_classes[gt_indices]
                            
                            cls_loss_batch = self.instance_cls_criterion(
                                cls_scores, matched_classes
                            )
                            instance_loss_components.append(cls_loss_batch)
            
            # Combine all instance loss components
            if instance_loss_components:
                instance_loss = torch.stack(instance_loss_components).mean()
                
                # Assign to individual loss components for logging
                # This is arbitrary - we could split differently
                mask_loss = instance_loss * 0.4
                box_loss = instance_loss * 0.3
                instance_cls_loss = instance_loss * 0.3
        
        # Combine all losses
        total_loss = semantic_loss + center_loss + box_loss + instance_cls_loss + mask_loss
        
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

    def _compute_box_cost_matrix(self, pred_boxes, gt_boxes):
        """
        Compute cost matrix for box matching using IoU.
        
        Args:
            pred_boxes: Predicted boxes [M, 4]
            gt_boxes: Ground truth boxes [N, 4]
            
        Returns:
            Cost matrix [M, N]
        """
        M = pred_boxes.shape[0]
        N = gt_boxes.shape[0]
        
        # Extract coordinates
        pred_x1 = pred_boxes[:, 0].view(M, 1)
        pred_y1 = pred_boxes[:, 1].view(M, 1)
        pred_x2 = pred_boxes[:, 2].view(M, 1)
        pred_y2 = pred_boxes[:, 3].view(M, 1)
        
        gt_x1 = gt_boxes[:, 0].view(1, N)
        gt_y1 = gt_boxes[:, 1].view(1, N)
        gt_x2 = gt_boxes[:, 2].view(1, N)
        gt_y2 = gt_boxes[:, 3].view(1, N)
        
        # Calculate areas
        pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
        gt_area = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)
        
        # Compute IoU
        # First, compute intersection
        inter_x1 = torch.max(pred_x1, gt_x1)
        inter_y1 = torch.max(pred_y1, gt_y1)
        inter_x2 = torch.min(pred_x2, gt_x2)
        inter_y2 = torch.min(pred_y2, gt_y2)
        
        # Clip to ensure valid dimensions
        inter_w = (inter_x2 - inter_x1).clamp(min=0)
        inter_h = (inter_y2 - inter_y1).clamp(min=0)
        
        # Intersection area
        intersection = inter_w * inter_h
        
        # Union area
        union = pred_area + gt_area.t() - intersection
        
        # IoU
        iou = intersection / (union + 1e-6)
        
        # Cost matrix is 1 - IoU
        cost_matrix = 1.0 - iou
        
        return cost_matrix
        
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
    
    def predict_instances(self, outputs, confidence_threshold=0.5, nms_threshold=0.3, max_instances=20):
        """
        Convert model outputs to actual instance segmentation results.
        
        Args:
            outputs: Dictionary of model outputs
            confidence_threshold: Threshold for center detection confidence
            nms_threshold: Threshold for non-maximum suppression
            max_instances: Maximum number of instances to return
            
        Returns:
            Dictionary with processed instance segmentation results
        """
        batch_size = outputs['sem_logits'].shape[0]
        height, width = outputs['sem_logits'].shape[2], outputs['sem_logits'].shape[3]
        
        # Get semantic segmentation class prediction
        semantic_pred = torch.argmax(outputs['sem_logits'], dim=1)  # [B, H, W]
        
        # Process results for each image in the batch
        instance_results = []
        
        for b in range(batch_size):
            # 1. Find center points from heatmap
            center_heatmap = outputs['center_heatmap'][b, 0]  # [H, W]
            
            # Apply non-maximum suppression to find center points
            # This is a simple version - you can use more sophisticated methods
            centers = self._find_center_points(
                center_heatmap, 
                threshold=confidence_threshold,
                nms_kernel_size=7,
                max_centers=max_instances
            )
            
            # Skip if no centers found
            if len(centers) == 0:
                instance_results.append({
                    'masks': torch.zeros((0, height, width), device=outputs['sem_logits'].device),
                    'boxes': torch.zeros((0, 4), device=outputs['sem_logits'].device),
                    'scores': torch.zeros(0, device=outputs['sem_logits'].device),
                    'classes': torch.zeros(0, dtype=torch.long, device=outputs['sem_logits'].device)
                })
                continue
            
            # 2. For each center point, get embeddings around it
            instance_masks = []
            center_scores = []
            instance_boxes = []
            instance_classes = []
            
            # Process mask coefficients based on your model architecture
            # Option 1: If mask_coeffs are directly instance masks
            if outputs['mask_coeffs'].shape[1] == outputs['boxes'].shape[1]:
                # Mask coeffs represent actual masks
                instance_masks = outputs['mask_coeffs'][b]  # [N, H, W]
                
                # Apply sigmoid if needed (depends on if BCEWithLogitsLoss or BCE was used)
                instance_masks = torch.sigmoid(instance_masks)
                
                # Get boxes and scores from model output
                instance_boxes = outputs['boxes'][b]  # [N, 4]
                cls_scores = outputs['cls_scores'][b]  # [N, C]
                
                # Get class predictions and confidence scores
                instance_classes = torch.argmax(cls_scores, dim=1)
                scores = torch.max(cls_scores, dim=1)[0]
                
            # Option 2: If mask_coeffs need to be decoded with embeddings
            else:
                # This branch handles the case where mask_coeffs are coefficients 
                # that need to be combined with instance embeddings at center points
                embeddings = outputs['instance_embeddings'][b]  # [E, H, W]
                
                for i, (y, x, score) in enumerate(centers):
                    # Get embedding at center point
                    center_embedding = embeddings[:, y, x]  # [E]
                    
                    # Use similarity between center embedding and all embeddings to create mask
                    similarity = self._compute_embedding_similarity(center_embedding, embeddings)
                    
                    # Threshold similarity to get instance mask
                    mask = (similarity > 0.5).float()
                    
                    # Apply post-processing to clean up mask
                    mask = self._refine_mask(mask, semantic_pred[b])
                    
                    # Calculate bounding box from mask
                    box = self._mask_to_box(mask)
                    
                    # Determine class from semantic mask in this region
                    mask_area = mask.sum()
                    if mask_area > 0:
                        # Get most common class in the masked region
                        masked_semantic = semantic_pred[b] * mask
                        unique_classes, counts = torch.unique(masked_semantic, return_counts=True)
                        if len(counts) > 0:
                            # Remove background class (0)
                            if 0 in unique_classes:
                                zero_idx = (unique_classes == 0).nonzero().item()
                                unique_classes = torch.cat([unique_classes[:zero_idx], unique_classes[zero_idx+1:]])
                                counts = torch.cat([counts[:zero_idx], counts[zero_idx+1:]])
                            
                            if len(counts) > 0:
                                # Get class with highest count
                                instance_class = unique_classes[counts.argmax()]
                            else:
                                # Default to background
                                instance_class = torch.tensor(0, device=unique_classes.device)
                        else:
                            # Default to background
                            instance_class = torch.tensor(0, device=semantic_pred.device)
                    else:
                        # Empty mask - default to background
                        instance_class = torch.tensor(0, device=semantic_pred.device)
                    
                    # Store results
                    instance_masks.append(mask)
                    center_scores.append(score)
                    instance_boxes.append(box)
                    instance_classes.append(instance_class)
                
                # Stack results if we have any
                if instance_masks:
                    instance_masks = torch.stack(instance_masks)
                    center_scores = torch.stack(center_scores)
                    instance_boxes = torch.stack(instance_boxes)
                    instance_classes = torch.stack(instance_classes)
                else:
                    # Empty results
                    instance_masks = torch.zeros((0, height, width), device=outputs['sem_logits'].device)
                    center_scores = torch.zeros(0, device=outputs['sem_logits'].device)
                    instance_boxes = torch.zeros((0, 4), device=outputs['sem_logits'].device)
                    instance_classes = torch.zeros(0, dtype=torch.long, device=outputs['sem_logits'].device)
            
            # 3. Apply NMS to remove overlapping instances
            if len(instance_masks) > 0:
                keep_indices = self._non_maximum_suppression(
                    instance_boxes, center_scores, nms_threshold
                )
                
                instance_masks = instance_masks[keep_indices]
                instance_boxes = instance_boxes[keep_indices]
                instance_scores = center_scores[keep_indices]
                instance_classes = instance_classes[keep_indices]
            else:
                instance_masks = torch.zeros((0, height, width), device=outputs['sem_logits'].device)
                instance_boxes = torch.zeros((0, 4), device=outputs['sem_logits'].device)
                instance_scores = torch.zeros(0, device=outputs['sem_logits'].device)
                instance_classes = torch.zeros(0, dtype=torch.long, device=outputs['sem_logits'].device)
            
            # Store results for this image
            instance_results.append({
                'masks': instance_masks,
                'boxes': instance_boxes,
                'scores': instance_scores,
                'classes': instance_classes
            })
        
        return instance_results

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