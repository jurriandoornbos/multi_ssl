import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union, Dict
from .msrgb_convnext_upernet import PPM, MSRGBConvNeXtFeatureExtractor
import pytorch_lightning as pl


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
        pretrained_backbone: Optional[str] = None
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
        pretrained_backbone: Optional[str] = None
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
            pretrained_backbone=pretrained_backbone
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
        self.box_criterion = nn.SmoothL1Loss()
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
        
        # Instance segmentation targets (in a real implementation)
        # These would come from your dataset
        center_target = batch.get('centers', torch.zeros_like(semantic_target, dtype=torch.float32))
        instance_masks = batch.get('instance_masks', None)
        instance_classes = batch.get('instance_classes', None)
        boxes = batch.get('boxes', None)
        
        # Forward pass
        outputs = self(rgb=rgb, ms=ms)
        
        # Calculate semantic segmentation loss
        semantic_loss = self.semantic_criterion(outputs['sem_logits'], semantic_target)
        
        # Calculate center point prediction loss
        center_loss = self.center_criterion(outputs['center_heatmap'], center_target)
        
        # Initialize instance losses
        box_loss = torch.tensor(0.0, device=self.device)
        instance_cls_loss = torch.tensor(0.0, device=self.device)
        mask_loss = torch.tensor(0.0, device=self.device)
        
        # Calculate instance losses if we have instance targets
        if instance_masks is not None and instance_classes is not None and boxes is not None:
            # Box regression loss
            box_loss = self.box_criterion(outputs['boxes'], boxes)
            
            # Instance classification loss
            instance_cls_loss = self.instance_cls_criterion(
                outputs['cls_scores'].view(-1, outputs['cls_scores'].shape[-1]),
                instance_classes.view(-1)
            )
            
            # Instance mask loss (this is a simplified version)
            # In practice, you'd need to match predicted instances with ground truth
            mask_loss = self.mask_criterion(
                outputs['mask_coeffs'], 
                instance_masks
            )
        
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