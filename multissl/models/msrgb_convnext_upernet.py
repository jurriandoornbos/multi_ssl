import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Dict, List, Optional, Tuple, Union

from .msrgb_convnext import MSRGBConvNeXtFeatureExtractor

class PPM(nn.Module):
    """
    Pyramid Pooling Module (PPM) from PSPNet
    
    Performs pooling at multiple scales and concatenates the results
    to capture global context information.
    """
    def __init__(self, in_dim, reduction_dim, bins=(1, 2, 3, 6)):
        super(PPM, self).__init__()
        self.features = []
        for bin in bins:
            self.features.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(bin),
                    nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(reduction_dim),
                    nn.ReLU(inplace=True)
                )
            )
        self.features = nn.ModuleList(self.features)

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
        return torch.cat(out, 1)

class UPerNet(nn.Module):
    """
    Unified Perceptual Parsing Network (UPerNet) segmentation head
    
    Combines feature maps from different stages of the backbone
    using Feature Pyramid Network (FPN) and Pyramid Pooling Module (PPM).
    """
    def __init__(
        self,
        in_channels: List[int],
        num_classes: int,
        fpn_dim: int = 256,
        ppm_bins: Tuple[int] = (1, 2, 3, 6),
        aux_heads: bool = True
    ):
        super(UPerNet, self).__init__()
        self.in_channels = in_channels
        self.aux_heads = aux_heads
        
        # Print input channel dimensions for debugging
        print(f"UPerNet input channels: {in_channels}")
        
        # PPM module on the last feature map
        ppm_reduction_dim = fpn_dim // len(ppm_bins)
        self.ppm = PPM(in_channels[-1], ppm_reduction_dim, ppm_bins)
        ppm_out_dim = in_channels[-1] + ppm_reduction_dim * len(ppm_bins)
        
        # FPN lateral connections (bottom-up path)
        self.fpn_in = nn.ModuleList()
        for in_c in in_channels[:-1]:  # Skip the last one as it's handled by PPM
            self.fpn_in.append(nn.Sequential(
                nn.Conv2d(in_c, fpn_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(fpn_dim),
                nn.ReLU(inplace=True)
            ))
            
        # FPN output connections (top-down path)
        self.fpn_out = nn.ModuleList()
        for _ in range(len(in_channels) - 1):  # -1 because the last one is handled by PPM
            self.fpn_out.append(nn.Sequential(
                nn.Conv2d(fpn_dim, fpn_dim, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(fpn_dim),
                nn.ReLU(inplace=True)
            ))
            
        # Handle the last feature map with PPM - adjust dimensions
        self.fpn_bottleneck = nn.Sequential(
            nn.Conv2d(ppm_out_dim, fpn_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fpn_dim),
            nn.ReLU(inplace=True)
        )
        
        # Feature fusion module
        self.fusion = nn.Sequential(
            nn.Conv2d(len(in_channels) * fpn_dim, fpn_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(fpn_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(fpn_dim, fpn_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fpn_dim),
            nn.ReLU(inplace=True)
        )
        
        # Final classifier
        self.classifier = nn.Conv2d(fpn_dim, num_classes, kernel_size=1)
        
        # Auxiliary heads for deep supervision if needed
        if aux_heads:
            self.aux_heads = nn.ModuleList([
                nn.Conv2d(fpn_dim, num_classes, kernel_size=1)
                for _ in range(len(in_channels))
            ])
        
    def forward(self, features: List[torch.Tensor], return_aux=False):
        """
        Forward pass for the UPerNet head
        
        Args:
            features: List of feature maps from the backbone [feat1, feat2, ..., featN]
                     ordered from lowest resolution to highest
            return_aux: Whether to return auxiliary predictions for deep supervision
            
        Returns:
            logits: Segmentation logits at the highest resolution
            aux_outputs: Optional auxiliary outputs at each FPN level
        """
        
        # Create a local copy of features list to avoid modifying the input
        feats = features.copy()
        
        # Apply PPM to the last (deepest) feature map
        ppm_out = self.ppm(feats[-1])
        feats[-1] = self.fpn_bottleneck(ppm_out)
        
        # Build FPN from bottom to top (high-level to low-level features)
        fpn_features = [feats[-1]]
        for i in reversed(range(len(self.in_channels) - 1)):
            feat = feats[i]
            lateral = self.fpn_in[i](feat)
            
            # Top-down pathway: upsample higher-level features
            top_down = F.interpolate(
                fpn_features[0], size=lateral.shape[2:], 
                mode='bilinear', align_corners=True
            )
            
            # Add lateral connection
            fpn_feat = lateral + top_down
            
            # Apply convolution to the summed features
            fpn_feat = self.fpn_out[i](fpn_feat)
            
            # Insert the new feature at the beginning of the list
            fpn_features.insert(0, fpn_feat)
        
        # Generate auxiliary outputs for deep supervision
        aux_outputs = None
        if self.aux_heads and return_aux:
            aux_outputs = []
            for i, feat in enumerate(fpn_features):
                aux_out = self.aux_heads[i](feat)
                aux_outputs.append(aux_out)
        
        # Upsample all feature maps to the highest resolution (level of the first feature map)
        output_size = fpn_features[0].shape[2:]
        aligned_features = []
        
        for i, feat in enumerate(fpn_features):
            if i == 0:  # Skip the first one as it's already at the highest resolution
                aligned_features.append(feat)
            else:
                # Upsample to match the resolution of the first feature map
                upsampled = F.interpolate(
                    feat, size=output_size, mode='bilinear', align_corners=True
                )
                aligned_features.append(upsampled)
        
        # Concatenate all FPN levels and apply fusion
        fused = torch.cat(aligned_features, dim=1)
        fused = self.fusion(fused)
        
        # Final classifier
        logits = self.classifier(fused)
        
        if return_aux:
            return logits, aux_outputs
        return logits

class MSRGBConvNeXtUPerNet(nn.Module):
    """
    Combined MSRGBConvNeXt backbone with UPerNet segmentation head
    """
    def __init__(
        self,
        num_classes: int,
        rgb_in_channels: int = 3,
        ms_in_channels: int = 5,
        model_size: str = 'tiny',  # 'tiny', 'small', 'base', 'large'
        fusion_strategy: str = 'hierarchical',
        fusion_type: str = 'attention',
        fpn_dim: int = 256,
        drop_path_rate: float = 0.1,
        aux_heads: bool = True,
        use_aux_loss: bool = True,
        aux_weight: float = 0.4,
        pretrained_backbone: Optional[str] = None
    ):
        super(MSRGBConvNeXtUPerNet, self).__init__()
        
        # Create backbone
        self.backbone = MSRGBConvNeXtFeatureExtractor(
            model_name=model_size,
            rgb_in_channels=rgb_in_channels,
            ms_in_channels=ms_in_channels,
            fusion_strategy=fusion_strategy,
            fusion_type=fusion_type,
            drop_path_rate=drop_path_rate
        )
        
        # Get feature dimensions from backbone and print them for debugging
        feature_dims = [
            self.backbone.feature_dims[f'layer{i+1}'] 
            for i in range(len(self.backbone.feature_dims) if 'flat' not in self.backbone.feature_dims 
                           else len(self.backbone.feature_dims) - 1)
        ]
        
        print(f"Feature dimensions from backbone: {feature_dims}")
        
        # Create segmentation head
        self.decode_head = UPerNet(
            in_channels=feature_dims,
            num_classes=num_classes,
            fpn_dim=fpn_dim,
            aux_heads=aux_heads
        )
        
        # Loss settings
        self.use_aux_loss = use_aux_loss
        self.aux_weight = aux_weight
        
        # Initialize from pretrained weights if provided
        if pretrained_backbone:
            self._load_pretrained_backbone(pretrained_backbone)
    
    def forward(self, rgb=None, ms=None, return_aux=False):
        """
        Forward pass through backbone and segmentation head
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
        
        # Convert to list ordered from lowest to highest resolution (for UPerNet)
        # Note: This ordering might need adjustment based on your ConvNeXt implementation
        features = []
        for i in range(len(feat_dict) if 'flat' not in feat_dict else len(feat_dict) - 1):
            key = f'layer{i+1}'
            if key in feat_dict:
                features.append(feat_dict[key])
        # Apply segmentation head
        if return_aux or self.use_aux_loss:
            logits, aux_outputs = self.decode_head(features, return_aux=True)
            
            # Upsample logits to match input size
            if logits.shape[2:] != input_size:
                logits = F.interpolate(
                    logits, size=input_size, mode='bilinear', align_corners=True
                )
                
                # Also upsample auxiliary outputs
                aux_outputs = [
                    F.interpolate(aux, size=input_size, mode='bilinear', align_corners=True)
                    for aux in aux_outputs
                ]
                
            if return_aux:
                return logits, aux_outputs
            return logits
        else:
            logits = self.decode_head(features, return_aux=False)
            
            # Upsample logits to match input size
            if logits.shape[2:] != input_size:
                logits = F.interpolate(
                    logits, size=input_size, mode='bilinear', align_corners=True
                )
                
            return logits
                # Load checkpoint if provided
        
    def _load_pretrained_backbone(self, checkpoint_path):
        """Load weights from a PyTorch Lightning checkpoint file"""
        print(f"Loading checkpoint from {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Extract state dict - handle both direct state_dict and Lightning format
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            # Remove 'backbone.' prefix if it exists (common in Lightning models)
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('backbone.'):
                    new_key = key[len('backbone.'):]
                    new_state_dict[new_key] = value
                # Also handle 'feature_extractor.' prefix
                elif key.startswith('feature_extractor.'):
                    new_key = key[len('feature_extractor.'):]
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value
            state_dict = new_state_dict
        else:
            state_dict = checkpoint
        
        # Load weights to backbone
        missing_keys, unexpected_keys = self.backbone_model.load_state_dict(state_dict, strict=False)
        
        if len(missing_keys) > 0:
            print(f"Missing keys: {missing_keys}")
        if len(unexpected_keys) > 0:
            print(f"Unexpected keys: {unexpected_keys}")


class MSRGBConvNeXtUPerNetModule(pl.LightningModule):
    """
    PyTorch Lightning module for MSRGBConvNeXtUPerNet
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
        use_aux_loss: bool = True,
        aux_weight: float = 0.4,
        class_weights: Optional[List[float]] = None,
        pretrained_backbone: Optional[str] = None
    ):
        super(MSRGBConvNeXtUPerNetModule, self).__init__()
        
        # Create the model
        self.model = MSRGBConvNeXtUPerNet(
            num_classes=num_classes,
            rgb_in_channels=rgb_in_channels,
            ms_in_channels=ms_in_channels,
            model_size=model_size,
            fusion_strategy=fusion_strategy,
            fusion_type=fusion_type,
            use_aux_loss=use_aux_loss,
            aux_weight=aux_weight,
            pretrained_backbone=pretrained_backbone
        )
        
        # Training parameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.use_aux_loss = use_aux_loss
        self.aux_weight = aux_weight
        
        # Setup loss function with optional class weights
        if class_weights is not None:
            self.register_buffer('class_weights', torch.tensor(class_weights))
            self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        # Save hyperparameters for logging
        self.save_hyperparameters(ignore=['model'])
        
    def forward(self, rgb=None, ms=None, return_aux=False):
        return self.model(rgb=rgb, ms=ms, return_aux=return_aux)
        
    def training_step(self, batch, batch_idx):
        # Extract inputs and target
        rgb = batch.get('rgb')
        ms = batch.get('ms')
        target = batch['mask']
        
        # Forward pass
        if self.use_aux_loss:
            logits, aux_outputs = self(rgb=rgb, ms=ms, return_aux=True)
            
            # Main loss
            main_loss = self.criterion(logits, target)
            
            # Auxiliary losses
            aux_losses = 0
            for aux_output in aux_outputs:
                aux_losses += self.criterion(aux_output, target)
            
            # Combined loss
            loss = main_loss + self.aux_weight * (aux_losses / len(aux_outputs))
            
            # Log components
            self.log('train_main_loss', main_loss, on_step=True, on_epoch=True, prog_bar=True)
            self.log('train_aux_loss', aux_losses / len(aux_outputs), on_step=True, on_epoch=True)
        else:
            logits = self(rgb=rgb, ms=ms)
            loss = self.criterion(logits, target)
        
        # Log total loss
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
        
    def validation_step(self, batch, batch_idx):
        # Extract inputs and target
        rgb = batch.get('rgb')
        ms = batch.get('ms')
        target = batch['mask']
        
        # Forward pass (no aux outputs during validation)
        logits = self(rgb=rgb, ms=ms)
        
        # Calculate loss
        loss = self.criterion(logits, target)
        
        # Log metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        
        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        accuracy = (preds == target).float().mean()
        self.log('val_accuracy', accuracy, on_step=False, on_epoch=True, prog_bar=True)
        
        # Calculate IoU for each class
        iou_per_class = []
        for cls in range(logits.shape[1]):
            intersection = ((preds == cls) & (target == cls)).float().sum()
            union = ((preds == cls) | (target == cls)).float().sum()
            iou = intersection / (union + 1e-6)
            iou_per_class.append(iou)
            self.log(f'val_iou_class{cls}', iou, on_step=False, on_epoch=True)
        
        # Mean IoU
        mean_iou = torch.stack(iou_per_class).mean()
        self.log('val_mean_iou', mean_iou, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
        
    def configure_optimizers(self):
        # Set up separate parameter groups with different learning rates if needed
        backbone_params = []
        head_params = []
        
        # Separate backbone and segmentation head parameters
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
                'monitor': 'val_loss',
                'interval': 'epoch'
            }
        }
    
    def test_step(self, batch, batch_idx):
        # Similar to validation_step but includes more comprehensive metrics
        rgb = batch.get('rgb')
        ms = batch.get('ms')
        target = batch['mask']
        
        # Forward pass
        logits = self(rgb=rgb, ms=ms)
        
        # Calculate loss
        loss = self.criterion(logits, target)
        
        # Get predictions
        preds = torch.argmax(logits, dim=1)
        
        # Calculate accuracy
        accuracy = (preds == target).float().mean()
        
        # Calculate metrics for each class
        num_classes = logits.shape[1]
        metrics_per_class = []
        
        for c in range(num_classes):
            # True positives, false positives, false negatives
            true_pos = ((preds == c) & (target == c)).sum().float()
            false_pos = ((preds == c) & (target != c)).sum().float()
            false_neg = ((preds != c) & (target == c)).sum().float()
            
            # Calculate precision, recall, F1, IoU
            precision = true_pos / (true_pos + false_pos + 1e-6)
            recall = true_pos / (true_pos + false_neg + 1e-6)
            f1 = 2 * precision * recall / (precision + recall + 1e-6)
            iou = true_pos / (true_pos + false_pos + false_neg + 1e-6)
            
            metrics_per_class.append({
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'iou': iou
            })
        
        # Calculate mean metrics
        mean_precision = torch.mean(torch.stack([m['precision'] for m in metrics_per_class]))
        mean_recall = torch.mean(torch.stack([m['recall'] for m in metrics_per_class]))
        mean_f1 = torch.mean(torch.stack([m['f1'] for m in metrics_per_class]))
        mean_iou = torch.mean(torch.stack([m['iou'] for m in metrics_per_class]))
        
        # Log metrics
        self.log('test_loss', loss)
        self.log('test_accuracy', accuracy)
        self.log('test_mean_precision', mean_precision)
        self.log('test_mean_recall', mean_recall)
        self.log('test_mean_f1', mean_f1)
        self.log('test_mean_iou', mean_iou)
        
        for i, metrics in enumerate(metrics_per_class):
            self.log(f'test_class{i}_precision', metrics['precision'])
            self.log(f'test_class{i}_recall', metrics['recall'])
            self.log(f'test_class{i}_f1', metrics['f1'])
            self.log(f'test_class{i}_iou', metrics['iou'])
        
        return {
            'test_loss': loss,
            'test_accuracy': accuracy,
            'metrics_per_class': metrics_per_class,
            'mean_iou': mean_iou
        }