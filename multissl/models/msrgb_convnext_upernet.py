import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Union, Tuple
import pytorch_lightning as pl
import torchmetrics
from torchmetrics import JaccardIndex, Accuracy, Precision, Recall, F1Score

from .msrgb_convnext import MSRGBConvNeXtFeatureExtractor

class PPM(nn.Module):
    """
    Pyramid Pooling Module (PPM) as used in PSPNet and UPerNet
    """
    def __init__(self, in_channels, pool_sizes=(1, 2, 3, 6), norm_layer=nn.BatchNorm2d):
        super(PPM, self).__init__()
        self.pool_sizes = pool_sizes
        out_channels = in_channels // len(pool_sizes)
        
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(pool_size),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                norm_layer(out_channels),
                nn.ReLU(inplace=True)
            )
            for pool_size in pool_sizes
        ])
        
    def forward(self, x):
        h, w = x.shape[2], x.shape[3]
        features = [x]
        
        for stage in self.stages:
            feat = stage(x)
            feat = F.interpolate(feat, size=(h, w), mode='bilinear', align_corners=True)
            features.append(feat)
            
        return torch.cat(features, dim=1)

class FPNFPN(nn.Module):
    """
    Feature Pyramid Network (FPN) with lateral connections
    """
    def __init__(self, in_channels_list, out_channels, norm_layer=nn.BatchNorm2d):
        super(FPNFPN, self).__init__()
        
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        
        # Lateral connections - reduce each input to the same channel dimension
        for in_channels in in_channels_list:
            self.lateral_convs.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                norm_layer(out_channels),
                nn.ReLU(inplace=True)
            ))
        
        # FPN convs - apply 3x3 convolution to each output
        for _ in range(len(in_channels_list)):
            self.fpn_convs.append(nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                norm_layer(out_channels),
                nn.ReLU(inplace=True)
            ))
            
    def forward(self, inputs):
        # Apply lateral convs to create same channel features
        laterals = [conv(inp) for conv, inp in zip(self.lateral_convs, inputs)]
        
        # Top-down pathway with upsampling
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            h, w = laterals[i-1].shape[2:]
            laterals[i-1] = laterals[i-1] + F.interpolate(
                laterals[i], size=(h, w), mode='bilinear', align_corners=True)
            
        # Apply FPN convs
        outs = [fpn_conv(lateral) for fpn_conv, lateral in zip(self.fpn_convs, laterals)]
        
        return outs

class UPerNetHead(nn.Module):
    """
    UPerNet Segmentation Head combining PPM and FPN for multi-scale feature fusion
    """
    def __init__(
        self,
        feature_dims: Dict[str, int],
        num_classes: int,
        fpn_channels: int = 256,
        dropout: float = 0.1,
        aux_loss: bool = False,
        pool_scales: Tuple[int] = (1, 2, 3, 6),
        norm_layer: nn.Module = nn.BatchNorm2d
    ):
        super(UPerNetHead, self).__init__()
        
        # Extract feature dimensions from backbone
        self.in_channels = list(feature_dims.values())  # Ordered list of channel dims
        
        # PPM module on the last layer
        self.ppm = PPM(self.in_channels[-1], pool_sizes=pool_scales, norm_layer=norm_layer)
        
        # Calculate input channels after PPM (original + outputs from each pooling branch)
        ppm_channels = self.in_channels[-1] + self.in_channels[-1] // len(pool_scales) * len(pool_scales)
        
        # 1x1 conv after PPM to reduce channels
        self.ppm_bottleneck = nn.Sequential(
            nn.Conv2d(ppm_channels, fpn_channels, kernel_size=1, bias=False),
            norm_layer(fpn_channels),
            nn.ReLU(inplace=True)
        )
        
        # FPN to fuse features from different levels
        fpn_in_channels = self.in_channels[:-1]  # All but last layer (which uses PPM)
        fpn_in_channels.append(fpn_channels)  # Add PPM output
        self.fpn = FPNFPN(fpn_in_channels, fpn_channels, norm_layer=norm_layer)
        
        # Final fusion conv for all FPN outputs
        self.fpn_fusion = nn.Sequential(
            nn.Conv2d(fpn_channels * len(fpn_in_channels), fpn_channels, kernel_size=1, bias=False),
            norm_layer(fpn_channels),
            nn.ReLU(inplace=True)
        )
        
        # Dropout
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        
        # Final classifier
        self.classifier = nn.Conv2d(fpn_channels, num_classes, kernel_size=1)
        
        # Auxiliary loss if needed
        self.aux_loss = aux_loss
        if aux_loss:
            self.aux_classifier = nn.Sequential(
                nn.Conv2d(self.in_channels[2], fpn_channels, kernel_size=3, padding=1, bias=False),
                norm_layer(fpn_channels),
                nn.ReLU(inplace=True),
                nn.Dropout2d(dropout),
                nn.Conv2d(fpn_channels, num_classes, kernel_size=1)
            )
            
    def forward(self, features: Dict[str, torch.Tensor]):
        # Extract features in order from shallow to deep
        feat_list = [features[f'layer{i+1}'] for i in range(len(self.in_channels))]
        
        # Apply PPM to the deepest features
        ppm_out = self.ppm(feat_list[-1])
        ppm_out = self.ppm_bottleneck(ppm_out)
        
        # Replace the last feature with PPM output for FPN
        fpn_in = feat_list[:-1] + [ppm_out]
        
        # Apply FPN
        fpn_out = self.fpn(fpn_in)
        
        # Resize all FPN outputs to the size of the smallest feature map
        target_h, target_w = fpn_out[0].shape[2:]
        
        # Upsample and concat all FPN outputs
        concat_features = []
        for feat in fpn_out:
            if feat.shape[2:] != (target_h, target_w):
                feat = F.interpolate(feat, size=(target_h, target_w), 
                                    mode='bilinear', align_corners=True)
            concat_features.append(feat)
        
        # Fuse all features
        output = self.fpn_fusion(torch.cat(concat_features, dim=1))
        output = self.dropout(output)
        
        # Final classifier
        output = self.classifier(output)
        
        # Upsample to input resolution if needed (often done in the loss function)
        
        # Handle auxiliary loss if needed
        if self.aux_loss and self.training:
            aux_output = self.aux_classifier(feat_list[2])  # Use mid-level features
            return output, aux_output
        
        return output
    
class UPerNet(nn.Module):
    """
    Complete UPerNet model combining the backbone feature extractor with the UPerNet head
    """
    def __init__(
        self,
        backbone,
        num_classes: int,
        fpn_channels: int = 256,
        dropout: float = 0.1,
        aux_loss: bool = False,
        upsample_output: bool = True
    ):
        super(UPerNet, self).__init__()
        self.backbone = backbone

        self.upsample_output = upsample_output
        
        # Get feature dimensions from backbone
        self.feature_dims = backbone.feature_dims
        
        # Create segmentation head
        self.decode_head = UPerNetHead(
            feature_dims=self.feature_dims,
            num_classes=num_classes,
            fpn_channels=fpn_channels,
            dropout=dropout,
            aux_loss=aux_loss
        )
        
    def forward(self, rgb=None, ms=None):
        # Extract features from backbone
        features = self.backbone(rgb=rgb, ms=ms)
        
        # Apply segmentation head
        seg_logits = self.decode_head(features)
        
        # Upsample to input resolution if needed
        if self.upsample_output and isinstance(seg_logits, tuple):
            # Handle case with auxiliary loss
            main_out, aux_out = seg_logits
            input_size = rgb.shape[2:] if rgb is not None else ms.shape[2:]
            main_out = F.interpolate(main_out, size=input_size, 
                                    mode='bilinear', align_corners=True)
            aux_out = F.interpolate(aux_out, size=input_size, 
                                   mode='bilinear', align_corners=True)
            return main_out, aux_out
        elif self.upsample_output:
            # Regular case
            input_size = rgb.shape[2:] if rgb is not None else ms.shape[2:]
            seg_logits = F.interpolate(seg_logits, size=input_size, 
                                     mode='bilinear', align_corners=True)
            
        return seg_logits
    

class MSRGBConvNeXtUPerNet(pl.LightningModule):
    """
    PyTorch Lightning module for training and evaluating a dual-modality segmentation model
    """
    def __init__(
        self,
        model_size='tiny',
        rgb_in_channels=3,
        ms_in_channels=5,
        num_classes=2,
        drop_path_rate=0.0,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        lr_scheduler: str = 'cosine',  # 'cosine', 'step', 'poly'
        lr_warmup_epochs: int = 5,
        use_aux_loss: bool = False,
        aux_weight: float = 0.4,
        class_weights: Optional[torch.Tensor] = None,
        ignore_index: int = 255,
        pretrained_backbone = None,
        freeze_backbone: bool = False,
        upsample_output: bool = True,
        fpn_channels: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.lr_scheduler = lr_scheduler
        self.lr_warmup_epochs = lr_warmup_epochs
        self.use_aux_loss = use_aux_loss
        self.aux_weight = aux_weight
        self.upsample_output = upsample_output
        self.ignore_index = ignore_index
        

        self.backbone = MSRGBConvNeXtFeatureExtractor(
            model_name=model_size,
            rgb_in_channels=rgb_in_channels,
            ms_in_channels=ms_in_channels,
            drop_path_rate=drop_path_rate
        )

        feature_dims = self.backbone.feature_dims
        fpn_channels = fpn_channels
        dropout = dropout
        aux_loss = use_aux_loss
        # Create segmentation head
        self.decode_head = UPerNetHead(
            feature_dims=feature_dims,
            num_classes=self.num_classes,
            fpn_channels=fpn_channels,
            dropout=dropout,
            aux_loss=aux_loss
        )
        
        # Initialize loss function
        if class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=ignore_index)
        else:
            self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
            
        # Metrics
        self.train_accuracy = Accuracy(task='multiclass', num_classes=num_classes, ignore_index=ignore_index)
        self.val_accuracy = Accuracy(task='multiclass', num_classes=num_classes, ignore_index=ignore_index)
        self.val_iou = JaccardIndex(task='multiclass', num_classes=num_classes, ignore_index=ignore_index)
        self.val_f1 = F1Score(task='multiclass', num_classes=num_classes, ignore_index=ignore_index)
        
        # Per-class metrics for validation
        self.val_per_class_iou = JaccardIndex(
            task='multiclass', 
            num_classes=num_classes, 
            average=None, 
            ignore_index=ignore_index
        )
        self.val_per_class_f1 = F1Score(
            task='multiclass', 
            num_classes=num_classes, 
            average=None, 
            ignore_index=ignore_index
        )
            
            
        # Initialize from pretrained weights if provided
        if pretrained_backbone:
            self._load_pretrained_backbone(pretrained_backbone)

        if freeze_backbone:
            self.backbone.requires_grad_(False)

    def forward(self, rgb=None, ms=None):

        features = self.backbone(rgb = rgb, ms= ms)

        # Apply segmentation head
        seg_logits = self.decode_head(features)
        
        # Upsample to input resolution if needed
        if self.upsample_output and isinstance(seg_logits, tuple):
            # Handle case with auxiliary loss
            main_out, aux_out = seg_logits
            input_size = rgb.shape[2:] if rgb is not None else ms.shape[2:]
            main_out = F.interpolate(main_out, size=input_size, 
                                    mode='bilinear', align_corners=True)
            aux_out = F.interpolate(aux_out, size=input_size, 
                                   mode='bilinear', align_corners=True)
            return main_out, aux_out
        
        elif self.upsample_output:
            # Regular case
            input_size = rgb.shape[2:] if rgb is not None else ms.shape[2:]
            seg_logits = F.interpolate(seg_logits, size=input_size, 
                                     mode='bilinear', align_corners=True)
            
        return seg_logits
    
    def _get_inputs(self, batch):
        """Extract RGB and MS inputs from batch based on requirements"""
        rgb = batch.get('rgb', None)
        ms = batch.get('ms', None) 
        return rgb, ms
    
    def training_step(self, batch, batch_idx):
        rgb, ms = self._get_inputs(batch)
        target = batch['mask']
        
        # Forward pass
        output = self(rgb=rgb, ms=ms)
        
        # Compute loss
        if self.use_aux_loss and isinstance(output, tuple):
            main_out, aux_out = output
            main_loss = self.criterion(main_out, target)
            aux_loss = self.criterion(aux_out, target)
            loss = main_loss + self.aux_weight * aux_loss
            preds = main_out
        else:
            loss = self.criterion(output, target)
            preds = output
            
        # Calculate metrics
        preds_argmax = preds.argmax(dim=1)
        accuracy = self.train_accuracy(preds_argmax, target)
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_accuracy', accuracy, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        rgb, ms = self._get_inputs(batch)
        target = batch['mask']
        
        # Forward pass
        output = self(rgb=rgb, ms=ms)
        
        # Handle aux loss outputs
        if isinstance(output, tuple):
            output = output[0]  # Use only main output for validation
            
        # Compute loss
        loss = self.criterion(output, target)
        
        # Calculate metrics
        preds = output.argmax(dim=1)
        accuracy = self.val_accuracy(preds, target)
        iou = self.val_iou(preds, target)
        f1 = self.val_f1(preds, target)
        
        # Per-class metrics
        per_class_iou = self.val_per_class_iou(preds, target)
        per_class_f1 = self.val_per_class_f1(preds, target)
        
        # Log metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_accuracy', accuracy, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_iou', iou, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_f1', f1, on_step=False, on_epoch=True, prog_bar=True)
        
        # Log per-class metrics
        for i in range(self.num_classes):
            self.log(f'val_iou_class_{i}', per_class_iou[i], on_step=False, on_epoch=True)
            self.log(f'val_f1_class_{i}', per_class_f1[i], on_step=False, on_epoch=True)
            
        return loss
    
    def test_step(self, batch, batch_idx):
        # Similar to validation step
        return self.validation_step(batch, batch_idx)
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        rgb, ms = self._get_inputs(batch)
        
        # Forward pass
        output = self(rgb=rgb, ms=ms)
        
        # Handle aux loss outputs
        if isinstance(output, tuple):
            output = output[0]  # Use only main output for predictions
            
        # Get prediction mask
        pred_mask = output.argmax(dim=1)
        
        # Return prediction and optionally other fields needed for visualization
        result = {'pred': pred_mask}
        
        # Add original inputs for visualization if available
        if 'rgb' in batch:
            result['rgb'] = batch['rgb']
        if 'ms' in batch:
            result['ms'] = batch['ms']
        if 'mask' in batch:
            result['mask'] = batch['mask']
            
        return result
    
    def configure_optimizers(self):
        # Define optimizer
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Define scheduler
        scheduler = None
        
        if self.lr_scheduler == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=self.trainer.max_epochs - self.lr_warmup_epochs
            )
        elif self.lr_scheduler == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=30,
                gamma=0.1
            )
        elif self.lr_scheduler == 'poly':
            def lambda_poly(epoch):
                return (1 - epoch / self.trainer.max_epochs) ** 0.9
                
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=lambda_poly
            )
            
        # Add warmup scheduler if needed
        if self.lr_warmup_epochs > 0:
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, 
                start_factor=0.1, 
                end_factor=1.0, 
                total_iters=self.lr_warmup_epochs
            )
            
            # Combine schedulers
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer, 
                schedulers=[warmup_scheduler, scheduler], 
                milestones=[self.lr_warmup_epochs]
            )
            
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
                "monitor": "val_loss"
            }
        }
    
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
        missing_keys, unexpected_keys = self.backbone.load_state_dict(state_dict, strict=False)
        
        if len(missing_keys) > 0:
            print(f"Missing keys: {missing_keys}")
        if len(unexpected_keys) > 0:
            print(f"Unexpected keys: {unexpected_keys}")