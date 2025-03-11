# Copyright 2025 Jurrian Doornbos
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from multissl.models.seghead import SegmentationModel, SegmentationHead


class DomainAdapter(nn.Module):
    """
    Domain adapter module that learns to adapt features from one domain to another.
    Implemented as a residual bottleneck module for efficient adaptation.
    Uses GroupNorm instead of BatchNorm for better stability with small datasets.
    """
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        hidden_dim = max(in_channels // reduction, 16)  # Ensure minimum channel count
        num_groups = min(32, hidden_dim)  # Ensure we don't have more groups than channels
        
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 1),
            nn.GroupNorm(num_groups=num_groups, num_channels=hidden_dim),  # GroupNorm instead of BatchNorm
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, in_channels, 1),
            nn.GroupNorm(num_groups=min(32, in_channels), num_channels=in_channels)  # GroupNorm
        )
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        return self.relu(x + self.bottleneck(x))  # Residual connection


class AdaptiveSegmentationHead(SegmentationHead):
    """
    Extended segmentation head with domain adapters at each level.
    Uses GroupNorm instead of BatchNorm for better stability with small datasets.
    """
    def __init__(self, layer1_channels, layer2_channels, layer3_channels, layer4_channels, 
                 num_classes=2, img_size=224):
        super().__init__(layer1_channels, layer2_channels, layer3_channels, layer4_channels,
                         num_classes, img_size)
        
        # Replace BatchNorm with GroupNorm in the upsampling paths
        self.up4_3 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=min(32, 256), num_channels=256),
            nn.ReLU(inplace=True)
        )
        
        self.up3_2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=min(32, 128), num_channels=128),
            nn.ReLU(inplace=True)
        )
        
        self.up2_1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=min(32, 64), num_channels=64),
            nn.ReLU(inplace=True)
        )
        
        self.final_up = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=min(16, 32), num_channels=32),
            nn.ReLU(inplace=True)
        )
        
        # Add domain adapters to process features before decoding
        self.adapter1 = DomainAdapter(64)   # After reduce1
        self.adapter2 = DomainAdapter(128)  # After reduce2
        self.adapter3 = DomainAdapter(256)  # After reduce3
        self.adapter4 = DomainAdapter(512)  # After reduce4
        
    def forward(self, features):
        """Process features through adapters and decoder path with skip connections"""
        # Extract and reduce features from backbone
        f1 = self.act(self.reduce1(features['layer1']))  # 1/4 resolution
        f2 = self.act(self.reduce2(features['layer2']))  # 1/8 resolution
        f3 = self.act(self.reduce3(features['layer3']))  # 1/16 resolution
        f4 = self.act(self.reduce4(features['layer4']))  # 1/32 resolution
        
        # Pass through domain adapters
        f1 = self.adapter1(f1)
        f2 = self.adapter2(f2)
        f3 = self.adapter3(f3)
        f4 = self.adapter4(f4)
        
        # Decoder path with skip connections (UNet style)
        # Layer 4 -> Layer 3 with explicit size matching
        x = self.up4_3(f4)
        x = F.interpolate(x, size=f3.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, f3], dim=1)
        x = self.act(self.decode3(x))
        
        # Layer 3 -> Layer 2 with explicit size matching
        x = self.up3_2(x)
        x = F.interpolate(x, size=f2.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, f2], dim=1)
        x = self.act(self.decode2(x))
        
        # Layer 2 -> Layer 1 with explicit size matching
        x = self.up2_1(x)
        x = F.interpolate(x, size=f1.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, f1], dim=1)
        x = self.act(self.decode1(x))
        
        # Final upsampling to original resolution
        x = self.final_up(x)
        x = F.interpolate(x, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)
        x = self.classifier(x)
        
        return x

class DomainAdaptiveSegmentationModel(SegmentationModel):
    """
    Segmentation model with domain adaptation capabilities
    """
    def __init__(
        self, 
        backbone_type="resnet50",
        pretrained_path=None,
        in_channels=4,
        num_classes=2, 
        img_size=224,
        lr=1e-4, 
        weight_decay=1e-5,
        consistency_weight=1.0,
        consistency_rampup=10,  # Epochs for ramping up consistency weight
        class_weights = [1.0,1.0]
    ):
        # Initialize with parent class but don't create the segmentation head yet
        super(SegmentationModel, self).__init__()  # Call LightningModule's init
        
        self.backbone_type = backbone_type
        self.img_size = img_size
        self.consistency_weight = consistency_weight
        self.consistency_rampup = consistency_rampup
        self._current_epoch = 0
        
        # Initialize the appropriate backbone (reusing parent class initialization logic)
        if backbone_type.startswith("resnet"):
            if backbone_type == "resnet18":
                from torchvision import models
                resnet = models.resnet18(weights=None)
                self._modify_resnet_for_4_channels(resnet, in_channels)
                self.feature_extractor = self._get_appropriate_extractor(resnet)
            elif backbone_type == "resnet50":
                from torchvision import models
                resnet = models.resnet50(weights=None)
                self._modify_resnet_for_4_channels(resnet, in_channels)
                self.feature_extractor = self._get_appropriate_extractor(resnet)
            else:
                raise ValueError(f"Unsupported ResNet type: {backbone_type}")
                
        elif backbone_type.startswith("vit"):
            from timm import create_model
            if backbone_type == "vit-s":
                vit = create_model(
                    "vit_small_patch16_224",
                    pretrained=False,
                    in_chans=in_channels,
                    num_classes=0
                )
                self.feature_extractor = self._get_appropriate_extractor(vit)
            else:
                raise ValueError(f"Unsupported ViT type: {backbone_type}")
                
        elif backbone_type.startswith("swin"):
            from timm import create_model
            if backbone_type == "swin-tiny":
                swin = create_model(
                    "swin_tiny_patch4_window7_224",
                    pretrained=False,
                    in_chans=in_channels,
                    num_classes=0
                )
                self.feature_extractor = self._get_appropriate_extractor(swin)
            else:
                raise ValueError(f"Unsupported Swin type: {backbone_type}")
        else:
            raise ValueError(f"Unsupported backbone type: {backbone_type}")
        
        # Load pretrained weights if provided
        if pretrained_path:
            self._load_pretrained_weights(pretrained_path)
        
        # Freeze backbone weights initially if using pretrained model
        if pretrained_path:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
        
        # Detect feature dimensions by running a forward pass
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, img_size, img_size)
            features = self.feature_extractor(dummy_input)
            
            # Extract channel dimensions from features
            layer1_channels = features['layer1'].size(1)
            layer2_channels = features['layer2'].size(1)
            layer3_channels = features['layer3'].size(1)
            layer4_channels = features['layer4'].size(1)
            
            print(f"Detected feature dimensions: layer1={layer1_channels}, layer2={layer2_channels}, "
                  f"layer3={layer3_channels}, layer4={layer4_channels}")
        
        # Initialize adaptive segmentation head with correct dimensions
        self.seg_head = AdaptiveSegmentationHead(
            layer1_channels=layer1_channels,
            layer2_channels=layer2_channels,
            layer3_channels=layer3_channels,
            layer4_channels=layer4_channels,
            num_classes=num_classes,
            img_size=img_size
        )
        
        # Training parameters
        self.lr = lr
        self.weight_decay = weight_decay
        self.num_classes = num_classes
        self.class_weights = torch.tensor(class_weights)

        # For loss balancing - will be dynamically updated
        self.supervised_loss_scale = 1.0
        self.supervised_loss_ema = None  # Will track supervised loss with EMA
        self.consistency_loss_ema = None  # Will track consistency loss with EMA
        self.ema_alpha = 0.9  # EMA smoothing factor
    
    def _get_appropriate_extractor(self, backbone):
        """Get the appropriate feature extractor for the backbone type"""
        if self.backbone_type.startswith("resnet"):
            from multissl.models.seghead import ResNetExtractor
            return ResNetExtractor(backbone)
        elif self.backbone_type.startswith("vit"):
            from multissl.models.seghead import ViTExtractor
            return ViTExtractor(backbone, img_size=self.img_size)
        elif self.backbone_type.startswith("swin"):
            from multissl.models.seghead import SwinExtractor
            return SwinExtractor(backbone)
    
    def _modify_resnet_for_4_channels(self, resnet, in_channels):
        """Modifies the first ResNet conv layer to accept multi-channel input."""
        if in_channels == 3:
            return  # No modification needed
            
        old_conv = resnet.conv1
        new_conv = nn.Conv2d(
            in_channels,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False
        )

        # Initialize with pretrained weights for first 3 channels if available
        if hasattr(old_conv, 'weight'):
            with torch.no_grad():
                new_conv.weight[:, :3] = old_conv.weight
                if in_channels > 3:
                    # For additional channels beyond RGB, initialize with average of RGB weights
                    # or just the red channel for the 4th channel (common for NIR)
                    for i in range(3, in_channels):
                        new_conv.weight[:, i] = old_conv.weight[:, 0]  # Use red channel for NIR
        
        resnet.conv1 = new_conv
    
    def _load_pretrained_weights(self, checkpoint_path):
        """Load pretrained weights from a checkpoint file"""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            
            # If backbone is from FastSiam, we need to clean the state dict
            if any(k.startswith('backbone.') for k in state_dict.keys()):
                # For ResNet in FastSiam
                if self.backbone_type.startswith('resnet'):
                    # Strip 'backbone.' prefix and remove projection/prediction head weights
                    cleaned_state_dict = {
                        k.replace('backbone.', ''): v for k, v in state_dict.items()
                        if k.startswith('backbone.') and not any(x in k for x in ['projection_head', 'prediction_head'])
                    }
                    
                    # Load weights to feature extractor components
                    missing, unexpected = self.feature_extractor.load_state_dict(cleaned_state_dict, strict=False)
                    print(f"Loaded pretrained weights. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
                
                # For ViT/Swin in FastSiam
                elif self.backbone_type.startswith('vit') or self.backbone_type.startswith('swin'):
                    # For ViT models, we need to load the weights directly to the backbone
                    cleaned_state_dict = {
                        k.replace('backbone.', ''): v for k, v in state_dict.items()
                        if k.startswith('backbone.') and not any(x in k for x in ['projection_head', 'prediction_head'])
                    }
                    
                    # Load weights
                    if self.backbone_type.startswith('vit'):
                        missing, unexpected = self.feature_extractor.vit.load_state_dict(cleaned_state_dict, strict=False)
                    else:  # Swin
                        missing, unexpected = self.feature_extractor.swin.load_state_dict(cleaned_state_dict, strict=False)
                    
                    print(f"Loaded pretrained weights. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
    
    def get_consistency_weight(self):
        """Ramps up the consistency weight during training"""
        # Linear ramp-up
        if self.consistency_rampup > 0:
            return self.consistency_weight * min(1.0, self._current_epoch / self.consistency_rampup)
        return self.consistency_weight
    
    def training_step(self, batch, batch_idx):
        """Training step supporting both labeled and unlabeled data with balanced loss scaling"""
        labeled_batch, unlabeled_batch = batch
        total_loss = 0
        
        # Track which losses are calculated for this batch
        has_supervised_loss = False
        has_consistency_loss = False
        
        # Process labeled data (if we have non-empty tensors)
        images_l, masks_l = labeled_batch
        if images_l.size(0) > 0:
            logits_l = self(images_l)
            supervised_loss = self.combined_loss(logits_l, masks_l)
            total_loss += supervised_loss
            self.log("train_supervised_loss", supervised_loss)
            has_supervised_loss = True
        
        # Process unlabeled data (if we have non-empty tensors)
        if unlabeled_batch.size(0) > 0:
            images_u = unlabeled_batch
            
            # Generate pseudo labels with model's current state
            with torch.no_grad():
                logits_u = self(images_u)
                pseudo_masks = torch.argmax(logits_u, dim=1)
                
                # Calculate confidence scores (max probability)
                probs = F.softmax(logits_u, dim=1)
                confidence = torch.max(probs, dim=1)[0]
                
                # Only use confident predictions (threshold: 0.75)
                confidence_mask = (confidence > 0.75)
            
            # Apply model again with gradient tracking
            logits_u_aug = self(images_u)
            
            # Apply consistency loss only to confident pixels
            consistency_loss = F.cross_entropy(
                logits_u_aug, 
                pseudo_masks, 
                reduction='none'
            )
            
            # Apply confidence mask and calculate mean
            if confidence_mask.sum() > 0:  # Avoid division by zero
                consistency_loss = (consistency_loss * confidence_mask.float()).sum() / confidence_mask.sum()
                
                # Apply consistency weight
                weighted_consistency_loss = self.get_consistency_weight() * consistency_loss
                total_loss += weighted_consistency_loss
                
                self.log("train_consistency_loss", consistency_loss)
                self.log("train_consistency_weight", self.get_consistency_weight())
                has_consistency_loss = True
        
        # Handle the case where we have only one type of loss
        if has_supervised_loss and not has_consistency_loss:
            # Only supervised loss exists, keep it as is
            pass
        elif not has_supervised_loss and has_consistency_loss:
            # Only consistency loss exists, scale it to be comparable to supervised loss
            # The scaling helps maintain similar gradient magnitudes between different batch types
            total_loss = total_loss * self.supervised_loss_scale
        
        self.log("train_total_loss", total_loss)
        return total_loss
    
    def on_train_epoch_start(self):
        """Update current epoch counter"""
        self._current_epoch = self.trainer.current_epoch
    
    # Reuse other methods from parent class
    def forward(self, x):
        """Forward pass through backbone and segmentation head"""
        features = self.feature_extractor(x)
        segmentation_map = self.seg_head(features)
        return segmentation_map
    
    def configure_optimizers(self):
        """Configure optimizer with learning rate scheduler"""
        # Use separate parameter groups with different learning rates
        backbone_params = []
        adapter_params = []
        decoder_params = []
        
        # Separate parameters
        for name, param in self.named_parameters():
            if 'feature_extractor' in name:
                backbone_params.append(param)
            elif 'adapter' in name:
                adapter_params.append(param)
            else:
                decoder_params.append(param)
        
        # Create optimizer with parameter groups
        optimizer = torch.optim.AdamW([
            {'params': backbone_params, 'lr': self.lr * 0.1},  # Lower LR for backbone
            {'params': adapter_params, 'lr': self.lr * 0.5},   # Medium LR for adapters
            {'params': decoder_params, 'lr': self.lr}          # Full LR for decoder
        ], weight_decay=self.weight_decay)
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5, 
            patience=5,
            verbose=True
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train_total_loss",
                "interval": "epoch"
            }
        }
    
    def dice_loss(self, outputs, targets, smooth=1.0, weight = torch.tensor([1.0,1.0])):
        num_classes = outputs.size(1)
        
        # One-hot encode targets
        targets_one_hot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()
        
        # Apply softmax to get probabilities
        probs = F.softmax(outputs, dim=1)
        
        # Calculate Dice loss
        intersection = (probs * targets_one_hot).sum(dim=(0, 2, 3))
        union = probs.sum(dim=(0, 2, 3)) + targets_one_hot.sum(dim=(0, 2, 3))
        
        dice_score = (2.0 * intersection + smooth) / (union + smooth)
        
        # Focus more on vine class (class 1)

        weighted_dice = dice_score * weight.to(outputs.device)
        
        return 1.0 - weighted_dice.mean()
        
    def combined_loss(self, outputs, targets, ce_weight=0.5, dice_weight=0.5):
        # Define class weights for CE loss
        cw = self.class_weights.to(outputs.device)
        # Calculate CE loss
        ce_loss = F.cross_entropy(outputs, targets, weight=cw)
        
        # Calculate Dice loss
        dice = self.dice_loss(outputs, targets, weight = cw)
        
        # Combine losses
        return ce_weight * ce_loss + dice_weight * dice