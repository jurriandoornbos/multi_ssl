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
from einops import rearrange
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import timm

class ResNetExtractor(nn.Module):
    def __init__(self, resnet):
        super().__init__()
        # Store the components we need
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        # Store the blocks
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
    
    def forward(self, x):
        features = {}
        
        # Initial processing
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        features['initial'] = x
        
        x = self.maxpool(x)
        
        # Process through blocks
        x = self.layer1(x)
        features['layer1'] = x
        
        x = self.layer2(x)
        features['layer2'] = x
        
        x = self.layer3(x)
        features['layer3'] = x
        
        x = self.layer4(x)
        features['layer4'] = x
        
        return features

class SegmentationHead(nn.Module):
    """
    Improved Segmentation head with UNet-style decoder path and skip connections
    that explicitly handles different feature map sizes
    """
    def __init__(self, layer1_channels, layer2_channels, layer3_channels, layer4_channels, 
                 num_classes=2, img_size=224):
        super().__init__()
        
        self.img_size = img_size
        self.act = nn.ReLU(inplace=True)
        
        # Reduce channel dimensions of encoder features, but keep more channels
        self.reduce1 = nn.Conv2d(layer1_channels, 64, kernel_size=1)  # 1/4 resolution
        self.reduce2 = nn.Conv2d(layer2_channels, 128, kernel_size=1)  # 1/8 resolution
        self.reduce3 = nn.Conv2d(layer3_channels, 256, kernel_size=1)  # 1/16 resolution
        self.reduce4 = nn.Conv2d(layer4_channels, 512, kernel_size=1)  # 1/32 resolution
        
        # Decoder path (upsampling + conv)
        # Layer 4 -> Layer 3
        self.up4_3 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.decode3 = nn.Conv2d(256 + 256, 256, kernel_size=3, padding=1)
        
        # Layer 3 -> Layer 2
        self.up3_2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.decode2 = nn.Conv2d(128 + 128, 128, kernel_size=3, padding=1)
        
        # Layer 2 -> Layer 1
        self.up2_1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.decode1 = nn.Conv2d(64 + 64, 64, kernel_size=3, padding=1)
        
        # Final upsampling to original resolution
        self.final_up = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, kernel_size=1)
        )
        
    def forward(self, features):
        """Process features through decoder path with skip connections"""
        # Extract and reduce features from backbone
        f1 = self.act(self.reduce1(features['layer1']))  # 1/4 resolution
        f2 = self.act(self.reduce2(features['layer2']))  # 1/8 resolution
        f3 = self.act(self.reduce3(features['layer3']))  # 1/16 resolution
        f4 = self.act(self.reduce4(features['layer4']))  # 1/32 resolution
        
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
        
    def _debug_print_shapes(self, features):
        """Helper method to print feature shapes for debugging"""
        print(f"layer1: {features['layer1'].shape}")
        print(f"layer2: {features['layer2'].shape}")
        print(f"layer3: {features['layer3'].shape}")
        print(f"layer4: {features['layer4'].shape}")
        
        f1 = self.reduce1(features['layer1'])
        f2 = self.reduce2(features['layer2'])
        f3 = self.reduce3(features['layer3'])
        f4 = self.reduce4(features['layer4'])
        
        print(f"f1: {f1.shape}")
        print(f"f2: {f2.shape}")
        print(f"f3: {f3.shape}")
        print(f"f4: {f4.shape}")

class ViTExtractor(nn.Module):
    """Extracts multi-scale features from a ViT model"""
    def __init__(self, vit, img_size=224, patch_size=16):
        super().__init__()
        self.vit = vit
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
    def forward(self, x):
        features = {}
        
        # Get patch embeddings and positional encodings from ViT
        if hasattr(self.vit, 'patch_embed'):
            # For timm ViT models
            x = self.vit.patch_embed(x)
            cls_token = self.vit.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)
            x = self.vit.pos_drop(x + self.vit.pos_embed)
            
            # Process through transformer blocks and collect intermediate outputs
            # This is a simplification and may need adjustment based on ViT implementation
            for i, block in enumerate(self.vit.blocks):
                x = block(x)
                if i in [3, 7, 11]:  # Extract features at different depths (adjust indices as needed)
                    patch_tokens = x[:, 1:]  # Exclude CLS token
                    h = w = int(self.num_patches ** 0.5)
                    # Reshape to spatial feature map [B, h, w, C]
                    spatial_tokens = patch_tokens.reshape(
                        patch_tokens.shape[0], h, w, patch_tokens.shape[-1]
                    )
                    # Convert to [B, C, h, w] format
                    spatial_tokens = spatial_tokens.permute(0, 3, 1, 2)
                    layer_idx = (i+1) // 4  # Map blocks to layer names (simplified)
                    features[f'layer{layer_idx}'] = spatial_tokens
            
            # Final layer features
            patch_tokens = x[:, 1:]  # Exclude CLS token
            h = w = int(self.num_patches ** 0.5)
            spatial_tokens = patch_tokens.reshape(
                patch_tokens.shape[0], h, w, patch_tokens.shape[-1]
            )
            spatial_tokens = spatial_tokens.permute(0, 3, 1, 2)
            features['layer4'] = spatial_tokens  # Final layer
        
        return features

class SwinExtractor(nn.Module):
    """Extracts multi-scale features from a Swin Transformer model"""
    def __init__(self, swin):
        super().__init__()
        self.swin = swin
        
    def forward(self, x):
        features = {}
        
        # For Swin, we need to adapt based on its implementation
        # This is a simplification and needs to be adjusted based on the exact Swin implementation
        x = self.swin.patch_embed(x)
        
        if hasattr(self.swin, 'absolute_pos_embed') and self.swin.absolute_pos_embed is not None:
            x = x + self.swin.absolute_pos_embed
        
        x = self.swin.pos_drop(x)
        
        # Process through Swin stages and collect outputs
        # Stage 1
        x = self.swin.layers[0](x)
        features['layer1'] = x.permute(0, 3, 1, 2)  # Convert to [B, C, H, W]
        
        # Stage 2
        x = self.swin.layers[1](x)
        features['layer2'] = x.permute(0, 3, 1, 2)
        
        # Stage 3
        x = self.swin.layers[2](x)
        features['layer3'] = x.permute(0, 3, 1, 2)
        
        # Stage 4
        x = self.swin.layers[3](x)
        features['layer4'] = x.permute(0, 3, 1, 2)
        
        return features

class SegmentationModel(pl.LightningModule):
    """
    Complete model combining pretrained backbone with segmentation head
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
        class_weights = [1.0,1.0]
    ):
        super().__init__()
        
        self.backbone_type = backbone_type
        self.img_size = img_size
    
        # Initialize the appropriate backbone
        if backbone_type.startswith("resnet"):
            if backbone_type == "resnet18":
                resnet = torchvision.models.resnet18(weights=None)
                self._modify_resnet_for_4_channels(resnet, in_channels)
                self.feature_extractor = ResNetExtractor(resnet)
            elif backbone_type == "resnet50":
                resnet = torchvision.models.resnet50(weights=None)
                self._modify_resnet_for_4_channels(resnet, in_channels)
                self.feature_extractor = ResNetExtractor(resnet)
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
                self.feature_extractor = ViTExtractor(vit, img_size=img_size)
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
                self.feature_extractor = SwinExtractor(swin)
            else:
                raise ValueError(f"Unsupported Swin type: {backbone_type}")
        else:
            raise ValueError(f"Unsupported backbone type: {backbone_type}")
        
        # Load pretrained weights if provided
        if pretrained_path:
            self._load_pretrained_weights(pretrained_path)
        
        # Freeze backbone weights if using pretrained model
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
        
        # Initialize segmentation head with correct dimensions
        self.seg_head = SegmentationHead(
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
        self.class_weights = torch.tensor(class_weights)
        
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
        
    def forward(self, x):
        """Forward pass through backbone and segmentation head"""
        features = self.feature_extractor(x)
        segmentation_map = self.seg_head(features)
        return segmentation_map
    
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
    
    def training_step(self, batch, batch_idx):
        """Training step with mixed dice, weighted cross-entropy loss"""
        images, masks = batch
        logits = self(images)

        # Dice loss for segmentation
        loss = self.combined_loss(logits, masks)
        
        # Log metrics
        self.log("train_loss", loss, on_step = True, on_epoch =True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step with metrics calculation"""
        images, masks = batch
        logits = self(images)
        
        # Cross entropy loss
        loss = F.cross_entropy(logits, masks)
        
        # Calculate pixel accuracy
        preds = torch.argmax(logits, dim=1)
        accuracy = (preds == masks).float().mean()
        
        # Log metrics
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_accuracy", accuracy, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):
        """Configure optimizer with learning rate scheduler"""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5, 
            patience=5,

        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train_loss",
                "interval": "epoch"
            }
        }
    
    def unfreeze_backbone(self, fine_tune_layers=None):
        """
        Unfreeze backbone layers for fine-tuning
        Args:
            fine_tune_layers: List of layer names to unfreeze, or None to unfreeze all
        """
        if fine_tune_layers is None:
            # Unfreeze all backbone parameters
            for param in self.feature_extractor.parameters():
                param.requires_grad = True
        else:
            # Unfreeze only specific layers
            for layer_name in fine_tune_layers:
                if hasattr(self.feature_extractor.backbone, layer_name):
                    for param in getattr(self.feature_extractor.backbone, layer_name).parameters():
                        param.requires_grad = True

    def test_step(self, batch, batch_idx):
        """
        Run test step on a single batch
        """
        images, masks = batch
        
        # Forward pass
        logits = self(images)
        
        # Cross entropy loss
        loss = F.cross_entropy(logits, masks)
        
        # Get predictions
        preds = torch.argmax(logits, dim=1)
        
        # Calculate accuracy
        accuracy = (preds == masks).float().mean()
        
        # Initialize metrics per class
        num_classes = logits.shape[1]
        metrics_per_class = []
        confusion_matrix = torch.zeros(num_classes, num_classes, dtype=torch.long, device=self.device)
        
        # Calculate metrics for each class and build confusion matrix
        for i in range(num_classes):
            for j in range(num_classes):
                confusion_matrix[i, j] = ((preds == j) & (masks == i)).sum()
        
        for c in range(num_classes):
            # True positives, false positives, false negatives
            true_pos = ((preds == c) & (masks == c)).sum().float()
            false_pos = ((preds == c) & (masks != c)).sum().float()
            false_neg = ((preds != c) & (masks == c)).sum().float()
            
            # Calculate metrics, avoiding division by zero
            precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else torch.tensor(0.0, device=self.device)
            recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else torch.tensor(0.0, device=self.device)
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else torch.tensor(0.0, device=self.device)
            
            # Calculate IoU (Intersection over Union) for this class
            iou = true_pos / (true_pos + false_pos + false_neg) if (true_pos + false_pos + false_neg) > 0 else torch.tensor(0.0, device=self.device)
            
            metrics_per_class.append({
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'iou': iou
            })
        
        # Calculate overall metrics (micro average across all classes)
        total_true_pos = sum(confusion_matrix[i, i] for i in range(num_classes)).float()
        total_elements = confusion_matrix.sum().float()
        total_precision = total_true_pos / total_elements if total_elements > 0 else torch.tensor(0.0, device=self.device)
        total_f1 = 2 * total_precision * total_precision / (total_precision + total_precision) if total_precision > 0 else torch.tensor(0.0, device=self.device)
        
        # Return metrics to be accumulated
        return {
            'test_loss': loss,
            'test_accuracy': accuracy,
            'confusion_matrix': confusion_matrix,
            'metrics_per_class': metrics_per_class,
            'total_f1': total_f1
        }
    
    def test_epoch_end(self, outputs):
        """
        Aggregate metrics from all test batches
        """
        num_classes = len(outputs[0]['metrics_per_class'])
        class_names = ["Background", "Vines"]  # Adjust based on your classes
        
        # Aggregate test loss and accuracy
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        avg_accuracy = torch.stack([x['test_accuracy'] for x in outputs]).mean()
        
        # Aggregate confusion matrix
        total_confusion_matrix = torch.zeros(num_classes, num_classes, dtype=torch.long, device=self.device)
        for output in outputs:
            total_confusion_matrix += output['confusion_matrix']
        
        # Aggregate metrics per class
        avg_metrics_per_class = []
        for c in range(num_classes):
            avg_precision = torch.stack([x['metrics_per_class'][c]['precision'] for x in outputs]).mean()
            avg_recall = torch.stack([x['metrics_per_class'][c]['recall'] for x in outputs]).mean()
            avg_f1 = torch.stack([x['metrics_per_class'][c]['f1'] for x in outputs]).mean()
            avg_iou = torch.stack([x['metrics_per_class'][c]['iou'] for x in outputs]).mean()
            
            avg_metrics_per_class.append({
                'precision': avg_precision,
                'recall': avg_recall,
                'f1': avg_f1,
                'iou': avg_iou
            })
        
        # Calculate mean IoU across all classes
        mean_iou = torch.stack([metrics['iou'] for metrics in avg_metrics_per_class]).mean()
        
        # Calculate total F1 score (micro-average)
        total_f1 = torch.stack([x['total_f1'] for x in outputs]).mean()
        
        # Log metrics
        self.log('test_loss', avg_loss)
        self.log('test_accuracy', avg_accuracy)
        self.log('test_mean_iou', mean_iou)
        self.log('test_total_f1', total_f1)
        
        for c in range(num_classes):
            class_name = class_names[c] if c < len(class_names) else f"class_{c}"
            self.log(f'test_{class_name}_precision', avg_metrics_per_class[c]['precision'])
            self.log(f'test_{class_name}_recall', avg_metrics_per_class[c]['recall'])
            self.log(f'test_{class_name}_f1', avg_metrics_per_class[c]['f1'])
            self.log(f'test_{class_name}_iou', avg_metrics_per_class[c]['iou'])
        
        # Print results
        print("\n===== Test Set Evaluation =====")
        print(f"Loss: {avg_loss:.4f}")
        print(f"Accuracy: {avg_accuracy:.4f}")
        print(f"Total F1 Score (micro-avg): {total_f1:.4f}")
        print(f"Mean IoU: {mean_iou:.4f}")
        
        print("\nPer-class metrics:")
        for i in range(num_classes):
            class_name = class_names[i] if i < len(class_names) else f"Class {i}"
            print(f"{class_name}:")
            print(f"  Precision: {avg_metrics_per_class[i]['precision']:.4f}")
            print(f"  Recall: {avg_metrics_per_class[i]['recall']:.4f}")
            print(f"  F1 Score: {avg_metrics_per_class[i]['f1']:.4f}")
            print(f"  IoU: {avg_metrics_per_class[i]['iou']:.4f}")
        
        print("\nConfusion Matrix:")
        print(total_confusion_matrix)
        
        # Return the metrics dictionary
        return {
            "test_loss": avg_loss,
            "test_accuracy": avg_accuracy,
            "test_total_f1": total_f1,
            "test_mean_iou": mean_iou,
            "test_confusion_matrix": total_confusion_matrix,
            "test_metrics_per_class": avg_metrics_per_class
        }
    
    def test(self, test_loader=None):
        """
        Comprehensive evaluation of segmentation model on test set
        This method can be called manually and provides a more organized output
        compared to trainer.test()
        
        Args:
            test_loader: Optional DataLoader for test data
        
        Returns:
            Dictionary with evaluation metrics
        """
        if test_loader is None:
            raise ValueError("test_loader must be provided")
        
        self.eval()
        torch.set_grad_enabled(False)
        
        # Run test batches through test_step
        outputs = []
        for batch_idx, batch in enumerate(test_loader):
            batch = [x.to(self.device) for x in batch]
            output = self.test_step(batch, batch_idx)
            outputs.append(output)
        
        # Process results
        results = self.test_epoch_end(outputs)
        torch.set_grad_enabled(True)
        
        return results
    