import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Dict, List, Optional, Tuple, Union
from einops import rearrange

# Import Pasiphae backbone
from .msrgb_vit import MSRGBViT


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


class UPerHead(nn.Module):
    """
    Unified Perceptual Parsing Head (UPerNet)
    
    Combines feature maps from different stages of the backbone
    using Feature Pyramid Network (FPN) and Pyramid Pooling Module (PPM).
    """
    def __init__(
        self,
        in_channels: List[int],
        num_classes: int,
        fpn_dim: int = 256,
        ppm_bins: Tuple[int] = (1, 2, 3, 6),
    ):
        super(UPerHead, self).__init__()
        # For single-scale ViT, we treat the output as a single-level feature
        self.in_channels = in_channels
        
        # PPM module on the last feature map
        self.ppm = PPM(in_channels[-1], fpn_dim // len(ppm_bins), ppm_bins)
        ppm_out_dim = in_channels[-1] + fpn_dim
        
        # FPN lateral connections
        self.fpn_in = nn.ModuleList()
        for in_c in in_channels[:-1]:  # Skip the last one as it's handled by PPM
            self.fpn_in.append(nn.Sequential(
                nn.Conv2d(in_c, fpn_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(fpn_dim),
                nn.ReLU(inplace=True)
            ))
            
        # FPN output connections
        self.fpn_out = nn.ModuleList()
        for _ in range(len(in_channels) - 1):  # -1 because the last one is handled by PPM
            self.fpn_out.append(nn.Sequential(
                nn.Conv2d(fpn_dim, fpn_dim, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(fpn_dim),
                nn.ReLU(inplace=True)
            ))
            
        # Handle the last feature map with PPM
        self.fpn_bottleneck = nn.Sequential(
            nn.Conv2d(ppm_out_dim, fpn_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fpn_dim),
            nn.ReLU(inplace=True)
        )
        
        # Final classifier
        self.classifier = nn.Conv2d(fpn_dim, num_classes, kernel_size=1)
        
    def forward(self, features):
        """
        Args:
            features: List of feature maps from the backbone
                      [feat1, feat2, ..., featN] from lowest to highest resolution
                      
        Returns:
            logits: Segmentation logits at 1/4 of input resolution
        """
        # Apply PPM to the last feature map
        f = features[-1]
        fpn_features = [self.fpn_bottleneck(self.ppm(f))]
        
        # Build FPN from top to bottom
        for i in reversed(range(len(self.in_channels) - 1)):
            feat = features[i]
            lateral = self.fpn_in[i](feat)
            fpn_feat = F.interpolate(fpn_features[0], size=lateral.shape[2:],
                                    mode='bilinear', align_corners=True)
            fpn_feat = lateral + fpn_feat
            fpn_feat = self.fpn_out[i](fpn_feat)
            fpn_features.insert(0, fpn_feat)
            
        # Use the highest resolution feature (first in the list)
        output = self.classifier(fpn_features[0])
        
        return output


class PasiphaeFeatureExtractor(nn.Module):
    """
    Adapts Pasiphae (MSRGBViT) to output features at different scales
    for semantic segmentation with UPerNet.
    """
    def __init__(self, backbone, feature_dims=None):
        super(PasiphaeFeatureExtractor, self).__init__()
        self.backbone = backbone
        self.feature_dims = feature_dims if feature_dims is not None else [576, 576, 576, 576]
        
        # Define indices for intermediate features
        # For a ViT-based model, we can use specific transformer blocks
        # Assuming a 12-layer ViT (0-indexed), we might use:
        self.feature_indices = [0, 2, 5, 8, 11]
        
    def forward_features(self, ms=None, rgb=None):
        # Get tokens from tokenizer
        ms_tokens, rgb_tokens, mask = self.backbone.tokenizer(ms=ms, rgb=rgb)
        
        # Prepare tokens for the transformer
        tokens, attn_mask = self.backbone._prepare_tokens(ms_tokens, rgb_tokens, mask)
        
        # Get spatial dimensions
        if ms_tokens is not None:
            H, W = ms_tokens.shape[1:3]
            C_ms = ms_tokens.shape[3]
        else:
            H, W = rgb_tokens.shape[1:3]
            C_ms = 0
            
        if rgb_tokens is not None:
            C_rgb = rgb_tokens.shape[3]
        else:
            C_rgb = 0
        
        features = []
        
        # Pass through transformer blocks and save intermediate features
        for i, block in enumerate(self.backbone.blocks):
            tokens = block(tokens, attn_mask=attn_mask)
            
            if i in self.feature_indices:
                # Reshape tokens back to spatial form for convolutional operations
                # The token arrangement depends on the model structure
                if ms_tokens is not None and rgb_tokens is not None:
                    # Split tokens back to respective modalities
                    ms_len = H * W * C_ms
                    curr_ms_tokens = tokens[:, :ms_len, :]
                    curr_rgb_tokens = tokens[:, ms_len:, :]
                    
                    # Reshape back to spatial form - need to flatten the embedding dimension with the channel dimension
                    # First reshape to [B, H, W, C, D]
                    curr_ms_tokens = curr_ms_tokens.reshape(curr_ms_tokens.shape[0], H, W, C_ms, curr_ms_tokens.shape[-1])
                    curr_rgb_tokens = curr_rgb_tokens.reshape(curr_rgb_tokens.shape[0], H, W, C_rgb, curr_rgb_tokens.shape[-1])
                    
                    # Then permute to [B, D, H, W, C]
                    curr_ms_tokens = curr_ms_tokens.permute(0, 4, 1, 2, 3)
                    curr_rgb_tokens = curr_rgb_tokens.permute(0, 4, 1, 2, 3)
                    
                    # Finally reshape to [B, D*C, H, W]
                    curr_ms_tokens = curr_ms_tokens.reshape(curr_ms_tokens.shape[0], 
                                                           curr_ms_tokens.shape[1] * C_ms, 
                                                           H, W)
                    curr_rgb_tokens = curr_rgb_tokens.reshape(curr_rgb_tokens.shape[0], 
                                                             curr_rgb_tokens.shape[1] * C_rgb, 
                                                             H, W)
                    
                    # Concatenate along channel dimension for UPerNet
                    combined = torch.cat([curr_ms_tokens, curr_rgb_tokens], dim=1)
                    features.append(combined)
                    
                elif ms_tokens is not None:
                    # Only MS tokens
                    # Use reshape+permute instead of rearrange
                    curr_tokens = tokens.reshape(tokens.shape[0], H, W, C_ms, tokens.shape[-1])
                    curr_tokens = curr_tokens.permute(0, 4, 1, 2, 3)
                    curr_tokens = curr_tokens.reshape(curr_tokens.shape[0], 
                                                     curr_tokens.shape[1] * C_ms, 
                                                     H, W)
                    features.append(curr_tokens)
                    
                else:
                    # Only RGB tokens
                    # Use reshape+permute instead of rearrange
                    curr_tokens = tokens.reshape(tokens.shape[0], H, W, C_rgb, tokens.shape[-1])
                    curr_tokens = curr_tokens.permute(0, 4, 1, 2, 3)
                    curr_tokens = curr_tokens.reshape(curr_tokens.shape[0], 
                                                     curr_tokens.shape[1] * C_rgb, 
                                                     H, W)
                    features.append(curr_tokens)
        
        # Apply final normalization to last tokens
        tokens = self.backbone.norm(tokens)
        
        # For the final output, also reshape to spatial form
        if ms_tokens is not None and rgb_tokens is not None:
            # Split tokens back to respective modalities
            ms_len = H * W * C_ms
            curr_ms_tokens = tokens[:, :ms_len, :]
            curr_rgb_tokens = tokens[:, ms_len:, :]
            
            # Reshape back to spatial form - using the same approach as above
            curr_ms_tokens = curr_ms_tokens.reshape(curr_ms_tokens.shape[0], H, W, C_ms, curr_ms_tokens.shape[-1])
            curr_rgb_tokens = curr_rgb_tokens.reshape(curr_rgb_tokens.shape[0], H, W, C_rgb, curr_rgb_tokens.shape[-1])
            
            curr_ms_tokens = curr_ms_tokens.permute(0, 4, 1, 2, 3)
            curr_rgb_tokens = curr_rgb_tokens.permute(0, 4, 1, 2, 3)
            
            curr_ms_tokens = curr_ms_tokens.reshape(curr_ms_tokens.shape[0], 
                                                   curr_ms_tokens.shape[1] * C_ms, 
                                                   H, W)
            curr_rgb_tokens = curr_rgb_tokens.reshape(curr_rgb_tokens.shape[0], 
                                                     curr_rgb_tokens.shape[1] * C_rgb, 
                                                     H, W)
            
            # Concatenate along channel dimension for UPerNet
            combined = torch.cat([curr_ms_tokens, curr_rgb_tokens], dim=1)
            features.append(combined)
            
        elif ms_tokens is not None:
            # Only MS tokens
            curr_tokens = tokens.reshape(tokens.shape[0], H, W, C_ms, tokens.shape[-1])
            curr_tokens = curr_tokens.permute(0, 4, 1, 2, 3)
            curr_tokens = curr_tokens.reshape(curr_tokens.shape[0], 
                                             curr_tokens.shape[1] * C_ms, 
                                             H, W)
            features.append(curr_tokens)
            
        else:
            # Only RGB tokens
            curr_tokens = tokens.reshape(tokens.shape[0], H, W, C_rgb, tokens.shape[-1])
            curr_tokens = curr_tokens.permute(0, 4, 1, 2, 3)
            curr_tokens = curr_tokens.reshape(curr_tokens.shape[0], 
                                             curr_tokens.shape[1] * C_rgb, 
                                             H, W)
            features.append(curr_tokens)
            
        return features
        
    def forward(self, ms=None, rgb=None):
        features = self.forward_features(ms=ms, rgb=rgb)
        return features


class PasiphaeUPerNet(nn.Module):
    """
    Combined Pasiphae backbone with UPerNet head for semantic segmentation
    """
    def __init__(
        self,
        num_classes: int,
        backbone_cfg: Dict = None,
        checkpoint_path: Optional[str] = None,
        freeze_backbone: bool = False,
    ):
        super(PasiphaeUPerNet, self).__init__()
        
        # Default backbone configuration if not provided
        if backbone_cfg is None:
            backbone_cfg = {
                'img_size': 224,
                'patch_size': 16,
                'embed_dim': 192,
                'depth': 12,
                'num_heads': 3,
                'mlp_ratio': 4.0,
                'qkv_bias': True,
                'use_channel_embeds': True,
            }
        
        # Initialize Pasiphae backbone
        self.backbone_model = MSRGBViT(**backbone_cfg)
        
        # Load checkpoint if provided
        if checkpoint_path is not None:
            self._load_from_checkpoint(checkpoint_path)
        
        # Determine feature dimensions based on configuration
        # For Pasiphae with RGB inputs, feature dimensions depend on embed_dim and number of RGB channels
        embed_dim = backbone_cfg['embed_dim']
        
        # For RGB images with 3 channels, each feature map has embed_dim * 3 channels
        feature_dim = embed_dim * 3  # For RGB (3 channels)
        
        # Feature dimensions at different levels
        feature_dims = [feature_dim] * 4
        
        # Create feature extractor wrapper
        self.backbone = PasiphaeFeatureExtractor(self.backbone_model, feature_dims)
        
        # Get actual feature dimensions from a forward pass with a dummy input
        with torch.no_grad():
            dummy_rgb = torch.zeros(1, 3, 224, 224)
            dummy_features = self.backbone(rgb=dummy_rgb)
            actual_feature_dims = [f.shape[1] for f in dummy_features]
            print(f"Actual feature dimensions: {actual_feature_dims}")
        
        # Create UPerNet head with actual feature dimensions
        self.decode_head = UPerHead(
            in_channels=actual_feature_dims,
            num_classes=num_classes,
            fpn_dim=256,
        )
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # For upsampling the final output to original image size
        self.upsample = lambda x, size: F.interpolate(
            x, size=size, mode='bilinear', align_corners=True
        )
        
    def _load_from_checkpoint(self, checkpoint_path):
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
    
    def forward(self, ms=None, rgb=None, original_size=None):
        """
        Forward pass
        
        Args:
            ms: MS input tensor of shape [B, 5, H, W]
            rgb: RGB input tensor of shape [B, 3, H, W]
            original_size: Original image size for upsampling
            
        Returns:
            logits: Segmentation logits at original input resolution
        """
        # Remember input size if not provided
        if original_size is None:
            if ms is not None:
                original_size = ms.shape[2:]  # (H, W)
            else:
                original_size = rgb.shape[2:]  # (H, W)
        
        # Extract features from backbone
        features = self.backbone(ms=ms, rgb=rgb)
        
        # Apply segmentation head
        logits = self.decode_head(features)
        
        # Upsample to original image size
        if original_size != logits.shape[2:]:
            logits = self.upsample(logits, original_size)
            
        return logits


class PasiphaeUPerNetModule(pl.LightningModule):
    """
    PyTorch Lightning module for PasiphaeUPerNet
    """
    def __init__(
        self,
        num_classes: int,
        backbone_cfg: Dict = None,
        checkpoint_path: Optional[str] = None,
        freeze_backbone: bool = False,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
    ):
        super(PasiphaeUPerNetModule, self).__init__()
        
        # Create the model
        self.model = PasiphaeUPerNet(
            num_classes=num_classes,
            backbone_cfg=backbone_cfg,
            checkpoint_path=checkpoint_path,
            freeze_backbone=freeze_backbone,
        )
        
        # Training parameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, ms=None, rgb=None):
        return self.model(ms=ms, rgb=rgb)
        
    def training_step(self, batch, batch_idx):
        # Extract inputs and target
        ms = batch.get('ms')
        rgb = batch.get('rgb')
        target = batch['mask']
        
        # Forward pass
        logits = self(ms=ms, rgb=rgb)
        
        # Calculate loss
        loss = self.criterion(logits, target)
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
        
    def validation_step(self, batch, batch_idx):
        # Extract inputs and target
        ms = batch.get('ms')
        rgb = batch.get('rgb')
        target = batch['mask']
        
        # Forward pass
        logits = self(ms=ms, rgb=rgb)
        
        # Calculate loss
        loss = self.criterion(logits, target)
        
        # Log metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        
        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        accuracy = (preds == target).float().mean()
        self.log('val_accuracy', accuracy, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }