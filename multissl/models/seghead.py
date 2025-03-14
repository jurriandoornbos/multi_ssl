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

class ResNetBackbone(nn.Module):
    """Enhanced ResNet feature extractor with dimension tracking and adaptive pooling"""
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
        
        # Add adaptive pooling for consistent output dimensions
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten(1)
        
        # Store feature dimensions for reference
        self.feature_dims = {
            'stem': self.conv1.out_channels,  # 64 for ResNet18
            'layer1': self._get_layer_out_channels(self.layer1),  # 64 for ResNet18
            'layer2': self._get_layer_out_channels(self.layer2),  # 128 for ResNet18
            'layer3': self._get_layer_out_channels(self.layer3),  # 256 for ResNet18
            'layer4': self._get_layer_out_channels(self.layer4),  # 512 for ResNet18
        }
    
    def _get_layer_out_channels(self, layer):
        """Get output channels for a layer by checking its last block"""
        if hasattr(layer[-1], 'conv3'):  # For bottleneck blocks (ResNet50+)
            return layer[-1].conv3.out_channels
        else:  # For basic blocks (ResNet18/34)
            return layer[-1].conv2.out_channels
    
    def forward(self, x):
        features = {}
        
        # Stem features
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        features['stem'] = x
        
        x = self.maxpool(x)
        
        # Layer features
        x = self.layer1(x)
        features['layer1'] = x
        
        x = self.layer2(x)
        features['layer2'] = x
        
        x = self.layer3(x)
        features['layer3'] = x
        
        x = self.layer4(x)
        features['layer4'] = x
        
        # Global features
        features['pooled'] = self.avgpool(x)
        features['flat'] = self.flatten(features['pooled'])
        
        return features
    
    def get_block_features(self, x):
        """Extract features from individual blocks within layers"""
        block_features = {}
        
        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        block_features['stem'] = x
        
        x = self.maxpool(x)
        
        # Process blocks individually
        for i, block in enumerate(self.layer1):
            x = block(x)
            block_features[f'layer1.{i}'] = x
            
        for i, block in enumerate(self.layer2):
            x = block(x)
            block_features[f'layer2.{i}'] = x
            
        for i, block in enumerate(self.layer3):
            x = block(x)
            block_features[f'layer3.{i}'] = x
            
        for i, block in enumerate(self.layer4):
            x = block(x)
            block_features[f'layer4.{i}'] = x
        
        return block_features

class ResNetBackboneUNet(nn.Module):
    """
    UNet architecture with ResNet encoder.
    Combines high-level semantic features with low-level spatial features
    through skip connections between encoder and decoder.
    """
    def __init__(self, feat_dims, num_classes=2, img_size=224):
        super().__init__()
        

        self.img_size = img_size
        # Global context module - uses pooled features to create context vectors
        # This helps the model understand global image context
        self.global_context = nn.Sequential(
            nn.Linear(feat_dims['layer4'], 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True)
        )
        
        # Create decoder path with skip connections
        
        # Decoder Block 4 (starting from deepest layer)
        # Takes layer4 features and global context, upsamples to layer3's dimensions
        self.decoder4 = nn.Sequential(
            nn.ConvTranspose2d(feat_dims['layer4'], feat_dims['layer3'], kernel_size=2, stride=2),
            nn.BatchNorm2d(feat_dims['layer3']),
            nn.ReLU(inplace=True)
        )
        # Additional convolution to integrate global context with layer4 features
        self.global_inject4 = nn.Conv2d(feat_dims['layer4'] + 256, feat_dims['layer4'], kernel_size=1)
        self.up_conv4 = nn.Sequential(
            nn.Conv2d(feat_dims['layer3'] * 2, feat_dims['layer3'], kernel_size=3, padding=1),
            nn.BatchNorm2d(feat_dims['layer3']),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_dims['layer3'], feat_dims['layer3'], kernel_size=3, padding=1),
            nn.BatchNorm2d(feat_dims['layer3']),
            nn.ReLU(inplace=True)
        )
        
        # Decoder Block 3
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(feat_dims['layer3'], feat_dims['layer2'], kernel_size=2, stride=2),
            nn.BatchNorm2d(feat_dims['layer2']),
            nn.ReLU(inplace=True)
        )
        self.up_conv3 = nn.Sequential(
            nn.Conv2d(feat_dims['layer2'] * 2, feat_dims['layer2'], kernel_size=3, padding=1),
            nn.BatchNorm2d(feat_dims['layer2']),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_dims['layer2'], feat_dims['layer2'], kernel_size=3, padding=1),
            nn.BatchNorm2d(feat_dims['layer2']),
            nn.ReLU(inplace=True)
        )
        
        # Decoder Block 2
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(feat_dims['layer2'], feat_dims['layer1'], kernel_size=2, stride=2),
            nn.BatchNorm2d(feat_dims['layer1']),
            nn.ReLU(inplace=True)
        )
        self.up_conv2 = nn.Sequential(
            nn.Conv2d(feat_dims['layer1'] * 2, feat_dims['layer1'], kernel_size=3, padding=1),
            nn.BatchNorm2d(feat_dims['layer1']),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_dims['layer1'], feat_dims['layer1'], kernel_size=3, padding=1),
            nn.BatchNorm2d(feat_dims['layer1']),
            nn.ReLU(inplace=True)
        )
        
        # Decoder Block 1
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(feat_dims['layer1'], feat_dims['stem'], kernel_size=2, stride=2),
            nn.BatchNorm2d(feat_dims['stem']),
            nn.ReLU(inplace=True)
        )
        self.up_conv1 = nn.Sequential(
            nn.Conv2d(feat_dims['stem'] * 2, feat_dims['stem'], kernel_size=3, padding=1),
            nn.BatchNorm2d(feat_dims['stem']),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_dims['stem'], feat_dims['stem'], kernel_size=3, padding=1),
            nn.BatchNorm2d(feat_dims['stem']),
            nn.ReLU(inplace=True)
        )
        
        # Final upsampling to original image size
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(feat_dims['stem'], 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # Final convolution to produce class logits
        self.final_conv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, kernel_size=1)
        )
    
    def forward(self, features):
        
        # Extract features from different layers
        stem_features = features['stem']       # 1/2 spatial size
        layer1_features = features['layer1']   # 1/4 spatial size
        layer2_features = features['layer2']   # 1/8 spatial size
        layer3_features = features['layer3']   # 1/16 spatial size
        layer4_features = features['layer4']   # 1/32 spatial size
        
        # Get global context from pooled features
        pooled_features = features['flat']     # Global features
        global_context = self.global_context(pooled_features)
        
        # Reshape global context to be injected into spatial features
        # Convert from [B, C] to [B, C, 1, 1] for broadcasting
        global_context = global_context.unsqueeze(-1).unsqueeze(-1)
        
        # Broadcast and tile global context to match spatial dimensions of layer4
        h, w = layer4_features.shape[2:]
        global_context_expanded = global_context.expand(-1, -1, h, w)
        
        # Combine global context with deepest feature map (layer4)
        layer4_with_context = torch.cat([layer4_features, global_context_expanded], dim=1)
        layer4_with_context = self.global_inject4(layer4_with_context)
        
        # Decoder path with skip connections
        
        # Decoder Block 4: layer4 -> layer3 (1/32 -> 1/16)
        d4 = self.decoder4(layer4_with_context)
        # Ensure dimensions match exactly before concatenation
        if d4.shape != layer3_features.shape:
            d4 = F.interpolate(d4, size=layer3_features.shape[2:], mode='bilinear', align_corners=False)
        d4 = torch.cat([d4, layer3_features], dim=1)
        d4 = self.up_conv4(d4)
        
        # Decoder Block 3: d4 -> layer2 (1/16 -> 1/8)
        d3 = self.decoder3(d4)
        if d3.shape != layer2_features.shape:
            d3 = F.interpolate(d3, size=layer2_features.shape[2:], mode='bilinear', align_corners=False)
        d3 = torch.cat([d3, layer2_features], dim=1)
        d3 = self.up_conv3(d3)
        
        # Decoder Block 2: d3 -> layer1 (1/8 -> 1/4)
        d2 = self.decoder2(d3)
        if d2.shape != layer1_features.shape:
            d2 = F.interpolate(d2, size=layer1_features.shape[2:], mode='bilinear', align_corners=False)
        d2 = torch.cat([d2, layer1_features], dim=1)
        d2 = self.up_conv2(d2)
        
        # Decoder Block 1: d2 -> stem (1/4 -> 1/2)
        d1 = self.decoder1(d2)
        if d1.shape != stem_features.shape:
            d1 = F.interpolate(d1, size=stem_features.shape[2:], mode='bilinear', align_corners=False)
        d1 = torch.cat([d1, stem_features], dim=1)
        d1 = self.up_conv1(d1)
        
        # Final upsampling to original image size (1/2 -> 1/1)
        out = self.final_up(d1)
        
        # Handle any size discrepancies for the final output
        if out.shape[2:] != (self.img_size, self.img_size):
            out = F.interpolate(out, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)
        
        # Final convolution to produce class logits
        out = self.final_conv(out)
        
        return out

class HighResolutionHead(nn.Module):
    """
    High-resolution segmentation head inspired by HRNetV2.
    Maintains high resolution throughout and uses multi-scale fusion for precise boundaries.
    """
    def __init__(self, layer1_channels, layer2_channels,layer3_channels,layer4_channels,num_classes, img_size, high_res_channels=128):
        super().__init__()
        layer_channels = {"layer1": layer1_channels,
                    "layer2": layer2_channels,
                    "layer3": layer3_channels,
                    "layer4": layer4_channels}
        self.img_size = img_size
        
        # 1. Initial reduction of feature dimensions from backbone
        self.reduce_layer1 = nn.Sequential(
            nn.Conv2d(layer_channels['layer1'], high_res_channels, kernel_size=1),
            nn.GroupNorm(32, high_res_channels),
            nn.ReLU(inplace=True)
        )
        self.reduce_layer2 = nn.Sequential(
            nn.Conv2d(layer_channels['layer2'], high_res_channels, kernel_size=1),
            nn.GroupNorm(32, high_res_channels),
            nn.ReLU(inplace=True)
        )
        self.reduce_layer3 = nn.Sequential(
            nn.Conv2d(layer_channels['layer3'], high_res_channels, kernel_size=1),
            nn.GroupNorm(32, high_res_channels),
            nn.ReLU(inplace=True)
        )
        self.reduce_layer4 = nn.Sequential(
            nn.Conv2d(layer_channels['layer4'], high_res_channels, kernel_size=1),
            nn.GroupNorm(32, high_res_channels),
            nn.ReLU(inplace=True)
        )
        
        # 2. High-resolution convolutions for each scale level
        self.high_res_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(high_res_channels, high_res_channels, kernel_size=3, padding=1),
                nn.GroupNorm(32, high_res_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(high_res_channels, high_res_channels, kernel_size=3, padding=1),
                nn.GroupNorm(32, high_res_channels),
                nn.ReLU(inplace=True)
            ) for _ in range(4)  # One for each resolution
        ])
        
        # 3. Multi-scale fusion module
        self.multi_scale_fusion = nn.Sequential(
            nn.Conv2d(high_res_channels * 4, high_res_channels, kernel_size=3, padding=1),
            nn.GroupNorm(32, high_res_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(high_res_channels, high_res_channels, kernel_size=3, padding=1),
            nn.GroupNorm(32, high_res_channels),
            nn.ReLU(inplace=True)
        )
        
        # 4. Final boundary refinement module
        self.boundary_refinement = nn.Sequential(
            nn.Conv2d(high_res_channels + layer_channels['layer1'], high_res_channels, kernel_size=3, padding=1),
            nn.GroupNorm(32, high_res_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(high_res_channels, high_res_channels, kernel_size=3, padding=1),
            nn.GroupNorm(32, high_res_channels),
            nn.ReLU(inplace=True)
        )
        
        # 5. Auxiliary deep supervision heads for better gradient flow
        self.aux_head = nn.Conv2d(high_res_channels, num_classes, kernel_size=1)
        
        # 6. Final classifier
        self.classifier = nn.Sequential(
            nn.Conv2d(high_res_channels, high_res_channels, kernel_size=3, padding=1),
            nn.GroupNorm(32, high_res_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(high_res_channels, num_classes, kernel_size=1)
        )
        
        # Optional: Edge attention module
        self.edge_attention = nn.Sequential(
            nn.Conv2d(high_res_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, features):
        # Extract features from backbone
        feat1 = features['layer1']  # 1/4 resolution
        feat2 = features['layer2']  # 1/8 resolution
        feat3 = features['layer3']  # 1/16 resolution
        feat4 = features['layer4']  # 1/32 resolution
        
        # Record sizes for precise upsampling
        size1 = feat1.shape[2:]  # Highest resolution feature map size
        
        # Reduce channel dimensions
        feat1 = self.reduce_layer1(feat1)
        feat2 = self.reduce_layer2(feat2)
        feat3 = self.reduce_layer3(feat3)
        feat4 = self.reduce_layer4(feat4)
        
        # Upsample all features to the highest resolution (layer1)
        feat2_up = F.interpolate(feat2, size=size1, mode='bilinear', align_corners=True)
        feat3_up = F.interpolate(feat3, size=size1, mode='bilinear', align_corners=True)
        feat4_up = F.interpolate(feat4, size=size1, mode='bilinear', align_corners=True)
        
        # Apply high-resolution convs to each feature
        feat1 = self.high_res_convs[0](feat1)
        feat2_up = self.high_res_convs[1](feat2_up)
        feat3_up = self.high_res_convs[2](feat3_up)
        feat4_up = self.high_res_convs[3](feat4_up)
        
        # Concatenate multi-scale features
        multi_scale_features = torch.cat([feat1, feat2_up, feat3_up, feat4_up], dim=1)
        
        # Apply multi-scale fusion
        fused_features = self.multi_scale_fusion(multi_scale_features)
        
        # Optional: Edge attention
        edge_map = self.edge_attention(fused_features)
        
        # Concatenate original high-res features (layer1) for boundary refinement
        concat_features = torch.cat([fused_features, features['layer1']], dim=1)
        refined_features = self.boundary_refinement(concat_features)
        
        # Apply edge-aware refinement
        refined_features = refined_features * (1 + edge_map)
        
        # Auxiliary output for deep supervision (optional in training)
        aux_output = self.aux_head(fused_features)
        aux_output = F.interpolate(aux_output, size=(self.img_size, self.img_size), 
                                 mode='bilinear', align_corners=True)
        
        # Final classification
        output = self.classifier(refined_features)
        
        # Upsample to original image size
        output = F.interpolate(output, size=(self.img_size, self.img_size), 
                              mode='bilinear', align_corners=True)
        
        # During training you can return both outputs for deep supervision
        # return output, aux_output
        return output
    
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

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwinBackbone(nn.Module):
    """
    Enhanced Swin Transformer feature extractor with proper feature mapping
    and dimension tracking for compatibility with UNet-style decoders.
    """
    def __init__(self, swin):
        super().__init__()
        self.swin = swin
        
        # Store patch size and feature dimensions
        self.patch_size = swin.patch_embed.patch_size[0] if hasattr(swin.patch_embed, 'patch_size') else 4
        
        # Detect feature dimensions from swin layers
        # Note: These need to be calculated based on the specific Swin variant
        self.feature_dims = self._detect_feature_dimensions()
        
        # Add global pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten(1)
        
    def _detect_feature_dimensions(self):
        """Determine output dimensions of each Swin stage"""
        feature_dims = {}
        
        # Get the embedding dimension from patch embed
        if hasattr(self.swin, 'patch_embed'):
            # For timm Swin models
            stem_dim = self.swin.patch_embed.proj.out_channels
            feature_dims['stem'] = stem_dim
        
        # Extract dimensions from each layer
        # For timm Swin models, dimensions are typically:
        # - layer0 (stem): C
        # - layer1: C
        # - layer2: 2C
        # - layer3: 4C
        # - layer4: 8C
        # where C is the base embedding dimension
        
        if hasattr(self.swin, 'layers') and len(self.swin.layers) >= 4:
            # For most Swin implementations with 4 stages
            feature_dims['layer1'] = self.swin.layers[0].blocks[0].mlp.fc2.out_features
            feature_dims['layer2'] = self.swin.layers[1].blocks[0].mlp.fc2.out_features
            feature_dims['layer3'] = self.swin.layers[2].blocks[0].mlp.fc2.out_features
            feature_dims['layer4'] = self.swin.layers[3].blocks[0].mlp.fc2.out_features
        
        return feature_dims
        
    def forward(self, x):
        """
        Forward pass through Swin Transformer, extracting features at each stage.
        
        Args:
            x: Input tensor of shape [B, C, H, W]
            
        Returns:
            Dictionary of features from different stages of the network
        """
        features = {}
        
        # Initial processing
        x = self.swin.patch_embed(x)
        original_shape = x.shape  # Save shape for reference
        
        # Add positional embedding if present
        if hasattr(self.swin, 'absolute_pos_embed') and self.swin.absolute_pos_embed is not None:
            x = x + self.swin.absolute_pos_embed
        
        # Apply dropout
        x = self.swin.pos_drop(x)
        
        # Store stem features (after patch embedding)
        # Need to reshape from [B, L, C] to [B, C, H, W]
        H, W = original_shape[2], original_shape[3]
        stem_features = x.permute(0, 2, 1).reshape(-1, self.feature_dims['stem'], H, W)
        features['stem'] = stem_features
        
        # Process through Swin stages
        # Stage 1
        x = self.swin.layers[0](x)
        H1, W1 = H, W  # Same resolution as stem for first stage
        features['layer1'] = x.permute(0, 2, 1).reshape(-1, self.feature_dims['layer1'], H1, W1)
        
        # Stage 2 (2x downsampling)
        x = self.swin.layers[1](x)
        H2, W2 = H1 // 2, W1 // 2  # Downsampled by 2x
        features['layer2'] = x.permute(0, 2, 1).reshape(-1, self.feature_dims['layer2'], H2, W2)
        
        # Stage 3 (4x downsampling from original)
        x = self.swin.layers[2](x)
        H3, W3 = H2 // 2, W2 // 2  # Downsampled by 2x again
        features['layer3'] = x.permute(0, 2, 1).reshape(-1, self.feature_dims['layer3'], H3, W3)
        
        # Stage 4 (8x downsampling from original)
        x = self.swin.layers[3](x)
        H4, W4 = H3 // 2, W3 // 2  # Downsampled by 2x again
        features['layer4'] = x.permute(0, 2, 1).reshape(-1, self.feature_dims['layer4'], H4, W4)
        
        # Create global features
        features['pooled'] = self.avgpool(features['layer4'])
        features['flat'] = self.flatten(features['pooled'])
        
        return features


class SwinBackboneUNet(nn.Module):
    """
    UNet architecture adapted specifically for Swin Transformer encoder.
    Incorporates attention mechanisms in the decoder to better leverage
    transformer features.
    """
    def __init__(self, feat_dims, num_classes=2, img_size=224, use_attention=False):
        super().__init__()
        self.img_size = img_size
        self.use_attention = use_attention
        
        # Global context module with attention
        self.global_context = nn.Sequential(
            nn.Linear(feat_dims['layer4'], 512),
            nn.LayerNorm(512),  # LayerNorm works better for transformer features
            nn.GELU(),  # GELU activation as used in transformers
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU()
        )
        
        # Attention modules for skip connections (if enabled)
        if use_attention:
            self.attn4 = CrossAttentionBlock(feat_dims['layer3'], feat_dims['layer4'])
            self.attn3 = CrossAttentionBlock(feat_dims['layer2'], feat_dims['layer3'])
            self.attn2 = CrossAttentionBlock(feat_dims['layer1'], feat_dims['layer2'])
            self.attn1 = CrossAttentionBlock(feat_dims['stem'], feat_dims['layer1'])
        
        # Decoder block 4: layer4 -> layer3 size
        self.decoder4 = nn.Sequential(
            nn.ConvTranspose2d(feat_dims['layer4'], feat_dims['layer3'], kernel_size=2, stride=2),
            nn.GroupNorm(32, feat_dims['layer3']),  # GroupNorm is more stable than BatchNorm
            nn.GELU()
        )
        self.global_inject4 = nn.Conv2d(feat_dims['layer4'] + 256, feat_dims['layer4'], kernel_size=1)
        self.up_conv4 = nn.Sequential(
            nn.Conv2d(feat_dims['layer3'] * 2, feat_dims['layer3'], kernel_size=3, padding=1),
            nn.GroupNorm(32, feat_dims['layer3']),
            nn.GELU(),
            nn.Conv2d(feat_dims['layer3'], feat_dims['layer3'], kernel_size=3, padding=1),
            nn.GroupNorm(32, feat_dims['layer3']),
            nn.GELU()
        )
        
        # Decoder block 3: layer3 -> layer2 size
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(feat_dims['layer3'], feat_dims['layer2'], kernel_size=2, stride=2),
            nn.GroupNorm(32, feat_dims['layer2']),
            nn.GELU()
        )
        self.up_conv3 = nn.Sequential(
            nn.Conv2d(feat_dims['layer2'] * 2, feat_dims['layer2'], kernel_size=3, padding=1),
            nn.GroupNorm(32, feat_dims['layer2']),
            nn.GELU(),
            nn.Conv2d(feat_dims['layer2'], feat_dims['layer2'], kernel_size=3, padding=1),
            nn.GroupNorm(32, feat_dims['layer2']),
            nn.GELU()
        )
        
        # Decoder block 2: layer2 -> layer1 size
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(feat_dims['layer2'], feat_dims['layer1'], kernel_size=2, stride=2),
            nn.GroupNorm(16, feat_dims['layer1']),
            nn.GELU()
        )
        self.up_conv2 = nn.Sequential(
            nn.Conv2d(feat_dims['layer1'] * 2, feat_dims['layer1'], kernel_size=3, padding=1),
            nn.GroupNorm(16, feat_dims['layer1']),
            nn.GELU(),
            nn.Conv2d(feat_dims['layer1'], feat_dims['layer1'], kernel_size=3, padding=1),
            nn.GroupNorm(16, feat_dims['layer1']),
            nn.GELU()
        )
        
        # Decoder block 1: layer1 -> stem size
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(feat_dims['layer1'], feat_dims['stem'], kernel_size=2, stride=2),
            nn.GroupNorm(16, feat_dims['stem']),
            nn.GELU()
        )
        self.up_conv1 = nn.Sequential(
            nn.Conv2d(feat_dims['stem'] * 2, feat_dims['stem'], kernel_size=3, padding=1),
            nn.GroupNorm(16, feat_dims['stem']),
            nn.GELU(),
            nn.Conv2d(feat_dims['stem'], feat_dims['stem'], kernel_size=3, padding=1),
            nn.GroupNorm(16, feat_dims['stem']),
            nn.GELU()
        )
        
        # Final upsampling to original size
        # Using pixel shuffle for better upsampling quality
        self.final_up = nn.Sequential(
            nn.Conv2d(feat_dims['stem'], 128, kernel_size=3, padding=1),
            nn.GroupNorm(16, 128),
            nn.GELU(),
            nn.PixelShuffle(2),  # Upscale by 2x using pixel shuffle (128 -> 32 channels)
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32),
            nn.GELU()
        )
        
        # Final classification layers
        self.final_conv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv2d(32, num_classes, kernel_size=1)
        )
    
    def forward(self, features):
        # Extract features from feature dictionary
        stem_features = features['stem']       # Highest resolution
        layer1_features = features['layer1']   
        layer2_features = features['layer2']   
        layer3_features = features['layer3']   
        layer4_features = features['layer4']   # Lowest resolution
        pooled_features = features['flat']     # Global features
        
        # Process global context
        global_context = self.global_context(pooled_features)
        global_context = global_context.unsqueeze(-1).unsqueeze(-1)
        
        # Expand global context to match layer4 spatial dimensions
        h4, w4 = layer4_features.shape[2:]
        global_context_expanded = global_context.expand(-1, -1, h4, w4)
        
        # Combine global context with layer4 features
        layer4_with_context = torch.cat([layer4_features, global_context_expanded], dim=1)
        layer4_with_context = self.global_inject4(layer4_with_context)
        
        # Apply attention to skip connections if enabled
        if self.use_attention:
            # Enhanced skip connections with attention
            layer3_enhanced = self.attn4(layer3_features, layer4_features)
            layer2_enhanced = self.attn3(layer2_features, layer3_features)
            layer1_enhanced = self.attn2(layer1_features, layer2_features)
            stem_enhanced = self.attn1(stem_features, layer1_features)
        else:
            # Standard skip connections
            layer3_enhanced = layer3_features
            layer2_enhanced = layer2_features
            layer1_enhanced = layer1_features
            stem_enhanced = stem_features
        
        # Decoder path with enhanced skip connections
        # Decoder Block 4: layer4 -> layer3 size
        d4 = self.decoder4(layer4_with_context)
        if d4.shape[2:] != layer3_enhanced.shape[2:]:
            d4 = F.interpolate(d4, size=layer3_enhanced.shape[2:], mode='bilinear', align_corners=False)
        d4 = torch.cat([d4, layer3_enhanced], dim=1)
        d4 = self.up_conv4(d4)
        
        # Decoder Block 3: d4 -> layer2 size
        d3 = self.decoder3(d4)
        if d3.shape[2:] != layer2_enhanced.shape[2:]:
            d3 = F.interpolate(d3, size=layer2_enhanced.shape[2:], mode='bilinear', align_corners=False)
        d3 = torch.cat([d3, layer2_enhanced], dim=1)
        d3 = self.up_conv3(d3)
        
        # Decoder Block 2: d3 -> layer1 size
        d2 = self.decoder2(d3)
        if d2.shape[2:] != layer1_enhanced.shape[2:]:
            d2 = F.interpolate(d2, size=layer1_enhanced.shape[2:], mode='bilinear', align_corners=False)
        d2 = torch.cat([d2, layer1_enhanced], dim=1)
        d2 = self.up_conv2(d2)
        
        # Decoder Block 1: d2 -> stem size
        d1 = self.decoder1(d2)
        if d1.shape[2:] != stem_enhanced.shape[2:]:
            d1 = F.interpolate(d1, size=stem_enhanced.shape[2:], mode='bilinear', align_corners=False)
        d1 = torch.cat([d1, stem_enhanced], dim=1)
        d1 = self.up_conv1(d1)
        
        # Final upsampling to original image size
        out = self.final_up(d1)
        
        # Ensure correct output size
        if out.shape[2:] != (self.img_size, self.img_size):
            out = F.interpolate(out, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)
        
        # Final convolution to produce class logits
        out = self.final_conv(out)
        
        return out


class CrossAttentionBlock(nn.Module):
    """
    Cross-attention block for enhancing skip connections.
    Uses higher-level features to attend to lower-level features.
    """
    def __init__(self, low_channels, high_channels, mid_channels=None):
        super().__init__()
        if mid_channels is None:
            mid_channels = low_channels // 2
        
        # Query from low-level features, Key and Value from high-level features
        self.query_conv = nn.Conv2d(low_channels, mid_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(high_channels, mid_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(high_channels, mid_channels, kernel_size=1)
        
        # Output projection and residual connection
        self.output_conv = nn.Conv2d(mid_channels, low_channels, kernel_size=1)
        
        # Scaling factor for dot-product attention
        self.scale = mid_channels ** -0.5
        
        # Normalization layers
        self.norm_low = nn.GroupNorm(8, low_channels)
        self.norm_out = nn.GroupNorm(8, low_channels)
        
    def forward(self, low_features, high_features):
        """
        Args:
            low_features: Lower-level features to be enhanced
            high_features: Higher-level features that provide context
            
        Returns:
            Enhanced low-level features
        """
        batch_size, _, h, w = low_features.shape
        
        # Apply normalization to inputs
        low_features_norm = self.norm_low(low_features)
        
        # If high features have different resolution, upsample them
        if high_features.shape[2:] != low_features.shape[2:]:
            high_features = F.interpolate(
                high_features, 
                size=low_features.shape[2:], 
                mode='bilinear', 
                align_corners=False
            )
        
        # Generate Q, K, V
        q = self.query_conv(low_features_norm)
        k = self.key_conv(high_features)
        v = self.value_conv(high_features)
        
        # Reshape for attention computation
        q = q.view(batch_size, -1, h*w).permute(0, 2, 1)  # B, HW, C
        k = k.view(batch_size, -1, h*w)  # B, C, HW
        v = v.view(batch_size, -1, h*w).permute(0, 2, 1)  # B, HW, C
        
        # Compute attention scores
        attn = torch.bmm(q, k) * self.scale  # B, HW, HW
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention weights to values
        out = torch.bmm(attn, v)  # B, HW, C
        out = out.permute(0, 2, 1).view(batch_size, -1, h, w)  # B, C, H, W
        
        # Project back to original dimension
        out = self.output_conv(out)
        
        # Residual connection and final normalization
        out = out + low_features
        out = self.norm_out(out)
        
        return out

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
        self.num_classes = num_classes
        # Initialize the appropriate backbone
        if backbone_type.startswith("resnet"):
            if backbone_type == "resnet18":
                resnet = torchvision.models.resnet18(weights=None)
                self._modify_resnet_for_4_channels(resnet, in_channels)
                self.feature_extractor = ResNetBackbone(resnet)
            elif backbone_type == "resnet50":
                resnet = torchvision.models.resnet50(weights=None)
                self._modify_resnet_for_4_channels(resnet, in_channels)
                self.feature_extractor = ResNetBackbone(resnet)
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
                self.feature_extractor = SwinBackbone(swin)
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
        
        if backbone_type == "resnet18":
            self.seg_head = ResNetBackboneUNet(self.feature_extractor.feature_dims, num_classes=2, img_size=224)
        elif backbone_type == "resnet50":
            self.seg_head = ResNetBackboneUNet(self.feature_extractor.feature_dims, num_classes=2, img_size=224)
        elif backbone_type == "swin-tiny":
            self.seg_head = SwinBackboneUNet(self.feature_extractor.feature_dims, num_classes=2, img_size=224)
        elif backbone_type == "vit-s":
            self.seg_head = ResNetBackboneUNet(self.feature_extractor.feature_dims, num_classes=2, img_size=224)
        else:
            # Initialize segmentation head with correct dimensions
            self.seg_head = HighResolutionHead(
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
    
    def focal_loss(self, outputs, targets, alpha=0.25, gamma=2.0, reduction='mean'):
        """
        Focal Loss implementation for supervised segmentation.
        
        Args:
            outputs: Model predictions (logits) [B, C, H, W]
            targets: Ground truth labels [B, H, W]
            alpha: Weighting factor for the rare class
            gamma: Focusing parameter that reduces the loss for well-classified examples
            reduction: How to reduce the loss ('mean', 'sum', or 'none')
        
        Returns:
            Focal loss value
        """
        # Convert targets to one-hot encoding
        batch_size, num_classes = outputs.shape[0], outputs.shape[1]
        one_hot_targets = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()
        
        # Apply softmax to get class probabilities
        probs = F.softmax(outputs, dim=1)
        
        # Get the probability for the target class
        pt = torch.sum(probs * one_hot_targets, dim=1)
        
        # Compute focal weight
        focal_weight = (1 - pt).pow(gamma)
        
        # Apply alpha weighting - different weight for each class
        if isinstance(alpha, (list, tuple)):
            # If alpha is per-class
            alpha_weight = torch.ones_like(pt)
            for c in range(num_classes):
                alpha_weight[targets == c] = alpha[c]
        else:
            # Simple binary alpha
            alpha_weight = torch.ones_like(pt)
            alpha_weight[targets == 1] = alpha        # Foreground (vine) weight
            alpha_weight[targets == 0] = 1 - alpha    # Background weight
        
        # Apply alpha to focal weight
        loss = -alpha_weight * focal_weight * torch.log(pt + 1e-8)
        
        # Apply reduction
        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss
    
    def training_step(self, batch, batch_idx):
        """Training step with mixed dice, weighted cross-entropy loss"""
        images, masks = batch
        logits = self(images)

        # Combined dice + cross entropy loss for segmentation
        #loss = self.combined_loss(logits, masks)
        # Cross entropy loss
        #loss = F.cross_entropy(logits, masks)
        
        # Focal loss
        loss = self.focal_loss(logits, masks)
        
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
 