# Copyright 2025 Jurrian Doornbos
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0



import pytorch_lightning as pl
import torch
import torchvision
from torch import nn
from lightly.loss import NegativeCosineSimilarity
from lightly.models.modules import SimSiamPredictionHead, SimSiamProjectionHead
from .lr import WarmupCosineAnnealingScheduler
from .msrgb_convnext import MSRGBConvNeXtFeatureExtractor

from einops import rearrange


class Flatten(nn.Module):
    """Simple module to flatten from [B, C, H, W] -> [B, C*H*W] or [B, C, 1, 1] -> [B, C]"""
    def forward(self, x):
        return x.flatten(start_dim=1)

class ResNetFeatureExtractor(nn.Module):
    """
    Properly extract features from a ResNet backbone at different layers
    """
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
        
        # Add adaptive pooling to ensure correct output size
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Flatten for output
        self.flatten = Flatten()
        
        # Store the expected output dimensions for each layer
        self.feature_dims = {
            'conv1': self.conv1.out_channels,  # Typically 64
            'layer1': self._get_layer_out_channels(self.layer1),  # 64 for ResNet18, 256 for ResNet50
            'layer2': self._get_layer_out_channels(self.layer2),  # 128 for ResNet18, 512 for ResNet50
            'layer3': self._get_layer_out_channels(self.layer3),  # 256 for ResNet18, 1024 for ResNet50
            'layer4': self._get_layer_out_channels(self.layer4),  # 512 for ResNet18, 2048 for ResNet50
        }
    
    def _get_layer_out_channels(self, layer):
        """Get the output channels for a layer by looking at its last block"""
        if hasattr(layer, 'conv3'):  # For bottleneck blocks (ResNet50+)
            return layer[-1].conv3.out_channels
        else:  # For basic blocks (ResNet18/34)
            return layer[-1].conv2.out_channels
    
    def forward(self, x):
        features = {}
        
        # Initial processing
        x = self.conv1(x)  # 64 channels
        x = self.bn1(x)
        x = self.relu(x)
        features['conv1'] = x
        
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
        
        # Average pool and then flatten output
        x = self.avgpool(x)
        features['flat'] = self.flatten(x)
        
        return features
    
    def backbone_forward(self, x):
        """Run a complete forward pass and return only the final flattened features"""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Apply average pooling to get consistent feature dimensions
        x = self.avgpool(x)
        return self.flatten(x)
    
class FastSiam(pl.LightningModule):
    def __init__(self, 
                 backbone="resnet18", 
                 hidden_dim=512, 
                 proj_dim=128, 
                 pred_dim=64, 
                 lr=0.125, 
                 in_channels=4, 
                 batch_size = 32,
                 epochs = 400,
                 momentum = 0.9,
                 weight_decay = 1e-4,
                 dataset_size = 1_000_000):
        super().__init__()

        # Load backbone dynamically based on user input
        if backbone == "resnet18":
            resnet = torchvision.models.resnet18(weights=None)
            self._modify_resnet_for_4_channels(resnet, in_channels)
            self.feature_extractor = ResNetFeatureExtractor(resnet)
            backbone_dim = 512  # ResNet18 final output dimension
            self.using_vit= False
            self.using_pasiphae = False

        elif backbone == "resnet50":
            resnet = torchvision.models.resnet50(weights=None)
            self._modify_resnet_for_4_channels(resnet, in_channels)
            self.feature_extractor = ResNetFeatureExtractor(resnet)
            backbone_dim = 2048  # ResNet50 final output dimension
            self.using_vit= False
            self.using_pasiphae = False

        elif backbone =="vit-s":
            from timm import create_model

            vit = create_model(
            "vit_small_patch16_224",
            pretrained=False,   # set True if you want to load TIMM's pretrained weights
            in_chans=in_channels,
            num_classes=0       # set 0 so that the model outputs features instead of logits
            )
            backbone_dim = 384
            self.backbone = vit
            self.using_vit= True
            self.using_pasiphae = False
            
        elif backbone == "swin-tiny":
            from timm import create_model
            
            swin = create_model(
                "swin_tiny_patch4_window7_224",
                pretrained=False,
                in_chans=in_channels,
                num_classes=0
            )
            backbone_dim = 768  # Output dimension for swin_tiny
            self.backbone = swin
            self.using_vit= True
            self.using_pasiphae = False
            
        elif backbone == "pasiphae":
            self.backbone =  MSRGBConvNeXtFeatureExtractor(
                model_name = "tiny")
            backbone_dim = 768 # output dim for convnext tiny   
            self.using_vit = False
            self.using_pasiphae = True

        else:
            raise ValueError("Unsupported backbone. Choose resnet18 or resnet50.")
        # Store if we're using a ViT or ResNet
        

        # Projection and Prediction heads
        self.projection_head = SimSiamProjectionHead(backbone_dim, hidden_dim, proj_dim)
        self.prediction_head = SimSiamPredictionHead(proj_dim, pred_dim, proj_dim)
        self.momentum = momentum
        self.weight_decay = weight_decay


        # Loss function
        self.criterion = NegativeCosineSimilarity()
        self.batch_size = batch_size
        # Learning rate
        # Base LR for batch_size=32
        self.lr = lr
        # Scale linearly for larger batch sizes
        self.scaled_lr = self.lr * (batch_size / 32.0)
        self.epochs = epochs

    def _modify_resnet_for_4_channels(self, resnet, in_channels):
        """Modifies the first ResNet conv layer to accept 4-channel input."""
        old_conv = resnet.conv1
        new_conv = nn.Conv2d(
            in_channels,  # Change from 3 to 4 channels
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False
        )

        # Reinitialize new conv weights while keeping pretrained filter values
        with torch.no_grad():
            new_conv.weight[:, :3] = old_conv.weight  # Copy original RGB filters
            new_conv.weight[:, 3] = old_conv.weight[:, 0]  # Initialize NIR with Red channel weights
        
        resnet.conv1 = new_conv  # Replace conv1 with modified conv

    def forward(self, x):
        """Forward pass for FastSiam"""
        if self.using_vit:
            f = self.backbone(x)         
        else:
            f = self.feature_extractor.backbone_forward(x)

        
        
        z = self.projection_head(f)
        p = self.prediction_head(z)
        z = z.detach()
        return z, p
    
    def extract_features(self, x):
        """Extract intermediate features from the backbone"""
        if self.using_vit:
            # For ViT, we don't have intermediate features in the same way
            return {'final': self.backbone(x)}
        else:
            return self.feature_extractor(x)
        
    def forward_pasiphae(self, batch, batch_idx):
        """
        Process batches with Pasiphae model for FastSiam with special handling for small/single sample batches.
        Combines processing across different input types to ensure batch normalization works properly.
        
        Args:
            batch: Dictionary containing:
                - rgb_only: Dict with 'data' (list of views) and 'indices' (sample indices)
                - ms_only: Dict with 'data' (list of views) and 'indices' (sample indices)
                - aligned: Dict with 'rgb' (list of views), 'ms' (list of views) and 'indices' (sample indices)
            batch_idx: Batch index
        
        Returns:
            features: List of (z, p) tuples for each view
            sample_features: Dictionary mapping sample indices to lists of feature indices
        """
        features = []  # Will store (z, p) tuples
        sample_features = {}  # Will map sample indices to list of feature indices
        
        # Calculate the number of views (should be the same across all types)
        num_views = 0
        if batch['rgb_only']['data']:
            num_views = len(batch['rgb_only']['data'])
        elif batch['ms_only']['data']:
            num_views = len(batch['ms_only']['data'])
        elif batch['aligned']['rgb']:
            num_views = len(batch['aligned']['rgb'])
        
        if num_views == 0:
            # Empty batch, return empty results
            return features, sample_features
        
        # Process each view separately, but combine all sample types within the view
        for view_idx in range(num_views):
            # Collect all embeddings for this view
            all_outputs = []
            all_sample_info = []  # To track which sample each output belongs to
            
            # 1. Process RGB-only samples for this view
            if batch['rgb_only']['data'] and view_idx < len(batch['rgb_only']['data']):
                rgb_view = batch['rgb_only']['data'][view_idx]
                for i, sample_idx in enumerate(batch['rgb_only']['indices']):
                    # Get individual sample but keep batch dimension
                    sample_rgb = rgb_view[i:i+1]
                    
                    # Pass through backbone
                    output = self.backbone(rgb=sample_rgb)["flat"]
                    
                    # Store output and sample info
                    all_outputs.append(output)
                    all_sample_info.append(('rgb_only', sample_idx, i))
            
            # 2. Process MS-only samples for this view
            if batch['ms_only']['data'] and view_idx < len(batch['ms_only']['data']):
                ms_view = batch['ms_only']['data'][view_idx]
                for i, sample_idx in enumerate(batch['ms_only']['indices']):
                    # Get individual sample but keep batch dimension
                    sample_ms = ms_view[i:i+1]
                    
                    # Pass through backbone
                    output = self.backbone(ms=sample_ms)["flat"]
                    
                    # Store output and sample info
                    all_outputs.append(output)
                    all_sample_info.append(('ms_only', sample_idx, i))
            
            # 3. Process aligned RGB+MS samples for this view
            if batch['aligned']['rgb'] and view_idx < len(batch['aligned']['rgb']):
                rgb_view = batch['aligned']['rgb'][view_idx]
                ms_view = batch['aligned']['ms'][view_idx]
                for i, sample_idx in enumerate(batch['aligned']['indices']):
                    # Get individual sample but keep batch dimension
                    sample_rgb = rgb_view[i:i+1]
                    sample_ms = ms_view[i:i+1]
                    
                    # Pass through backbone
                    output = self.backbone(rgb=sample_rgb, ms=sample_ms)["flat"]
                    
                    # Store output and sample info
                    all_outputs.append(output)
                    all_sample_info.append(('aligned', sample_idx, i))
            
            # If we have outputs for this view
            if all_outputs:
                # Combine all outputs into a single batch
                combined_outputs = torch.cat(all_outputs, dim=0)
                
                # Process through projection and prediction heads as a single batch
                z_batch = self.projection_head(combined_outputs)
                p_batch = self.prediction_head(z_batch)
                z_batch = z_batch.detach()  # Stop gradient flow through z
                
                # Split results back to individual samples
                for idx, (data_type, sample_idx, _) in enumerate(all_sample_info):
                    # Initialize sample features list if not already done
                    if sample_idx not in sample_features:
                        sample_features[sample_idx] = []
                    
                    # Extract this sample's features
                    z = z_batch[idx:idx+1]  # Keep batch dimension
                    p = p_batch[idx:idx+1]
                    
                    # Store feature and track its index
                    feature_idx = len(features)
                    features.append((z, p))
                    sample_features[sample_idx].append(feature_idx)
        
        return features, sample_features

    def training_step(self, batch, batch_idx):
        """
        Compute the SSL loss
        
        For regular FastSiam, computes loss between each view and the average of other views.
        For Pasiphae-based FastSiam, organizes features by sample and computes loss similarly.
        """
        if self.using_pasiphae:
            # Get features from all views and organize by sample
            features, sample_features = self.forward_pasiphae(batch, batch_idx)
            
            # Compute loss across all samples and views
            loss = 0.0
            num_samples = len(sample_features)
            
            if num_samples == 0:
                # Handle empty batch case - create a zero tensor with gradient
                return torch.tensor(0.0, requires_grad=True, device=self.device)
            
            sample_count = 0  # Count samples that actually contribute to loss
            
            # For each sample
            for sample_idx, feature_indices in sample_features.items():
                sample_loss = 0.0
                num_views = len(feature_indices)
                
                if num_views <= 1:
                    # Skip samples with only one view (can't compute contrastive loss)
                    continue
                
                sample_count += 1
                
                # For each view of this sample
                for i, feature_idx in enumerate(feature_indices):
                    # Get current view's prediction
                    p_i = features[feature_idx][1]
                    
                    # Get embeddings from other views of the same sample
                    other_zs = []
                    for j, other_idx in enumerate(feature_indices):
                        if j != i:  # Exclude current view
                            other_zs.append(features[other_idx][0])
                    
                    if other_zs:  # Only if there are other views
                        # Stack and mean the embeddings
                        other_zs = torch.cat(other_zs, dim=0)
                        mean_z_others = torch.mean(other_zs, dim=0, keepdim=True)
                        
                        # Compute loss for this view against mean of other views
                        view_loss = self.criterion(p_i, mean_z_others)
                        sample_loss += view_loss
                
                # Normalize by number of views and add to total loss
                if num_views > 1:
                    sample_loss = sample_loss / num_views
                    loss += sample_loss
            
            # Average over samples that contributed to loss
            if sample_count > 0:
                loss = loss / sample_count
            else:
                # If no samples had multiple views, return a zero tensor with gradient
                loss = torch.tensor(0.0, requires_grad=True, device=self.device)
        
        else:
            # Original FastSiam implementation for non-Pasiphae backbones
            views = batch[0]  # Get all views
            l = len(views)
            features = [self.forward(view) for view in views]
        
            zs = torch.stack([z for z, _ in features])
            ps = torch.stack([p for _, p in features])
            
            loss = 0.0
            for i in range(l):
                # Create mask to exclude current view
                mask = torch.arange(l, device=self.device) != i
                # Compute loss between current prediction and mean of other embeddings
                loss += self.criterion(ps[i], torch.mean(zs[mask], dim=0)) / l
        
        # Log the loss
        self.log("train_loss_ssl", loss)
        return loss

    def configure_optimizers(self):
        """Set up the optimizer"""

        optimizer = torch.optim.SGD(self.parameters(), 
                                lr=self.scaled_lr, 
                                momentum=self.momentum, 
                                weight_decay=self.weight_decay)
        

        total_steps = self.trainer.estimated_stepping_batches
        steps_per_epoch = total_steps/self.epochs
        scheduler = WarmupCosineAnnealingScheduler(
            optimizer, 
            total_steps = total_steps, 
            steps_per_epoch = steps_per_epoch, 
            warmup_fraction=0.5, 
        )

   

        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
    
    def lr_scheduler_step(self, scheduler,metric):

        scheduler.step(self.trainer.global_step)

class TokenPredictionHead(nn.Module):
    """Prediction head that operates on token sequences"""
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.layer1 = nn.Linear(in_dim, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.layer2 = nn.Linear(hidden_dim, out_dim)
        
    def forward(self, x):
        # x shape: [B, N, C]
        batch_size, num_tokens, in_dim = x.shape
        
        # Reshape for batch norm
        x = x.reshape(-1, in_dim)  # [B*N, C]
        x = self.layer1(x)
        x = x.reshape(-1, self.layer1.out_features)  # Ensure correct shape for BN
        x = self.bn(x)
        x = self.relu(x)
        x = self.layer2(x)
        
        # Reshape back to token sequence
        x = x.reshape(batch_size, num_tokens, -1)
        return x


class DualObjectiveFastSiam(FastSiam):
    """
    Enhanced FastSiam with dual objective learning (global and local)
    inspired by Galileo's approach
    """
    def __init__(self, 
                 backbone="pasiphae",
                 hidden_dim=512, 
                 proj_dim=128, 
                 pred_dim=64, 
                 lr=0.125, 
                 in_channels=4, 
                 batch_size=32,
                 epochs=400,
                 momentum=0.9,
                 weight_decay=1e-4,
                 local_weight=0.5,
                 global_weight=0.5):
        
        super().__init__(backbone=backbone, 
                         hidden_dim=hidden_dim,
                         proj_dim=proj_dim,
                         pred_dim=pred_dim,
                         lr=lr,
                         in_channels=in_channels,
                         batch_size=batch_size,
                         epochs=epochs,
                         momentum=momentum,
                         weight_decay=weight_decay)
        
        # Add prediction head for local features (for pixel-level learning)

        self.local_prediction_head = TokenPredictionHead(
            in_dim=self.backbone.embed_dim,  # 192 in your case
            hidden_dim=pred_dim,  # Original pred_dim
            out_dim=self.backbone.embed_dim  # Same as input dim
        )
        
        # Loss weights for balancing global and local objectives
        self.local_weight = local_weight
        self.global_weight = global_weight
    
    def extract_linear_projection_features(self, x):
        """
        Extract features directly from the linear projection layer
        (skipping transformer blocks as in Galileo's local learning)
        """
        if self.using_pasiphae:
            ms_data = x.get('ms')
            rgb_data = x.get('rgb')
            
            linear_features = []
            
            # Extract features from the tokenizer's linear projections
            if ms_data is not None:
                ms_tokens_dict = {}
                ms_channels = self.backbone.tokenizer.extract_single_channels(ms_data, 'MS_')
                
                for band_name, band_data in ms_channels.items():
                    # Get only the linear projection (Conv2d) output
                    proj = self.backbone.tokenizer.ms_embed[band_name].proj
                    tokens = proj(band_data)
                    tokens = rearrange(tokens, "b c h w -> b (h w) c")
                    linear_features.append(tokens)
            
            if rgb_data is not None:
                rgb_channels = self.backbone.tokenizer.extract_single_channels(rgb_data, 'RGB_')
                
                for band_name, band_data in rgb_channels.items():
                    # Get only the linear projection (Conv2d) output
                    proj = self.backbone.tokenizer.rgb_embed[band_name].proj
                    tokens = proj(band_data)
                    tokens = rearrange(tokens, "b c h w -> b (h w) c")
                    linear_features.append(tokens)
            
            # Concatenate all features
            if linear_features:
                linear_features = torch.cat(linear_features, dim=1)
                return linear_features
            else:
                return None
        
        elif self.using_vit:
            # For standard ViT backbones, extract patch embedding output
            # Access depends on the specific ViT implementation
            x_input = x if isinstance(x, torch.Tensor) else x.get('data')
            patch_embed = self.backbone.patch_embed(x_input)
            return patch_embed
        
        else:
            # For ResNet backbones, use the first conv layer output
            x_input = x if isinstance(x, torch.Tensor) else x.get('data')
            features = self.feature_extractor(x_input)
            return features['conv1'].flatten(2).transpose(1, 2)  # Convert to (B, N, C) format
    
    def training_step(self, batch, batch_idx):
        """Compute both global and local SSL losses with proper batching"""
        if self.using_pasiphae:
            # Get features and organize by sample type
            global_loss = 0.0
            local_loss = 0.0
            
            # Process data by type
            for data_type in ['rgb_only', 'ms_only', 'aligned']:
                # Skip empty data types
                if data_type == 'rgb_only' and not batch[data_type]['data']:
                    continue
                if data_type == 'ms_only' and not batch[data_type]['data']:
                    continue
                if data_type == 'aligned' and not batch[data_type]['rgb']:
                    continue
                
                # Number of views
                num_views = len(batch[data_type]['data'] if data_type != 'aligned' else batch[data_type]['rgb'])
                
                # Process each view as a batch instead of individual samples
                for view_idx in range(num_views):
                    # Prepare batch inputs for this view
                    if data_type == 'rgb_only':
                        rgb_data = batch[data_type]['data'][view_idx]
                        input_dict = {'rgb': rgb_data}
                    elif data_type == 'ms_only':
                        ms_data = batch[data_type]['data'][view_idx]
                        input_dict = {'ms': ms_data}
                    else:  # aligned
                        rgb_data = batch[data_type]['rgb'][view_idx]
                        ms_data = batch[data_type]['ms'][view_idx]
                        input_dict = {'rgb': rgb_data, 'ms': ms_data}
                    
                    # Extract global features (full backbone)
                    full_feat = self.backbone(**input_dict)
                    z = self.projection_head(full_feat)
                    p = self.prediction_head(z)
                    z = z.detach()  # Stop gradient
                    
                    # Extract local features (linear projection only)
                    local_feat = self.extract_linear_projection_features(input_dict)
                    local_p = self.local_prediction_head(local_feat)
                    
                    # Store features by view and sample
                    if view_idx == 0:
                        # Initialize storage for first view
                        global_zs_by_sample = [[] for _ in range(len(batch[data_type]['indices']))]
                        global_ps_by_sample = [[] for _ in range(len(batch[data_type]['indices']))]
                        local_feats_by_sample = [[] for _ in range(len(batch[data_type]['indices']))]
                        local_ps_by_sample = [[] for _ in range(len(batch[data_type]['indices']))]
                    
                    # Split batch results by sample
                    for i, sample_idx in enumerate(batch[data_type]['indices']):
                        global_zs_by_sample[i].append(z[i:i+1])
                        global_ps_by_sample[i].append(p[i:i+1])
                        local_feats_by_sample[i].append(local_feat[i:i+1])
                        local_ps_by_sample[i].append(local_p[i:i+1])
                
                # Compute losses for each sample with multiple views
                num_valid_samples = 0
                for i, sample_idx in enumerate(batch[data_type]['indices']):
                    if len(global_zs_by_sample[i]) <= 1:
                        # Skip samples with only one view
                        continue
                    
                    num_valid_samples += 1
                    
                    # Compute global loss for this sample
                    sample_global_loss = 0.0
                    for j in range(len(global_zs_by_sample[i])):
                        # Get prediction for this view
                        p_j = global_ps_by_sample[i][j]
                        
                        # Get embeddings from other views
                        other_zs = [global_zs_by_sample[i][k] for k in range(len(global_zs_by_sample[i])) if k != j]
                        other_zs = torch.cat(other_zs, dim=0)
                        mean_z_others = torch.mean(other_zs, dim=0, keepdim=True)
                        
                        # Compute loss
                        sample_global_loss += self.criterion(p_j, mean_z_others)
                    
                    # Average over views
                    sample_global_loss /= len(global_zs_by_sample[i])
                    global_loss += sample_global_loss
                    
                    # Compute local loss for this sample
                    sample_local_loss = 0.0
                    for j in range(len(local_ps_by_sample[i])):
                        # Get prediction for this view
                        local_p_j = local_ps_by_sample[i][j]
                        
                        # Get local features from other views
                        other_local_feats = [local_feats_by_sample[i][k].detach() for k in range(len(local_feats_by_sample[i])) if k != j]
                        other_local_feats = torch.cat(other_local_feats, dim=0)
                        mean_local_others = torch.mean(other_local_feats, dim=0, keepdim=True)
                        
                        # Compute loss
                        sample_local_loss += self.criterion(local_p_j, mean_local_others)
                    
                    # Average over views
                    sample_local_loss /= len(local_ps_by_sample[i])
                    local_loss += sample_local_loss
                
                # Normalize by number of valid samples
                if num_valid_samples > 0:
                    global_loss /= num_valid_samples
                    local_loss /= num_valid_samples
            
            # Handle case where no valid samples were found
            if global_loss == 0.0 and local_loss == 0.0:
                total_loss = torch.tensor(0.0, requires_grad=True, device=self.device)
            else:
                # Combine losses
                total_loss = self.global_weight * global_loss + self.local_weight * local_loss
            
            # Log losses
            self.log("train_global_loss", global_loss)
            self.log("train_local_loss", local_loss)
            self.log("train_loss_ssl", total_loss)
            
            return total_loss
            
        else:
            # For non-Pasiphae backbones
            views = batch[0]  # Get all views
            l = len(views)
            
            # Process all views as batches
            global_features = []
            local_features = []
            local_predictions = []
            
            for view in views:
                # Extract global features
                full_feat = self.backbone(view) if self.using_vit else self.feature_extractor.backbone_forward(view)
                z = self.projection_head(full_feat)
                p = self.prediction_head(z)
                z = z.detach()
                global_features.append((z, p))
                
                # Extract local features
                local_feat = self.extract_linear_projection_features(view)
                local_p = self.local_prediction_head(local_feat)
                local_features.append(local_feat)
                local_predictions.append(local_p)
            
            # Compute global loss
            global_loss = 0.0
            for i in range(l):
                z_i, p_i = global_features[i]
                other_zs = [global_features[j][0] for j in range(l) if j != i]
                other_zs = torch.cat(other_zs, dim=0)
                mean_z_others = torch.mean(other_zs, dim=0, keepdim=True)
                global_loss += self.criterion(p_i, mean_z_others) / l
            
            # Compute local loss
            local_loss = 0.0
            for i in range(l):
                local_p_i = local_predictions[i]
                other_local_feats = [local_features[j].detach() for j in range(l) if j != i]
                other_local_feats = torch.cat(other_local_feats, dim=0)
                mean_local_others = torch.mean(other_local_feats, dim=0, keepdim=True)
                local_loss += self.criterion(local_p_i, mean_local_others) / l
            
            # Combine losses
            total_loss = self.global_weight * global_loss + self.local_weight * local_loss
            
            # Log losses
            self.log("train_global_loss", global_loss)
            self.log("train_local_loss", local_loss)
            self.log("train_total_loss", total_loss)
            
            return total_loss