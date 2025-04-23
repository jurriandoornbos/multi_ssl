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
import numpy as np
from sklearn.decomposition import PCA
import torchvision
import matplotlib.pyplot as plt
from copy import deepcopy

from multissl.models.seghead import ResNetBackbone, ViTExtractor, SwinBackbone

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
import numpy as np
from sklearn.decomposition import PCA
from torchvision import transforms
import matplotlib.pyplot as plt
from copy import deepcopy

from multissl.models.seghead import ResNetBackbone, ViTExtractor, SwinBackbone


class OnlineFeaturePCA(nn.Module):
    """
    Online PCA for feature visualization that learns the transformation
    directly from the data during the forward pass.
    """
    def __init__(self, in_channels, out_channels=3, eps=1e-10):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.eps = eps
        self.register_buffer('mean', torch.zeros(in_channels))
        self.register_buffer('components', torch.zeros(out_channels, in_channels))
        self.register_buffer('variance', torch.zeros(out_channels))
        self.register_buffer('n_samples_seen', torch.tensor(0))
        self.initialized = False
        
    def _init_params(self, features):
        """Initialize mean and covariance params from the initial batch"""
        # Features shape: [B, C, H, W]
        batch_size, channels, height, width = features.shape
        features_flat = features.permute(0, 2, 3, 1).reshape(-1, channels)
        
        # Calculate mean
        self.mean = features_flat.mean(0)
        
        # Center the data
        features_centered = features_flat - self.mean
        
        # Calculate covariance matrix (memory-efficient)
        cov_matrix = torch.mm(features_centered.t(), features_centered) / (features_centered.size(0) - 1)
        
        # Get eigenvalues and eigenvectors
        eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)
        
        # Sort eigenvalues in descending order
        idx = torch.argsort(eigenvalues, descending=True)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Select the top components
        self.variance = eigenvalues[:self.out_channels]
        self.components = eigenvectors[:, :self.out_channels].t()
        
        # Scale components by sqrt of eigenvalues for visualization
        scaling = torch.sqrt(self.variance + self.eps).view(-1, 1)
        self.components = self.components * scaling
        
        self.n_samples_seen = torch.tensor(features_flat.size(0))
        self.initialized = True
        
    def _update_params(self, features, update_proportion=0.01):
        """Update parameters incrementally with new batch of features"""
        # Features shape: [B, C, H, W]
        batch_size, channels, height, width = features.shape
        features_flat = features.permute(0, 2, 3, 1).reshape(-1, channels)
        
        # Incremental mean update
        batch_mean = features_flat.mean(0)
        new_n_samples = self.n_samples_seen + features_flat.size(0)
        alpha = features_flat.size(0) / float(new_n_samples)
        self.mean = (1 - alpha) * self.mean + alpha * batch_mean
        
        # Center the data
        features_centered = features_flat - self.mean
        
        # Calculate partial covariance matrix (memory-efficient)
        batch_cov = torch.mm(features_centered.t(), features_centered) / (features_centered.size(0) - 1)
        
        # Get eigenvalues and eigenvectors from updated covariance
        # Use mini-batch update for efficiency
        cov_update = update_proportion * batch_cov
        # Get existing eigenvectors in the right form for multiplication
        existing_components = self.components.t()
        # Approximately update the eigenvectors with gradient step
        updated_components = existing_components + torch.mm(cov_update, existing_components)
        
        # Re-orthogonalize (simplified Gram-Schmidt)
        # For a more stable implementation, we could use SVD or QR decomposition
        q, r = torch.linalg.qr(updated_components)
        
        # Get top eigenvectors and eigenvalues
        self.components = q[:, :self.out_channels].t()
        
        # Update sample count
        self.n_samples_seen = new_n_samples
        
    def transform(self, features):
        """Apply PCA transformation to features"""
        # Features shape: [B, C, H, W]
        batch_size, channels, height, width = features.shape
        
        # Reshape to [B*H*W, C]
        features_flat = features.permute(0, 2, 3, 1).reshape(-1, channels)
        
        # Center the data
        features_centered = features_flat - self.mean
        
        # Project using principal components
        transformed = torch.mm(features_centered, self.components.t())
        
        # Reshape back to [B, out_channels, H, W]
        transformed = transformed.reshape(batch_size, height, width, self.out_channels)
        transformed = transformed.permute(0, 3, 1, 2)
        
        return transformed
    
    def inverse_transform(self, transformed_features):
        """Inverse transform from reduced feature space"""
        # Features shape: [B, out_channels, H, W]
        batch_size, channels, height, width = transformed_features.shape
        
        # Reshape to [B*H*W, out_channels]
        features_flat = transformed_features.permute(0, 2, 3, 1).reshape(-1, channels)
        
        # Inverse project using principal components
        inverse_transformed = torch.mm(features_flat, self.components)
        
        # Add back the mean
        inverse_transformed = inverse_transformed + self.mean
        
        # Reshape back to [B, in_channels, H, W]
        inverse_transformed = inverse_transformed.reshape(batch_size, height, width, self.in_channels)
        inverse_transformed = inverse_transformed.permute(0, 3, 1, 2)
        
        return inverse_transformed
    
    def forward(self, features, update=True):
        """Forward pass applying PCA to input features"""
        if not self.initialized:
            with torch.no_grad():
                self._init_params(features)
        elif update and self.training:
            with torch.no_grad():
                self._update_params(features)
                
        return self.transform(features)
    

    
class FeaturePCA(pl.LightningModule):
    """
    PyTorch Lightning module for feature visualization using PCA.
    
    Extracts features from a specified layer of a backbone network,
    applies PCA to reduce dimensions for visualization, and upsamples
    the results back to the original image size.
    """
    def __init__(
        self, 
        backbone_type="resnet50",
        pretrained_path=None,
        loaded_weights = None,
        in_channels=4,
        layer="layer4",
        out_channels=3,
        img_size=224,
        lr=1e-4,
        weight_decay=1e-5,
        use_online_pca=True,
        normalize_output=True,
        enhance = False,
    ):
        super().__init__()
        
        self.backbone_type = backbone_type
        self.layer = layer
        self.img_size = img_size
        self.out_channels = out_channels
        self.use_online_pca = use_online_pca
        self.normalize_output = normalize_output
        self.enhance = enhance
        
        # Initialize the appropriate backbone for feature extraction
        if backbone_type.startswith("resnet"):
            from torchvision import models
            if backbone_type == "resnet18":
                resnet = models.resnet18(weights=None)
                self._modify_resnet_for_4_channels(resnet, in_channels)
                self.feature_extractor = ResNetBackbone(resnet)
            elif backbone_type == "resnet50":
                resnet = models.resnet50(weights=None)
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
        
        if loaded_weights:
            name = loaded_weights.__class__.__name__
            if name in ["SwinBackbone", "ResNetBackbone", "ResNetBackboneUNet", "ViTExtractor"]:  
                self.feature_extractor = loaded_weights

        # Detect feature dimensions by running a forward pass
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, img_size, img_size)
            features = self.feature_extractor(dummy_input)
            
            # Extract feature dimensions from the specified layer
            if layer not in features:
                raise ValueError(f"Layer '{layer}' not found in feature extractor. Available layers: {list(features.keys())}")
            
            layer_features = features[layer]
            feature_channels = layer_features.size(1)
            feature_size = layer_features.size(2)
            
            print(f"Feature extractor initialized with {layer} features:")
            print(f"  - Channels: {feature_channels}")
            print(f"  - Spatial size: {feature_size}x{feature_size}")
            print(f"  - Original image size: {img_size}x{img_size}")
            
        # Online PCA for dimensionality reduction
        if use_online_pca:
            self.pca = OnlineFeaturePCA(feature_channels, out_channels=out_channels)
        else:
            # If not using online PCA, we'll compute it during the first forward pass
            self.pca = None
            self.register_buffer('components', torch.zeros(out_channels, feature_channels))
            self.register_buffer('mean', torch.zeros(feature_channels))

        # Training parameters
        self.lr = lr
        self.weight_decay = weight_decay
        
        # For tracking when PCA is computed
        self.pca_computed = False
        self.save_hyperparameters(ignore=['feature_extractor', "loaded_weights"])
    
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
                if in_channels >= 3:
                    new_conv.weight[:, :3] = old_conv.weight
                    # For additional channels beyond RGB, initialize with average of RGB weights
                    # or just the red channel for the 4th channel (common for NIR)
                    for i in range(3, in_channels):
                        new_conv.weight[:, i] = old_conv.weight[:, 0]  # Use red channel for NIR
                else:
                    # For 1 or 2 channels, just copy the available weights
                    for i in range(in_channels):
                        new_conv.weight[:, i] = old_conv.weight[:, i % 3]
        
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
    
    def _compute_pca(self, features):
        """Compute PCA transformation from features"""
        # Detach features to avoid gradients
        features_detached = features.detach()
        
        # Get the original shape and flatten the spatial dimensions
        batch_size, channels, height, width = features_detached.shape
        features_flat = features_detached.permute(0, 2, 3, 1).reshape(-1, channels)
        
        # Use scikit-learn PCA for stable computation
        pca = PCA(n_components=self.out_channels)
        
        # Move to CPU for scikit-learn processing
        features_cpu = features_flat.cpu().numpy()
        pca.fit(features_cpu)
        
        # Store the transformation matrix and mean
        self.mean = torch.from_numpy(pca.mean_).float().to(features.device)
        self.components = torch.from_numpy(pca.components_).float().to(features.device)
        
        self.pca_computed = True
        
        print(f"PCA computed with explained variance ratios: {pca.explained_variance_ratio_}")
    
    def _apply_pca(self, features):
        """Apply pre-computed PCA transformation to features"""
        # Get the original shape and flatten the spatial dimensions
        batch_size, channels, height, width = features.shape
        features_flat = features.permute(0, 2, 3, 1).reshape(-1, channels)
        
        # Center the data
        features_centered = features_flat - self.mean
        
        # Project using principal components
        transformed = torch.mm(features_centered, self.components.t())
        
        # Reshape back to [B, out_channels, H, W]
        transformed = transformed.reshape(batch_size, height, width, self.out_channels)
        transformed = transformed.permute(0, 3, 1, 2)
        
        return transformed
    
    def forward(self, x):
        """
        Forward pass to extract features, apply PCA, and upsample with enhanced vibrant colors.
        
        Args:
            x: Input image tensor of shape [B, C, H, W]
            
        Returns:
            Visualization tensor of shape [B, out_channels, H, W]
        """
        # Extract features from the backbone
        x 
        features = self.feature_extractor(x)
        
        # Get features from the specified layer
        layer_features = features[self.layer]
        
        # Apply PCA for dimensionality reduction
        if self.use_online_pca:
            # Use online PCA module
            reduced_features = self.pca(layer_features)
        else:
            # Compute PCA if not already done
            if not self.pca_computed:
                with torch.no_grad():
                    self._compute_pca(layer_features)
            
            # Apply pre-computed PCA
            reduced_features = self._apply_pca(layer_features)
        
        # Enhance colors for more vibrant visualization
        if self.out_channels ==3 and self.enhance:

            enhanced_features = self._enhance_colors(reduced_features)
                
            # Apply color enhancement again after upsampling
            output = self._post_process_colors(enhanced_features)
        else:
            output = reduced_features
        
        return output

    def _enhance_colors(self, features):
        """
        Enhance colors to make them more vibrant.
        
        Args:
            features: Tensor of shape [B, C, H, W]
            
        Returns:
            Enhanced features with more vibrant colors
        """
        B, C, H, W = features.shape
        
        # Make sure we're working with 3 channels for RGB visualization
        if C >= 3:
            # Get the first 3 channels for RGB
            rgb_features = features[:, :3]
            
            # Center the features around zero
            centered = rgb_features - rgb_features.mean(dim=(2, 3), keepdim=True)
            
            # Scale features to increase contrast
            scaled = centered * 1.0
            
            # Apply non-linear transformation to emphasize strong activations
            enhanced = torch.tanh(scaled) * 0.5 + 0.5  # Map to [0, 1]
            
            # Increase saturation by scaling colors away from gray
            gray = enhanced.mean(dim=1, keepdim=True)
            saturated = enhanced + (enhanced - gray) * 0.7  # Increase saturation
            
            # Clip values to [0, 1]
            saturated = torch.clamp(saturated, 0, 1)
            
            # Apply gamma correction for more vibrant colors
            gamma = 1  # Gamma < 1 increases brightness
            gamma_corrected = saturated ** gamma
            
            return gamma_corrected
        else:
            # For fewer channels, we apply a colormap-like transformation
            # Map grayscale to artificial rainbow colors
            if C == 1:
                # Create artificial RGB channels from single channel
                # Red: original, Green: shifted, Blue: inverted
                channel = features[:, 0:1]
                red = channel
                green = torch.roll(channel, shifts=H//4, dims=2)  # Shifted version
                blue = 1.0 - channel  # Inverted version
                
                # Combine to create colorful output
                colorful = torch.cat([red, green, blue], dim=1)
                
                # Normalize each channel
                for i in range(3):
                    min_val = colorful[:, i:i+1].min(dim=2, keepdim=True)[0].min(dim=3, keepdim=True)[0]
                    max_val = colorful[:, i:i+1].max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]
                    divisor = torch.max(max_val - min_val, torch.tensor(1e-6, device=features.device))
                    colorful[:, i:i+1] = (colorful[:, i:i+1] - min_val) / divisor
                
                return colorful
            else:
                # For 2 channels, map to R and G, then derive B
                red = features[:, 0:1]
                green = features[:, 1:2]
                blue = 1.0 - (red + green) / 2.0  # Complementary color
                
                colorful = torch.cat([red, green, blue], dim=1)
                
                # Normalize each channel
                for i in range(3):
                    min_val = colorful[:, i:i+1].min(dim=2, keepdim=True)[0].min(dim=3, keepdim=True)[0]
                    max_val = colorful[:, i:i+1].max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]
                    divisor = torch.max(max_val - min_val, torch.tensor(1e-6, device=features.device))
                    colorful[:, i:i+1] = (colorful[:, i:i+1] - min_val) / divisor
                
                return colorful

    def _post_process_colors(self, features):
        """
        Final color enhancement after upsampling.
        
        Args:
            features: Tensor of shape [B, C, H, W]
            
        Returns:
            Color-enhanced features
        """
        B, C, H, W = features.shape
        
        # Ensure we're working with RGB (3 channels)
        if C >= 3:
            # Use just the first 3 channels
            rgb = features[:, :3]
            
            # Apply histogram-like stretching to each channel
            processed = torch.zeros_like(rgb)
            for i in range(3):
                channel = rgb[:, i:i+1]
                # Compute percentiles for robust stretching
                flat_channel = channel.reshape(B, -1)
                lower = torch.quantile(flat_channel, 0.05, dim=1, keepdim=True).unsqueeze(-1).unsqueeze(-1)
                upper = torch.quantile(flat_channel, 0.95, dim=1, keepdim=True).unsqueeze(-1).unsqueeze(-1)
                
                # Apply contrast stretching
                stretched = (channel - lower) / (upper - lower + 1e-6)
                stretched = torch.clamp(stretched, 0, 1)
                
                processed[:, i:i+1] = stretched
            
            # Color balancing - normalize each channel separately
            for i in range(3):
                percentile_99 = torch.quantile(processed[:, i].reshape(B, -1), 0.99, dim=1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                processed[:, i:i+1] = torch.clamp(processed[:, i:i+1] / (percentile_99 + 1e-6), 0, 1)
            
            # Apply final saturation boost
            gray = processed.mean(dim=1, keepdim=True)
            saturated = processed + (processed - gray) * 0.5
            saturated = torch.clamp(saturated, 0, 1)
            
            # If model has more than 3 output channels, preserve them
            if C > 3:
                result = torch.cat([saturated, features[:, 3:]], dim=1)
                return result
            else:
                return saturated
        else:
            # For fewer channels, just return normalized features
            return self._normalize_output(features)

    def _normalize_output(self, output):
        """
        Normalize output to [0, 1] range.
        
        Args:
            output: Tensor to normalize
            
        Returns:
            Normalized tensor
        """
        # Normalize each channel independently
        B, C, H, W = output.shape
        normalized = torch.zeros_like(output)
        
        for i in range(C):
            channel = output[:, i:i+1]
            min_val = channel.min(dim=2, keepdim=True)[0].min(dim=3, keepdim=True)[0]
            max_val = channel.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]
            
            # Avoid division by zero
            divisor = torch.max(max_val - min_val, torch.tensor(1e-6, device=output.device))
            normalized[:, i:i+1] = (channel - min_val) / divisor
        
        return normalized
    
    def training_step(self, batch, batch_idx):
        """Training step - currently just a no-op as we're using for visualization"""
        images, _ = batch  # Assuming batch is (images, labels) but we don't need labels
        output = self(images)
        
        # No loss function, this is primarily for visualization
        # For actual training, you could add a reconstruction loss here
        
        return {"visualizations": output.detach()}
    
    def validation_step(self, batch, batch_idx):
        """Validation step for visualization"""
        images, _ = batch
        output = self(images)
        
        # Log a sample image and visualization if using a logger
        if self.logger and batch_idx == 0:
            # Select the first image in the batch
            input_img = images[0].detach().cpu().numpy()
            vis_img = output[0].detach().cpu().numpy()
            
            # Create a figure with input and visualization
            fig, axs = plt.subplots(1, 2, figsize=(12, 6))
            
            # Plot the input image (using first 3 channels or create a false color composite)
            if input_img.shape[0] >= 3:
                # For RGB display, use first 3 channels
                rgb_input = input_img[:3].transpose(1, 2, 0)
                # Normalize to [0, 1]
                if rgb_input.max() > 1.0:
                    rgb_input = rgb_input / 255.0
                
                axs[0].imshow(np.clip(rgb_input, 0, 1))
            else:
                # For grayscale display
                axs[0].imshow(input_img[0], cmap='gray')
            
            # Plot the visualization
            vis_img = vis_img.transpose(1, 2, 0)
            if vis_img.shape[2] == 3:
                # For RGB visualization
                axs[1].imshow(np.clip(vis_img, 0, 1))
            else:
                # For grayscale visualization
                axs[1].imshow(vis_img[0], cmap='viridis')
            
            axs[0].set_title("Input Image")
            axs[1].set_title(f"PCA Visualization ({self.layer})")
            
            # Remove axis ticks
            for ax in axs:
                ax.set_xticks([])
                ax.set_yticks([])
            
            plt.tight_layout()
            
            # Log the figure
            self.logger.experiment.add_figure("Input vs. Visualization", fig, global_step=self.global_step)
        
        return {"visualizations": output.detach()}
    
    def configure_optimizers(self):
        """Configure optimizer with learning rate scheduler"""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        
        return optimizer
    
    def freeze_backbone(self):
        """Freeze backbone weights for feature extraction only"""
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
    
    def visualize_batch(self, batch, num_samples=4):
        """
        Generate visualizations for a batch of images
        
        Args:
            batch: Batch of images or (images, labels) tuple
            num_samples: Number of samples to visualize
            
        Returns:
            Tuple of (input images, visualizations)
        """
        self.eval()
        with torch.no_grad():
            # Extract images from batch if it's a tuple
            if isinstance(batch, tuple) or isinstance(batch, list):
                images = batch[0]
            else:
                images = batch
            
            # Limit to specified number of samples
            images = images[:num_samples]
            
            # Generate visualizations
            visualizations = self(images)
            
            return images.cpu(), visualizations.cpu()