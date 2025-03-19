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
from copy import deepcopy

from multissl.models.seghead import SegmentationModel
from multissl.models.lr import CosineAnnealingWarmRestartsDecay

class DecoderBlock(nn.Module):
    """
    Decoder block for the style transfer generator.
    Uses transposed convolution to upsample features and includes skip connection handling.
    """
    def __init__(self, in_channels, skip_channels, out_channels=None):
        super().__init__()
        if out_channels is None:
            out_channels = skip_channels
            
        # Upsample via transposed convolution
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Fusion with skip connection
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x, skip):
        # Upsample input features
        x = self.upsample(x)
        
        # Ensure dimensions match
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            
        # Concatenate with skip connection
        x = torch.cat([x, skip], dim=1)
        
        # Apply fusion convolutions
        x = self.fusion(x)
        
        return x

class ResidualBlock(nn.Module):
    """
    Residual block used in the generator's bottleneck.
    """
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.in1 = nn.InstanceNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.in2 = nn.InstanceNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        residual = x
        x = self.relu(self.in1(self.conv1(x)))
        x = self.in2(self.conv2(x))
        x = x + residual  # Skip connection
        return self.relu(x)

class StyleTransferGenerator(nn.Module):
    """
    Style transfer generator using a pretrained encoder backbone.
    Implements a U-Net style architecture with residual connections.
    """
    def __init__(self, backbone, in_channels=4, out_channels=4):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.backbone = backbone
        
        # Get the feature dimensions from the backbone
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, 224, 224)
            features = self.backbone(dummy_input)
            self.feature_dims = {}
            for key, value in features.items():
                if isinstance(value, torch.Tensor):
                    self.feature_dims[key] = value.shape[1]  # Channel dimension
        
        # Add residual blocks at the bottleneck level
        bottleneck_channels = self.feature_dims['layer4']
        self.bottleneck = nn.Sequential(
            ResidualBlock(bottleneck_channels),
            ResidualBlock(bottleneck_channels),
            ResidualBlock(bottleneck_channels)
        )
        
        # Create decoder blocks
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(self.feature_dims['layer4'], self.feature_dims['layer3']),
            DecoderBlock(self.feature_dims['layer3'], self.feature_dims['layer2']),
            DecoderBlock(self.feature_dims['layer2'], self.feature_dims['layer1']),
            DecoderBlock(self.feature_dims['layer1'], self.feature_dims['stem'] 
                         if 'stem' in self.feature_dims else 64)
        ])
        
        # Final output layers to generate the transformed image
        stem_channels = self.feature_dims['stem'] if 'stem' in self.feature_dims else 64
        self.output_conv = nn.Sequential(
            nn.Conv2d(stem_channels, 64, kernel_size=3, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, kernel_size=1)
        )
        
        # Upsampling layer for final image size, if needed
        self.final_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
        # Whether to use residual connection for the final output (helps preserve content)
        self.use_residual = True
        
    def forward(self, x):
        # Extract features using the encoder backbone
        features = self.backbone(x)
        
        # Get the deepest features and apply bottleneck residual blocks
        bottleneck_features = features['layer4']
        bottleneck_output = self.bottleneck(bottleneck_features)
        
        # Initialize decoder path with bottleneck output
        decoder_output = bottleneck_output
        
        # Apply decoder blocks with skip connections
        skip_features = [features['layer3'], features['layer2'], features['layer1']]
        if 'stem' in features:
            skip_features.append(features['stem'])
        else:
            # If no stem features, use a low-level feature or create one
            if 'layer1' in features:
                # Create a simple stem feature by downsampling layer1
                skip_features.append(F.avg_pool2d(features['layer1'], kernel_size=2, stride=2))
            else:
                # Fallback: just use a zero tensor of the right shape
                h, w = x.shape[2] // 2, x.shape[3] // 2  # Half resolution
                skip_features.append(torch.zeros(x.shape[0], 64, h, w, device=x.device))
        
        # Apply decoder blocks with skip connections
        for i, decoder_block in enumerate(self.decoder_blocks):
            if i < len(skip_features):
                decoder_output = decoder_block(decoder_output, skip_features[i])
            else:
                # No skip connection available
                dummy_skip = torch.zeros_like(decoder_output)
                decoder_output = decoder_block(decoder_output, dummy_skip)
        
        # Final convolution to get output image
        output = self.output_conv(decoder_output)
        
        # If output size doesn't match input, upsample
        if output.shape[2:] != x.shape[2:]:
            output = F.interpolate(output, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        # Apply residual connection to preserve content
        if self.use_residual:
            output = torch.tanh(output)  # Range [-1, 1]
            output = output * 0.1  # Scale down the changes
            output = x + output  # Only learn the "style difference"
            output = torch.clamp(output, 0, 1)  # Ensure valid range
        else:
            output = torch.sigmoid(output)  # Range [0, 1]
            
        return output

class StyleDiscriminator(nn.Module):
    """
    Multi-scale patch discriminator for style transfer.
    Uses features from the pretrained backbone at multiple scales.
    """
    def __init__(self, backbone, in_channels=4):
        super().__init__()
        self.backbone = backbone
        self.in_channels = in_channels
        
        # Get feature dimensions from the backbone
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, 224, 224)
            features = self.backbone(dummy_input)
            self.feature_dims = {}
            for key, value in features.items():
                if isinstance(value, torch.Tensor):
                    self.feature_dims[key] = value.shape[1]  # Channel dimension
        
        # Create discriminator heads at multiple scales (PatchGAN style)
        self.discriminator_heads = nn.ModuleDict({
            'layer2': self._create_disc_head(self.feature_dims['layer2']),
            'layer3': self._create_disc_head(self.feature_dims['layer3']),
            'layer4': self._create_disc_head(self.feature_dims['layer4'])
        })
    
    def _create_disc_head(self, in_channels):
        """Create a discriminator head for a specific feature level"""
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels // 2, in_channels // 4, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels // 4, 1, kernel_size=1)
        )
    
    def forward(self, x):
        # Extract features using the backbone
        features = self.backbone(x)
        
        # Apply discriminator heads at multiple scales
        outputs = {}
        for layer_name, disc_head in self.discriminator_heads.items():
            if layer_name in features:
                outputs[layer_name] = disc_head(features[layer_name])
        
        return outputs

class CycleMeanTeacher(pl.LightningModule):
    """
    Style-Adaptive Mean Teacher for domain-adaptive semi-supervised segmentation.
    
    Extends MeanTeacher with style transfer capabilities for domain adaptation between
    labeled and unlabeled domains.
    """
    def __init__(
        self,
        backbone_type="resnet50",
        pretrained_path=None,
        in_channels=4,
        num_classes=2,
        img_size=224,
        lr=1e-4,
        style_transfer_lr=2e-4,
        weight_decay=1e-5,
        ema_decay=0.99,
        consistency_weight=1.0,
        consistency_rampup=40,
        confidence_threshold=0.8,
        style_transfer_weight=0.5,
        cycle_consistency_weight=10.0,
        identity_weight=5.0,
        class_weights=[1.0, 1.0],
        use_cutmix=True,
        cutmix_prob=0.5,
        update_teacher_every=1,
        update_style_every=1
    ):
        super().__init__()
        
        # Initialize student and teacher segmentation models
        self.student = SegmentationModel(
            backbone_type=backbone_type,
            pretrained_path=pretrained_path,
            in_channels=in_channels,
            num_classes=num_classes,
            img_size=img_size,
            lr=lr,
            weight_decay=weight_decay,
            class_weights=class_weights
        )
        
        # Initialize teacher as a copy of student with detached gradients
        self.teacher = deepcopy(self.student)
        self._disable_teacher_grad()
        
        # Initialize style transfer generators and discriminators
        # These share backbones with student and teacher to leverage pretrained weights
        
        # Generator from labeled to unlabeled domain
        self.gen_l2u = StyleTransferGenerator(
            backbone=deepcopy(self.student.feature_extractor),
            in_channels=in_channels,
            out_channels=in_channels
        )
        
        # Generator from unlabeled to labeled domain
        self.gen_u2l = StyleTransferGenerator(
            backbone=deepcopy(self.teacher.feature_extractor),
            in_channels=in_channels,
            out_channels=in_channels
        )
        
        # Discriminator for labeled domain
        self.disc_labeled = StyleDiscriminator(
            backbone=deepcopy(self.student.feature_extractor),
            in_channels=in_channels
        )
        
        # Discriminator for unlabeled domain
        self.disc_unlabeled = StyleDiscriminator(
            backbone=deepcopy(self.teacher.feature_extractor),
            in_channels=in_channels
        )
        
        # Freeze discriminator backbones to avoid interfering with feature extraction
        self._freeze_discriminator_backbones()
        
        # Save hyperparameters
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.img_size = img_size
        self.lr = lr
        self.style_transfer_lr = style_transfer_lr  # Separate LR for style transfer components
        self.weight_decay = weight_decay
        self.ema_decay = ema_decay
        self.consistency_weight = consistency_weight
        self.consistency_rampup = consistency_rampup
        self.confidence_threshold = confidence_threshold
        self.style_transfer_weight = style_transfer_weight
        self.cycle_consistency_weight = cycle_consistency_weight
        self.identity_weight = identity_weight
        self.class_weights = torch.tensor(class_weights)
        self.use_cutmix = use_cutmix
        self.cutmix_prob = cutmix_prob
        self.update_teacher_every = update_teacher_every
        self.update_style_every = update_style_every
        
        # For tracking training progress
        self.step_count = 0
        self._current_epoch = 0
        
        # Save hyperparameters for logging
        self.save_hyperparameters(ignore=['student', 'teacher', 'gen_l2u', 'gen_u2l', 
                                           'disc_labeled', 'disc_unlabeled'])
    
    def _disable_teacher_grad(self):
        """Disable gradient computation for teacher model"""
        for param in self.teacher.parameters():
            param.requires_grad = False
    
    def _freeze_discriminator_backbones(self):
        """Freeze backbone weights in discriminators"""
        for param in self.disc_labeled.backbone.parameters():
            param.requires_grad = False
        
        for param in self.disc_unlabeled.backbone.parameters():
            param.requires_grad = False
    
    def update_teacher(self):
        """Update teacher weights using EMA of student weights"""
        with torch.no_grad():
            for teacher_param, student_param in zip(
                self.teacher.parameters(), self.student.parameters()
            ):
                teacher_param.data.mul_(self.ema_decay).add_(
                    student_param.data, alpha=(1 - self.ema_decay)
                )
    
    def forward(self, x):
        """Forward pass using student model"""
        return self.student(x)
    
    def _get_current_consistency_weight(self):
        """Get consistency weight with linear ramp-up"""
        if self.consistency_rampup > 0:
            consistency_weight = self.consistency_weight * min(
                1.0, float(self._current_epoch) / self.consistency_rampup
            )
            return consistency_weight
        else:
            return self.consistency_weight
    
    def _get_current_style_weight(self):
        """Get style transfer weight with linear ramp-up"""
        rampup_length = self.consistency_rampup // 2  # Shorter ramp-up for style
        if rampup_length > 0:
            style_weight = self.style_transfer_weight * min(
                1.0, float(self._current_epoch) / rampup_length
            )
            return style_weight
        else:
            return self.style_transfer_weight
    
    def adversarial_loss(self, predictions, target_is_real=True):
        """
        Calculate adversarial loss (MSE/LSGAN style) for the discriminator outputs.
        
        Args:
            predictions: Dictionary of predictions at multiple scales
            target_is_real: Whether the target should be real (1) or fake (0)
            
        Returns:
            Mean adversarial loss across all scales
        """
        target_value = 1.0 if target_is_real else 0.0
        loss = 0.0
        count = 0
        
        # Calculate MSE loss at each scale
        for _, pred in predictions.items():
            target_tensor = torch.full_like(pred, target_value)
            loss += F.mse_loss(pred, target_tensor)
            count += 1
        
        if count > 0:
            return loss / count
        else:
            return torch.tensor(0.0, device=predictions[list(predictions.keys())[0]].device)
    
    def cycle_consistency_loss(self, real_img, reconstructed_img):
        """
        Calculate cycle consistency loss (L1 loss) between original and reconstructed images.
        
        Args:
            real_img: Original image
            reconstructed_img: Image after passing through both generators
            
        Returns:
            L1 loss between original and reconstructed images
        """
        return F.l1_loss(reconstructed_img, real_img)
    
    def identity_loss(self, real_img, identity_img):
        """
        Calculate identity mapping loss (L1 loss) for when a generator 
        receives an image from its target domain.
        
        Args:
            real_img: Original image
            identity_img: Image after passing through generator
            
        Returns:
            L1 loss for identity mapping
        """
        return F.l1_loss(identity_img, real_img)
    
    def feature_consistency_loss(self, real_features, fake_features, layers=None):
        """
        Calculate feature consistency loss between real and fake images.
        
        Args:
            real_features: Features from real image
            fake_features: Features from fake image
            layers: Optional list of specific layers to use
            
        Returns:
            Mean L2 loss between feature maps
        """
        if layers is None:
            # Use all layers except the final output
            layers = [key for key in real_features.keys() 
                     if isinstance(real_features[key], torch.Tensor) and key != 'flat']
        
        # Calculate L2 loss between feature maps
        loss = 0.0
        count = 0
        for layer in layers:
            if layer in real_features and layer in fake_features:
                real_feat = real_features[layer]
                fake_feat = fake_features[layer]
                
                # Ensure spatial dimensions match
                if real_feat.shape[2:] != fake_feat.shape[2:]:
                    fake_feat = F.interpolate(
                        fake_feat, size=real_feat.shape[2:], 
                        mode='bilinear', align_corners=False
                    )
                
                # L2 loss
                loss += F.mse_loss(real_feat, fake_feat)
                count += 1
        
        if count > 0:
            return loss / count
        else:
            return torch.tensor(0.0, device=real_features[layers[0]].device)
    
    def training_step(self, batch, batch_idx, optimizer_idx=0):
        """
        Training step with alternating optimization between segmentation and style transfer.
        
        Args:
            batch: Tuple of (labeled_batch, unlabeled_batch)
            batch_idx: Batch index
            optimizer_idx: Index of optimizer being used (0: segmentation, 1: style transfer)
            
        Returns:
            Loss for the current optimization step
        """
        labeled_batch, unlabeled_batch = batch
        labeled_images, labeled_masks = labeled_batch
        unlabeled_images = unlabeled_batch
        
        # Skip empty batches
        if labeled_images.size(0) == 0 or unlabeled_images.size(0) == 0:
            dummy_loss = torch.tensor(0.0, device=self.device)
            return dummy_loss
        
        # Get current weights
        consistency_weight = self._get_current_consistency_weight()
        style_weight = self._get_current_style_weight()
        
        # Segmentation optimization step (optimizer_idx = 0)
        if optimizer_idx == 0:
            # Generate domain-adapted images
            with torch.no_grad():
                labeled_as_unlabeled = self.gen_l2u(labeled_images)
                unlabeled_as_labeled = self.gen_u2l(unlabeled_images)
            
            # Teacher forward pass
            with torch.no_grad():
                teacher_logits_l = self.teacher(labeled_images)
                teacher_logits_u = self.teacher(unlabeled_as_labeled)
            
            # Student forward pass
            student_logits_l = self.student(labeled_images)
            student_logits_u = self.student(unlabeled_as_labeled)
            
            # Calculate supervised loss on labeled data
            supervised_loss = self.student.focal_loss(student_logits_l, labeled_masks)
            
            # Calculate pseudo-label loss on adapted unlabeled data
            pseudo_loss = self.focal_pseudo_loss(
                student_logits_u, teacher_logits_u, 
                threshold=self.confidence_threshold
            )
            
            # Combined segmentation loss
            seg_loss = supervised_loss + consistency_weight * pseudo_loss
            
            # Log losses
            self.log("train_supervised_loss", supervised_loss, prog_bar=True)
            self.log("train_pseudo_loss", pseudo_loss)
            self.log("train_seg_total_loss", seg_loss, prog_bar=True)
            self.log("train_consistency_weight", consistency_weight)
            
            return seg_loss
        
        # Style transfer optimization step (optimizer_idx = 1)
        elif optimizer_idx == 1:
            # Only update style transfer every N steps
            if self.step_count % self.update_style_every != 0:
                dummy_loss = torch.tensor(0.0, device=self.device)
                return dummy_loss
            
            # Forward cycle: labeled -> unlabeled -> labeled
            fake_unlabeled = self.gen_l2u(labeled_images)
            reconstructed_labeled = self.gen_u2l(fake_unlabeled)
            
            # Backward cycle: unlabeled -> labeled -> unlabeled
            fake_labeled = self.gen_u2l(unlabeled_images)
            reconstructed_unlabeled = self.gen_l2u(fake_labeled)
            
            # Identity mappings
            identity_labeled = self.gen_u2l(labeled_images)
            identity_unlabeled = self.gen_l2u(unlabeled_images)
            
            # Discriminator outputs
            disc_real_labeled = self.disc_labeled(labeled_images)
            disc_fake_labeled = self.disc_labeled(fake_labeled.detach())
            disc_real_unlabeled = self.disc_unlabeled(unlabeled_images)
            disc_fake_unlabeled = self.disc_unlabeled(fake_unlabeled.detach())
            
            # Feature consistency using teacher and student backbones
            with torch.no_grad():
                real_labeled_features = self.student.feature_extractor(labeled_images)
                real_unlabeled_features = self.teacher.feature_extractor(unlabeled_images)
            
            fake_labeled_features = self.student.feature_extractor(fake_labeled)
            fake_unlabeled_features = self.teacher.feature_extractor(fake_unlabeled)
            
            # Calculate style transfer losses
            
            # Generator adversarial losses
            gen_l2u_loss = self.adversarial_loss(
                self.disc_unlabeled(fake_unlabeled), target_is_real=True
            )
            gen_u2l_loss = self.adversarial_loss(
                self.disc_labeled(fake_labeled), target_is_real=True
            )
            
            # Discriminator losses
            disc_labeled_loss = 0.5 * (
                self.adversarial_loss(disc_real_labeled, target_is_real=True) +
                self.adversarial_loss(disc_fake_labeled, target_is_real=False)
            )
            disc_unlabeled_loss = 0.5 * (
                self.adversarial_loss(disc_real_unlabeled, target_is_real=True) +
                self.adversarial_loss(disc_fake_unlabeled, target_is_real=False)
            )
            
            # Cycle consistency losses
            forward_cycle_loss = self.cycle_consistency_loss(
                labeled_images, reconstructed_labeled
            )
            backward_cycle_loss = self.cycle_consistency_loss(
                unlabeled_images, reconstructed_unlabeled
            )
            cycle_loss = forward_cycle_loss + backward_cycle_loss
            
            # Identity mapping losses
            identity_labeled_loss = self.identity_loss(labeled_images, identity_labeled)
            identity_unlabeled_loss = self.identity_loss(unlabeled_images, identity_unlabeled)
            identity_loss = identity_labeled_loss + identity_unlabeled_loss
            
            # Feature consistency losses
            feature_l2u_loss = self.feature_consistency_loss(
                real_unlabeled_features, fake_unlabeled_features
            )
            feature_u2l_loss = self.feature_consistency_loss(
                real_labeled_features, fake_labeled_features
            )
            feature_loss = feature_l2u_loss + feature_u2l_loss
            
            # Combined style transfer loss
            style_loss = (
                gen_l2u_loss + gen_u2l_loss +  # Generator adversarial losses
                disc_labeled_loss + disc_unlabeled_loss +  # Discriminator losses
                self.cycle_consistency_weight * cycle_loss +  # Cycle consistency
                self.identity_weight * identity_loss +  # Identity mapping
                2.0 * feature_loss  # Feature consistency (higher weight)
            )
            
            # Scale by current style weight
            total_style_loss = style_weight * style_loss
            
            # Log style transfer losses
            self.log("train_gen_l2u_loss", gen_l2u_loss)
            self.log("train_gen_u2l_loss", gen_u2l_loss)
            self.log("train_disc_labeled_loss", disc_labeled_loss)
            self.log("train_disc_unlabeled_loss", disc_unlabeled_loss)
            self.log("train_cycle_loss", cycle_loss)
            self.log("train_identity_loss", identity_loss)
            self.log("train_feature_loss", feature_loss)
            self.log("train_total_style_loss", total_style_loss, prog_bar=True)
            self.log("train_style_weight", style_weight)
            
            # Return the combined style transfer loss
            return total_style_loss
    
    def focal_pseudo_loss(self, student_logits, teacher_logits, gamma=2.0, threshold=0.6):
        """
        Calculate focal pseudo-label loss for semi-supervised learning.
        
        Args:
            student_logits: Logits from student model
            teacher_logits: Logits from teacher model (for pseudo-labels)
            gamma: Focal loss gamma parameter
            threshold: Confidence threshold for pseudo-labels
            
        Returns:
            Focal loss with pseudo-labels
        """
        with torch.no_grad():
            teacher_probs = F.softmax(teacher_logits, dim=1)
            confidence, pseudo_labels = torch.max(teacher_probs, dim=1)
            mask = (confidence > threshold).float()
        
        # Get student probabilities
        student_probs = F.softmax(student_logits, dim=1)
        
        # Extract the predicted probability for the ground truth class
        batch_size, num_classes, h, w = student_logits.shape
        student_pt = torch.gather(student_probs, 1, pseudo_labels.unsqueeze(1)).squeeze(1)
        
        # Focal loss formula: -alpha * (1 - pt)^gamma * log(pt)
        focal_weight = (1 - student_pt).pow(gamma)
        
        # Standard cross entropy
        loss = F.cross_entropy(student_logits, pseudo_labels, reduction='none')
        
        # Apply focal weighting and confidence mask
        weighted_loss = focal_weight * loss * mask
        
        if mask.sum() > 0:
            return weighted_loss.sum() / mask.sum()
        return loss.mean()
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        """
        Update teacher model with EMA of student weights after training batch.
        """
        # Increment step counter
        self.step_count += 1
        
        # Update teacher model weights using EMA (every N steps)
        if self.step_count % self.update_teacher_every == 0:
            self.update_teacher()
    
    def on_train_epoch_start(self):
        """Update current epoch counter"""
        self._current_epoch = self.trainer.current_epoch
    
    def validation_step(self, batch, batch_idx):
        """
        Validation step using the teacher model.
        
        Teacher model is used for validation as it should perform better
        due to the ensembling effect of EMA updates.
        """
        images, masks = batch
        
        # Forward through teacher model
        with torch.no_grad():
            teacher_logits = self.teacher(images)
        
        # Calculate validation loss
        val_loss = F.cross_entropy(teacher_logits, masks)
        
        # Calculate metrics
        preds = torch.argmax(teacher_logits, dim=1)
        accuracy = (preds == masks).float().mean()
        
        # Calculate IoU for each class
        iou_per_class = []
        for cls in range(self.num_classes):
            intersection = ((preds == cls) & (masks == cls)).float().sum()
            union = ((preds == cls) | (masks == cls)).float().sum()
            iou = intersection / (union + 1e-6)
            iou_per_class.append(iou)
        
        mean_iou = torch.stack(iou_per_class).mean()
        
        # Log metrics
        self.log("val_loss", val_loss, prog_bar=True)
        self.log("val_accuracy", accuracy, prog_bar=True)
        self.log("val_mean_iou", mean_iou, prog_bar=True)
        
        return val_loss
    
    def configure_optimizers(self):
        """
        Configure optimizers with separate parameter groups and learning rates.
        Returns one optimizer for segmentation and one for style transfer.
        """
        # Segmentation optimizer (student model only)
        segmentation_params = self.student.parameters()
        
        # Style transfer parameters
        style_transfer_params = list(self.gen_l2u.parameters()) + \
                               list(self.gen_u2l.parameters()) + \
                               list(self.disc_labeled.parameters()) + \
                               list(self.disc_unlabeled.parameters())
        
        # Create optimizers
        segmentation_optimizer = torch.optim.AdamW(
            segmentation_params,
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        
        style_transfer_optimizer = torch.optim.AdamW(
            style_transfer_params,
            lr=self.style_transfer_lr,
            weight_decay=self.weight_decay * 0.1  # Lower weight decay for style transfer
        )
        
        # Calculate appropriate T_0 (first cycle length)
        T_0 = max(5, int(self.trainer.max_epochs * 0.2))  # At least 5 epochs
        
        # Create schedulers with warmup
        segmentation_scheduler = CosineAnnealingWarmRestartsDecay(
            segmentation_optimizer,
            T_0=T_0,
            T_mult=2,
            eta_min=self.lr * 0.01,
            warmup_epochs=5,
            warmup_start_lr=self.lr * 0.1,
            decay_factor=0.8
        )
        
        style_transfer_scheduler = CosineAnnealingWarmRestartsDecay(
            style_transfer_optimizer,
            T_0=T_0,
            T_mult=2,
            eta_min=self.style_transfer_lr * 0.01,
            warmup_epochs=3,  # Shorter warmup for style transfer
            warmup_start_lr=self.style_transfer_lr * 0.1,
            decay_factor=0.8
        )
        
        return [
            {"optimizer": segmentation_optimizer, 
             "lr_scheduler": {"scheduler": segmentation_scheduler, "interval": "epoch"}, 
             "frequency": 1},
            {"optimizer": style_transfer_optimizer, 
             "lr_scheduler": {"scheduler": style_transfer_scheduler, "interval": "epoch"}, 
             "frequency": 1}
        ]
    
    def test_step(self, batch, batch_idx):
        """
        Run test step on a single batch using the teacher model
        """
        images, masks = batch
        
        # Forward pass through teacher
        with torch.no_grad():
            logits = self.teacher(images)
        
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
    
    def visualize_domain_adaptation(self, images_labeled, images_unlabeled, n_samples=4):
        """
        Visualize the results of domain adaptation between labeled and unlabeled domains.
        
        Args:
            images_labeled: Batch of labeled images
            images_unlabeled: Batch of unlabeled images
            n_samples: Number of sample images to visualize
            
        Returns:
            Dictionary of visualizations of original and adapted images
        """
        self.eval()
        with torch.no_grad():
            # Generate domain-adapted images
            labeled_as_unlabeled = self.gen_l2u(images_labeled[:n_samples])
            unlabeled_as_labeled = self.gen_u2l(images_unlabeled[:n_samples])
            
            # Generate cycle reconstructions
            reconstructed_labeled = self.gen_u2l(labeled_as_unlabeled)
            reconstructed_unlabeled = self.gen_l2u(unlabeled_as_labeled)
            
            # Prepare visualization results
            results = {
                'labeled_original': images_labeled[:n_samples].cpu(),
                'unlabeled_original': images_unlabeled[:n_samples].cpu(),
                'labeled_as_unlabeled': labeled_as_unlabeled.cpu(),
                'unlabeled_as_labeled': unlabeled_as_labeled.cpu(),
                'labeled_reconstructed': reconstructed_labeled.cpu(),
                'unlabeled_reconstructed': reconstructed_unlabeled.cpu()
            }
            
            return results
            
    def test(self, test_loader=None):
        """
        Comprehensive evaluation of segmentation model on test set
        
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


class GANMeanTeacher(pl.LightningModule):
    """
    Simplified Style-Adaptive Mean Teacher for domain-adaptive semi-supervised segmentation.
    
    Uses a single direction style transfer (unlabeled→labeled) to make unlabeled data 
    look like labeled data. This simplifies the architecture while maintaining 
    the benefits of domain adaptation.
    """
    def __init__(
        self,
        backbone_type="resnet50",
        pretrained_path=None,
        in_channels=4,
        num_classes=2,
        img_size=224,
        lr=1e-4,
        style_transfer_lr=2e-4,
        weight_decay=1e-5,
        ema_decay=0.99,
        consistency_weight=1.0,
        consistency_rampup=40,
        confidence_threshold=0.8,
        style_transfer_weight=0.5,
        identity_weight=5.0,
        feature_consistency_weight=2.0,
        class_weights=[1.0, 1.0],
        use_cutmix=True,
        cutmix_prob=0.5,
        update_teacher_every=1,
        update_style_every=1
    ):
        super().__init__()
        
        # Initialize student and teacher segmentation models
        self.student = SegmentationModel(
            backbone_type=backbone_type,
            pretrained_path=pretrained_path,
            in_channels=in_channels,
            num_classes=num_classes,
            img_size=img_size,
            lr=lr,
            weight_decay=weight_decay,
            class_weights=class_weights
        )
        
        # Initialize teacher as a copy of student with detached gradients
        self.teacher = deepcopy(self.student)
        self._disable_teacher_grad()
        
        # Initialize style transfer generator
        # Generator from unlabeled to labeled domain only (simplified)
        self.gen_u2l = StyleTransferGenerator(
            backbone=deepcopy(self.teacher.feature_extractor),
            in_channels=in_channels,
            out_channels=in_channels
        )
        
        # Discriminator for labeled domain
        self.disc_labeled = StyleDiscriminator(
            backbone=deepcopy(self.student.feature_extractor),
            in_channels=in_channels
        )
        
        # Freeze discriminator backbone to avoid interfering with feature extraction
        self._freeze_discriminator_backbone()
        
        # Save hyperparameters
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.img_size = img_size
        self.lr = lr
        self.style_transfer_lr = style_transfer_lr
        self.weight_decay = weight_decay
        self.ema_decay = ema_decay
        self.consistency_weight = consistency_weight
        self.consistency_rampup = consistency_rampup
        self.confidence_threshold = confidence_threshold
        self.style_transfer_weight = style_transfer_weight
        self.identity_weight = identity_weight
        self.feature_consistency_weight = feature_consistency_weight
        self.class_weights = torch.tensor(class_weights)
        self.use_cutmix = use_cutmix
        self.cutmix_prob = cutmix_prob
        self.update_teacher_every = update_teacher_every
        self.update_style_every = update_style_every
        
        # For tracking training progress
        self.step_count = 0
        self._current_epoch = 0
        #for dopuble model optimzers
        self.automatic_optimization = False
        
        # Save hyperparameters for logging
        self.save_hyperparameters(ignore=['student', 'teacher', 'gen_u2l', 'disc_labeled'])
    
    def _disable_teacher_grad(self):
        """Disable gradient computation for teacher model"""
        for param in self.teacher.parameters():
            param.requires_grad = False
    
    def _freeze_discriminator_backbone(self):
        """Freeze backbone weights in discriminator"""
        for param in self.disc_labeled.backbone.parameters():
            param.requires_grad = False
    
    def update_teacher(self):
        """Update teacher weights using EMA of student weights"""
        with torch.no_grad():
            for teacher_param, student_param in zip(
                self.teacher.parameters(), self.student.parameters()
            ):
                teacher_param.data.mul_(self.ema_decay).add_(
                    student_param.data, alpha=(1 - self.ema_decay)
                )
    
    def forward(self, x):
        """Forward pass using student model"""
        return self.student(x)
    
    def _get_current_consistency_weight(self):
        """Get consistency weight with linear ramp-up"""
        if self.consistency_rampup > 0:
            consistency_weight = self.consistency_weight * min(
                1.0, float(self._current_epoch) / self.consistency_rampup
            )
            return consistency_weight
        else:
            return self.consistency_weight
    
    def _get_current_style_weight(self):
        """Get style transfer weight with linear ramp-up"""
        rampup_length = self.consistency_rampup // 2  # Shorter ramp-up for style
        if rampup_length > 0:
            style_weight = self.style_transfer_weight * min(
                1.0, float(self._current_epoch) / rampup_length
            )
            return style_weight
        else:
            return self.style_transfer_weight
    
    def adversarial_loss(self, predictions, target_is_real=True):
        """
        Calculate adversarial loss (MSE/LSGAN style) for the discriminator outputs.
        
        Args:
            predictions: Dictionary of predictions at multiple scales
            target_is_real: Whether the target should be real (1) or fake (0)
            
        Returns:
            Mean adversarial loss across all scales
        """
        target_value = 1.0 if target_is_real else 0.0
        loss = 0.0
        count = 0
        
        # Calculate MSE loss at each scale
        for _, pred in predictions.items():
            target_tensor = torch.full_like(pred, target_value)
            loss += F.mse_loss(pred, target_tensor)
            count += 1
        
        if count > 0:
            return loss / count
        else:
            return torch.tensor(0.0, device=predictions[list(predictions.keys())[0]].device)
    
    def identity_loss(self, real_img, identity_img):
        """
        Calculate identity mapping loss (L1 loss) for when a generator 
        receives an image from its target domain.
        
        Args:
            real_img: Original image
            identity_img: Image after passing through generator
            
        Returns:
            L1 loss for identity mapping
        """
        return F.l1_loss(identity_img, real_img)
    
    def feature_consistency_loss(self, real_features, fake_features, layers=None):
        """
        Calculate feature consistency loss between real and fake images.
        
        Args:
            real_features: Features from real image
            fake_features: Features from fake image
            layers: Optional list of specific layers to use
            
        Returns:
            Mean L2 loss between feature maps
        """
        if layers is None:
            # Use all layers except the final output
            layers = [key for key in real_features.keys() 
                     if isinstance(real_features[key], torch.Tensor) and key != 'flat']
        
        # Calculate L2 loss between feature maps
        loss = 0.0
        count = 0
        for layer in layers:
            if layer in real_features and layer in fake_features:
                real_feat = real_features[layer]
                fake_feat = fake_features[layer]
                
                # Ensure spatial dimensions match
                if real_feat.shape[2:] != fake_feat.shape[2:]:
                    fake_feat = F.interpolate(
                        fake_feat, size=real_feat.shape[2:], 
                        mode='bilinear', align_corners=False
                    )
                
                # L2 loss
                loss += F.mse_loss(real_feat, fake_feat)
                count += 1
        
        if count > 0:
            return loss / count
        else:
            return torch.tensor(0.0, device=real_features[layers[0]].device)
    
    def training_step(self, batch, batch_idx):
        """
        Training step with manual optimization between segmentation and style transfer.
        
        Args:
            batch: Tuple of (labeled_batch, unlabeled_batch)
            batch_idx: Batch index
            
        Returns:
            None (losses are handled with manual optimization)
        """
        # Get optimizers
        seg_optimizer, style_optimizer = self.optimizers()
        
        labeled_batch, unlabeled_batch = batch
        labeled_images, labeled_masks = labeled_batch
        unlabeled_images = unlabeled_batch
        
        # Skip empty batches
        if labeled_images.size(0) == 0 or unlabeled_images.size(0) == 0:
            return None
        
        # Get current weights
        consistency_weight = self._get_current_consistency_weight()
        style_weight = self._get_current_style_weight()
        
        # ==========================
        # STEP 1: Segmentation Optimization
        # ==========================
        
        # Generate domain-adapted images (unlabeled→labeled)
        with torch.no_grad():
            unlabeled_as_labeled = self.gen_u2l(unlabeled_images)
        
        # Teacher forward pass
        with torch.no_grad():
            teacher_logits_l = self.teacher(labeled_images)
            teacher_logits_u = self.teacher(unlabeled_as_labeled)
        
        # Student forward pass
        student_logits_l = self.student(labeled_images)
        student_logits_u = self.student(unlabeled_as_labeled)
        
        # Calculate supervised loss on labeled data
        supervised_loss = self.student.focal_loss(student_logits_l, labeled_masks)
        
        # Calculate pseudo-label loss on adapted unlabeled data
        pseudo_loss = self.focal_pseudo_loss(
            student_logits_u, teacher_logits_u, 
            threshold=self.confidence_threshold
        )
        
        # Combined segmentation loss
        seg_loss = supervised_loss + consistency_weight * pseudo_loss
        
        # Optimize segmentation model
        seg_optimizer.zero_grad()
        self.manual_backward(seg_loss)
        seg_optimizer.step()
        
        # Log losses
        self.log("train_supervised_loss", supervised_loss, prog_bar=True)
        self.log("train_pseudo_loss", pseudo_loss)
        self.log("train_seg_total_loss", seg_loss, prog_bar=True)
        self.log("train_consistency_weight", consistency_weight)
        
        # ==========================
        # STEP 2: Style Transfer Optimization 
        # ==========================
        
        # Only update style transfer every N steps
        if self.step_count % self.update_style_every == 0:
            # Apply style transfer for unlabeled images
            fake_labeled = self.gen_u2l(unlabeled_images)
            
            # Identity mapping for labeled images
            identity_labeled = self.gen_u2l(labeled_images)
            
            # Discriminator outputs
            disc_real_labeled = self.disc_labeled(labeled_images)
            disc_fake_labeled = self.disc_labeled(fake_labeled.detach())
            
            # Feature consistency using teacher backbone
            with torch.no_grad():
                real_labeled_features = self.student.feature_extractor(labeled_images)
            
            fake_labeled_features = self.student.feature_extractor(fake_labeled)
            
            # Calculate style transfer losses
            
            # Generator adversarial loss
            gen_u2l_loss = self.adversarial_loss(
                self.disc_labeled(fake_labeled), target_is_real=True
            )
            
            # Discriminator loss
            disc_labeled_loss = 0.5 * (
                self.adversarial_loss(disc_real_labeled, target_is_real=True) +
                self.adversarial_loss(disc_fake_labeled, target_is_real=False)
            )
            
            # Identity mapping loss
            identity_loss = self.identity_loss(labeled_images, identity_labeled)
            
            # Feature consistency loss
            feature_loss = self.feature_consistency_loss(
                real_labeled_features, fake_labeled_features
            )
            
            # Combined style transfer loss
            style_loss = (
                gen_u2l_loss +  # Generator adversarial loss
                disc_labeled_loss +  # Discriminator loss
                self.identity_weight * identity_loss +  # Identity mapping
                self.feature_consistency_weight * feature_loss  # Feature consistency
            )
            
            # Scale by current style weight
            total_style_loss = style_weight * style_loss
            
            # Optimize style transfer models
            style_optimizer.zero_grad()
            self.manual_backward(total_style_loss)
            style_optimizer.step()
            
            # Log style transfer losses
            self.log("train_gen_u2l_loss", gen_u2l_loss)
            self.log("train_disc_labeled_loss", disc_labeled_loss)
            self.log("train_identity_loss", identity_loss)
            self.log("train_feature_loss", feature_loss)
            self.log("train_total_style_loss", total_style_loss, prog_bar=True)
            self.log("train_style_weight", style_weight)
    
    def focal_pseudo_loss(self, student_logits, teacher_logits, gamma=2.0, threshold=0.6):
        """
        Calculate focal pseudo-label loss for semi-supervised learning.
        
        Args:
            student_logits: Logits from student model
            teacher_logits: Logits from teacher model (for pseudo-labels)
            gamma: Focal loss gamma parameter
            threshold: Confidence threshold for pseudo-labels
            
        Returns:
            Focal loss with pseudo-labels
        """
        with torch.no_grad():
            teacher_probs = F.softmax(teacher_logits, dim=1)
            confidence, pseudo_labels = torch.max(teacher_probs, dim=1)
            mask = (confidence > threshold).float()
        
        # Get student probabilities
        student_probs = F.softmax(student_logits, dim=1)
        
        # Extract the predicted probability for the ground truth class
        batch_size, num_classes, h, w = student_logits.shape
        student_pt = torch.gather(student_probs, 1, pseudo_labels.unsqueeze(1)).squeeze(1)
        
        # Focal loss formula: -alpha * (1 - pt)^gamma * log(pt)
        focal_weight = (1 - student_pt).pow(gamma)
        
        # Standard cross entropy
        loss = F.cross_entropy(student_logits, pseudo_labels, reduction='none')
        
        # Apply focal weighting and confidence mask
        weighted_loss = focal_weight * loss * mask
        
        if mask.sum() > 0:
            return weighted_loss.sum() / mask.sum()
        return loss.mean()
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        """
        Update teacher model with EMA of student weights after training batch.
        """
        # Increment step counter
        self.step_count += 1
        
        # Update teacher model weights using EMA (every N steps)
        if self.step_count % self.update_teacher_every == 0:
            self.update_teacher()
    
    def on_train_epoch_start(self):
        """Update current epoch counter"""
        self._current_epoch = self.trainer.current_epoch
    
    def validation_step(self, batch, batch_idx):
        """
        Validation step using the teacher model.
        
        Teacher model is used for validation as it should perform better
        due to the ensembling effect of EMA updates.
        """
        images, masks = batch
        
        # Forward through teacher model
        with torch.no_grad():
            teacher_logits = self.teacher(images)
        
        # Calculate validation loss
        val_loss = F.cross_entropy(teacher_logits, masks)
        
        # Calculate metrics
        preds = torch.argmax(teacher_logits, dim=1)
        accuracy = (preds == masks).float().mean()
        
        # Calculate IoU for each class
        iou_per_class = []
        for cls in range(self.num_classes):
            intersection = ((preds == cls) & (masks == cls)).float().sum()
            union = ((preds == cls) | (masks == cls)).float().sum()
            iou = intersection / (union + 1e-6)
            iou_per_class.append(iou)
        
        mean_iou = torch.stack(iou_per_class).mean()
        
        # Log metrics
        self.log("val_loss", val_loss, prog_bar=True)
        self.log("val_accuracy", accuracy, prog_bar=True)
        self.log("val_mean_iou", mean_iou, prog_bar=True)
        
        return val_loss
    
    def configure_optimizers(self):
        """
        Configure optimizers with separate parameter groups and learning rates.
        Returns one optimizer for segmentation and one for style transfer.
        """
        # Segmentation optimizer (student model only)
        segmentation_params = self.student.parameters()
        
        # Style transfer parameters
        style_transfer_params = list(self.gen_u2l.parameters()) + \
                               list(self.disc_labeled.parameters())
        
        # Create optimizers
        segmentation_optimizer = torch.optim.AdamW(
            segmentation_params,
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        
        style_transfer_optimizer = torch.optim.AdamW(
            style_transfer_params,
            lr=self.style_transfer_lr,
            weight_decay=self.weight_decay * 0.1  # Lower weight decay for style transfer
        )
        
        # Calculate appropriate T_0 (first cycle length)
        T_0 = max(5, int(self.trainer.max_epochs * 0.2))  # At least 5 epochs
        
        # Create schedulers with warmup
        segmentation_scheduler = CosineAnnealingWarmRestartsDecay(
            segmentation_optimizer,
            T_0=T_0,
            T_mult=2,
            eta_min=self.lr * 0.01,
            warmup_epochs=5,
            warmup_start_lr=self.lr * 0.1,
            decay_factor=0.8
        )
        
        style_transfer_scheduler = CosineAnnealingWarmRestartsDecay(
            style_transfer_optimizer,
            T_0=T_0,
            T_mult=2,
            eta_min=self.style_transfer_lr * 0.01,
            warmup_epochs=3,  # Shorter warmup for style transfer
            warmup_start_lr=self.style_transfer_lr * 0.1,
            decay_factor=0.8
        )
        
        return [
            {"optimizer": segmentation_optimizer, 
             "lr_scheduler": {"scheduler": segmentation_scheduler, "interval": "epoch"}, 
             "frequency": 1},
            {"optimizer": style_transfer_optimizer, 
             "lr_scheduler": {"scheduler": style_transfer_scheduler, "interval": "epoch"}, 
             "frequency": 1}
        ]
    
    def test_step(self, batch, batch_idx):
        """
        Run test step on a single batch using the teacher model
        """
        images, masks = batch
        
        # Forward pass through teacher
        with torch.no_grad():
            logits = self.teacher(images)
        
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
    
    def visualize_style_transfer(self, unlabeled_images, labeled_images=None):
        """
        Visualize the results of style transfer from unlabeled to labeled domain.
        
        Args:
            unlabeled_images: Batch of unlabeled images
            labeled_images: Optional batch of labeled images for identity mapping visualization
            
        Returns:
            Dictionary of visualizations of original and styled images
        """
        self.eval()
        with torch.no_grad():
            # Generate style-transferred images (unlabeled→labeled)
            styled_unlabeled = self.gen_u2l(unlabeled_images)
            
            # If labeled images are provided, show identity mapping
            if labeled_images is not None:
                identity_mapped = self.gen_u2l(labeled_images)
                
                # Prepare visualization results
                results = {
                    'unlabeled_original': unlabeled_images.cpu(),
                    'unlabeled_styled': styled_unlabeled.cpu(),
                    'labeled_original': labeled_images.cpu(),
                    'labeled_identity': identity_mapped.cpu()
                }
            else:
                # Just show unlabeled and styled
                results = {
                    'unlabeled_original': unlabeled_images.cpu(),
                    'unlabeled_styled': styled_unlabeled.cpu()
                }
            
            return results
            
    def test(self, test_loader=None):
        """
        Comprehensive evaluation of segmentation model on test set
        
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