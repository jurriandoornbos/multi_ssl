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

        elif backbone == "resnet50":
            resnet = torchvision.models.resnet50(weights=None)
            self._modify_resnet_for_4_channels(resnet, in_channels)
            self.feature_extractor = ResNetFeatureExtractor(resnet)
            backbone_dim = 2048  # ResNet50 final output dimension
            self.using_vit= False

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

    def training_step(self, batch, batch_idx):
        """Compute the SSL loss"""
        views = batch[0]  # Three from the batch is all the views
        features = [self.forward(view) for view in views]
        zs = torch.stack([z for z, _ in features])
        ps = torch.stack([p for _, p in features])

        loss = 0.0
        for i in range(len(views)):
            mask = torch.arange(len(views), device=self.device) != i
            loss += self.criterion(ps[i], torch.mean(zs[mask], dim=0)) / len(views)

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