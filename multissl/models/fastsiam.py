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
            backbone_dim = 512
            self._modify_resnet_for_4_channels(resnet, in_channels)

            # Remove final FC layer and store the feature extractor
            modules = list(resnet.children())[:-1]
            modules.append(Flatten())
            self.backbone = nn.Sequential(*modules)


        elif backbone == "resnet50":
            resnet = torchvision.models.resnet50(weights=None)
            backbone_dim = 2048
            self._modify_resnet_for_4_channels(resnet, in_channels)

            # Remove final FC layer and store the feature extractor
            modules = list(resnet.children())[:-1]
            modules.append(Flatten())
            self.backbone = nn.Sequential(*modules)

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
        else:
            raise ValueError("Unsupported backbone. Choose resnet18 or resnet50.")


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
        f = self.backbone(x)
        z = self.projection_head(f)
        p = self.prediction_head(z)
        z = z.detach()
        return z, p

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