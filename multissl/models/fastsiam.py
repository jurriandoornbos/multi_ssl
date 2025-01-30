import pytorch_lightning as pl
import torch
import torchvision
from torch import nn
from lightly.loss import NegativeCosineSimilarity
from lightly.models.modules import SimSiamPredictionHead, SimSiamProjectionHead
from torch.optim.lr_scheduler import CosineAnnealingLR

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
                 weight_decay = 1e-4,):
        super().__init__()

        # Load backbone dynamically based on user input
        if backbone == "resnet18":
            resnet = torchvision.models.resnet18(weights=None)
            backbone_dim = 512
        elif backbone == "resnet50":
            resnet = torchvision.models.resnet50(weights=None)
            backbone_dim = 2048
        else:
            raise ValueError("Unsupported backbone. Choose resnet18 or resnet50.")

        # Modify first convolution layer to accept 4-channel input instead of 3
        self._modify_resnet_for_4_channels(resnet, in_channels)

        # Remove final FC layer and store the feature extractor
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        # Projection and Prediction heads
        self.projection_head = SimSiamProjectionHead(backbone_dim, hidden_dim, proj_dim)
        self.prediction_head = SimSiamPredictionHead(proj_dim, pred_dim, proj_dim)

        # Loss function
        self.criterion = NegativeCosineSimilarity()

        # Learning rate
        # Base LR for batch_size=32
        self.lr = lr
        # Scale linearly for larger batch sizes
        self.scaled_lr = self.lr * (batch_size / 32.0)
        self.max_epochs = epochs

        self.momentum = momentum
        self.weight_decay = weight_decay

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
        f = self.backbone(x).flatten(start_dim=1)
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
        optim = torch.optim.SGD(self.parameters(), 
                                lr=self.scaled_lr, 
                                momentum=self.momentum, 
                                weight_decay=self.weight_decay)
        scheduler = CosineAnnealingLR(
            optim,
            T_max=self.max_epochs,    # typically #epochs or #iterations
            eta_min=0.0              # minimum LR after decay
        )
        return [optim], [scheduler]