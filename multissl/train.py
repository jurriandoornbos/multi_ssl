# Copyright 2025 Jurrian Doornbos
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import argparse
from data import get_transform, tifffile_loader

from models import build_model
import pytorch_lightning as pl

import torch

from lightly.data import LightlyDataset

def get_args():
    parser = argparse.ArgumentParser(description="Self-Supervised Learning with Lightly AI")

    # Dataset and DataLoader
    parser.add_argument("--input_dir", type=str, required=True, help="Path to the dataset folder")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
    parser.add_argument("--input_size", type=int, default=224, help="Input image size")
    
    # Self-Supervised Learning Model
    parser.add_argument("--in_channels", type=int, default = 4, help = "Number of input channels of the image")
    parser.add_argument("--backbone", type=str, default="resnet18", choices=["resnet18", "resnet50"], help="Backbone model for SSL")
    parser.add_argument("--ssl_method", type=str, default="fastsiam", choices=["simclr", "simsiam", "fastsiam"], help="SSL method")
    parser.add_argument("--hidden_dim", type=int, default=512, help="Hidden layer dimension")
    parser.add_argument("--proj_dim", type=int, default=128, help="Projection head dimension")
    parser.add_argument("--pred_dim", type=int, default=64, help="Prediction head dimension")
    
    # Training Hyperparameters
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.125, help="Learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum for optimizer")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")

    # Augmentation Settings
    parser.add_argument("--gaussian_blur", action="store_true", help="Apply Gaussian blur in augmentation")
 
    # Miscellaneous
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device for training")

    args = parser.parse_args()
    return args

args = get_args()

# Create a multiview transform that returns two different augmentations of each image.
transform_multispectral = get_transform(args)

# Create a dataset from your image folder.
dataset_train_ms = LightlyDataset(
    input_dir = args.input_dir,
    transform = transform_multispectral,
)
dataset_train_ms.dataset.loader = tifffile_loader

# Build a PyTorch dataloader.
dataloader_train_ms = torch.utils.data.DataLoader(
    dataset_train_ms,                            # Pass the dataset to the dataloader.
    batch_size=args.batch_size,         # A large batch size helps with learning.
    shuffle=True,                       # Shuffling is important!
    num_workers=args.num_workers,
)

# scale the learning rate
lr = 0.05 * args.batch_size / args.input_size
args.lr = lr

# seed torch and numpy
s = args.seed

# Step 5: Load FastSiam Model with 4-channel support
model = build_model(args)
# Step 6: Checkpointer:

checkpoint_callback = pl.callbacks.ModelCheckpoint(
    dirpath="checkpoints/",
    filename="{args.ssl_method}-{epoch:02d}-{train_loss_ssl:.4f}",
    save_top_k=3,  # Save the 3 best models
    monitor="train_loss_ssl",
    mode="min",  # Save based on lowest loss
    save_last=True  # Also save the latest model
)
lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="step")

accelerator = "gpu" if torch.cuda.is_available() else "cpu"
# every 10 epochs we save
wandb_logger = pl.loggers.WandbLogger(project="FastSiam", log_model=True)

trainer = pl.Trainer(max_epochs=args.epochs, 
                     devices=1, 
                     accelerator=accelerator,
                     log_every_n_steps=10,
                    callbacks=[checkpoint_callback, lr_monitor],
                    logger = wandb_logger)

if __name__ == "__main__":
    trainer.fit(model=model, train_dataloaders=dataloader_train_ms)

