# Copyright 2025 Jurrian Doornbos
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import argparse
import os
import torch
import pytorch_lightning as pl
import torch.nn as nn
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from timm import create_model
from segment_head import ViTSegmentationModel, SegmentationHead

def clean_state_dict(
    state_dict,
    prefix_to_strip="backbone.",
    skip_if_contains=None
):
    """
    Create a new state_dict with:
      1) The specified prefix stripped from the start of each key.
      2) Certain keys *entirely removed* if they match a skip pattern.

    :param state_dict: (dict) The original state_dict.
    :param prefix_to_strip: (str) If a key starts with this prefix, remove that prefix.
    :param skip_if_contains: (list of str or None)
            If not None, any key containing ANY of these substrings is skipped entirely.
    :return: (dict) A new state_dict with modified/filtered keys.
    """
    if skip_if_contains is None:
        skip_if_contains = []

    new_state_dict = {}
    for key, value in state_dict.items():
        # 1) Check if we should skip this key entirely
        #    (if the key contains any of the "skip" substrings)
        if any(skip_str in key for skip_str in skip_if_contains):
            continue

        # 2) Strip prefix if present
        if key.startswith(prefix_to_strip):
            new_key = key[len(prefix_to_strip):]
        else:
            new_key = key

        new_state_dict[new_key] = value

    return new_state_dict

def load_pretrained_vit(checkpoint_path, in_channels=4):
    """
    Load a pretrained ViT model from a FastSiam checkpoint.
    
    Args:
        checkpoint_path: Path to the FastSiam checkpoint
        in_channels: Number of input channels
        
    Returns:
        Pretrained ViT backbone
    """
    # Load the checkpoint state dict
    state_dict = torch.load(checkpoint_path)["state_dict"]
    
    # Clean the state dict to only keep backbone parameters and remove heads
    cleaned_state_dict = clean_state_dict(
        state_dict, 
        prefix_to_strip="backbone.", 
        skip_if_contains=["prediction_head", "projection_head"]
    )
    
    # Create a new ViT model
    vit = create_model(
        "vit_small_patch16_224",
        pretrained=False,
        in_chans=in_channels,
        num_classes=0  # Set to 0 to get features instead of logits
    )
    
    # Load the pretrained weights
    vit.load_state_dict(cleaned_state_dict)
    
    return vit

# Define a segmentation dataset class
class SegmentationDataset(torch.utils.data.Dataset):
    """
    Dataset for semantic segmentation.
    
    This is a placeholder - you'll need to implement this based on your specific data format.
    """
    def __init__(self, img_dir, mask_dir, transform=None, mask_transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.mask_transform = mask_transform
        
        # List all files in the image directory
        self.img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(".tif") or f.endswith(".tiff")])
        
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        # Get image file name
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        
        # Get corresponding mask file name (adjust this according to your naming convention)
        mask_name = img_name.replace(".tif", "_mask.tif").replace(".tiff", "_mask.tiff")
        mask_path = os.path.join(self.mask_dir, mask_name)
        
        # Load image and mask using tifffile_loader
        from multissl.data import tifffile_loader
        from multissl.data.transforms import UIntToFloat, Transpose, ToTensor
        from torchvision import transforms
        
        image = tifffile_loader(img_path)
        mask = tifffile_loader(mask_path)
        
        # Apply transforms if provided
        if self.transform:
            image = self.transform(image)
        else:
            # Default minimal processing
            image = UIntToFloat()(image)
            image = Transpose()(image)
            image = ToTensor()(image)
            
        if self.mask_transform:
            mask = self.mask_transform(mask)
        else:
            # Default mask processing - adjust based on your mask format
            # Assuming mask is an integer mask with class values
            mask = torch.from_numpy(mask).long()
            
        return image, mask

def get_args():
    parser = argparse.ArgumentParser(description="Training script for ViT-based semantic segmentation")
    
    # Dataset parameters
    parser.add_argument("--img_dir", type=str, required=True, help="Directory with input images")
    parser.add_argument("--mask_dir", type=str, required=True, help="Directory with segmentation masks")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
    
    # Model parameters
    parser.add_argument("--pretrained_checkpoint", type=str, required=True, help="Path to pretrained ViT model")
    parser.add_argument("--num_classes", type=int, required=True, help="Number of segmentation classes")
    parser.add_argument("--in_channels", type=int, default=4, help="Number of input channels")
    parser.add_argument("--img_size", type=int, default=224, help="Input image size")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension of segmentation head")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--unfreeze_backbone", action="store_true", help="Unfreeze backbone after initial training")
    parser.add_argument("--unfreeze_after", type=int, default=10, help="Unfreeze backbone after this many epochs")
    
    # Misc parameters
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--checkpoint_dir", type=str, default="segmentation_checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--project_name", type=str, default="ViT-Segmentation", help="Project name for logging")
    
    args = parser.parse_args()
    return args

def main():
    # Parse arguments
    args = get_args()
    
    # Set random seed
    pl.seed_everything(args.seed)
    
    # Load pretrained ViT backbone
    vit_backbone = load_pretrained_vit(args.pretrained_checkpoint, args.in_channels)
    
    # Create segmentation model
    model = ViTSegmentationModel(
        backbone=vit_backbone,
        num_classes=args.num_classes,
        img_size=args.img_size,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Create datasets and dataloaders
    # Implement transforms as needed based on your specific requirements
    train_dataset = SegmentationDataset(
        img_dir=os.path.join(args.img_dir, 'train'),
        mask_dir=os.path.join(args.mask_dir, 'train'),
    )
    
    val_dataset = SegmentationDataset(
        img_dir=os.path.join(args.img_dir, 'val'),
        mask_dir=os.path.join(args.mask_dir, 'val'),
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Create callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        filename="{epoch:02d}-{val_loss:.4f}",
        save_top_k=3,
        monitor="val_loss",
        mode="min",
        save_last=True
    )
    
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    
    # Create a callback to unfreeze backbone after specified epochs
    class UnfreezeBackboneCallback(pl.Callback):
        def __init__(self, unfreeze_after=10):
            super().__init__()
            self.unfreeze_after = unfreeze_after
            
        def on_epoch_end(self, trainer, pl_module):
            if trainer.current_epoch == self.unfreeze_after:
                print(f"Unfreezing backbone at epoch {trainer.current_epoch}")
                pl_module.unfreeze_backbone()
    
    callbacks = [checkpoint_callback, lr_monitor]
    if args.unfreeze_backbone:
        callbacks.append(UnfreezeBackboneCallback(args.unfreeze_after))
    
    # Create logger
    logger = WandbLogger(project=args.project_name, log_model=True)
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        devices=1,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=10
    )
    
    # Train the model
    trainer.fit(model, train_loader, val_loader)
    
    # Save the model
    trainer.save_checkpoint(os.path.join(args.checkpoint_dir, "final_model.ckpt"))

if __name__ == "__main__":
    main()
