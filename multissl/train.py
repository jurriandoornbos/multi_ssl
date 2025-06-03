# Copyright 2025 Jurrian Doornbos
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import argparse
from data import get_transform, tifffile_loader, MixedUAVDataset,  multisensor_views_collate_fn, BalancedSampler

from models import build_model
import pytorch_lightning as pl
from plots import ImageSaverCallback
import torch

from lightly.data import LightlyDataset
from lightly.transforms.multi_view_transform import MultiViewTransform
import pytorch_lightning as pl
import torch

def get_args():
    parser = argparse.ArgumentParser(description="Self-Supervised Learning with Lightly AI")

    # Dataset and DataLoader
    parser.add_argument("--input_dir", type=str, required=True, help="Path to the dataset folder")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
    parser.add_argument("--input_size", type=int, default=224, help="Input image size")
    
    # Self-Supervised Learning Model
    parser.add_argument("--in_channels", type=int, default = 4, help = "Number of input channels of the image")
    parser.add_argument("--backbone", type=str, default="resnet18", choices=["resnet18", "resnet50", "vit-s", "swin-tiny", "pasiphae"], help="Backbone model for SSL")
    parser.add_argument("--ssl_method", type=str, default="fastsiam", choices=["fastsiam", "galileo-fastsiam"], help="SSL method")
    parser.add_argument("--hidden_dim", type=int, default=2048, help="Hidden layer dimension")
    parser.add_argument("--proj_dim", type=int, default=256, help="Projection head dimension")
    parser.add_argument("--pred_dim", type=int, default=128, help="Prediction head dimension")
    

    # Training Hyperparameters
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum for optimizer")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--dataset_size", type=int, default = 1_000_000, help = "Dataset size, overridden after dataloading" )
    parser.add_argument("--num_views", type = int, default = 4, help = "Number of augmentation views to feed the SSL, FastSIAM found 3 target + 1 base to be best")

 
    # Miscellaneous
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device for training")
    parser.add_argument("--save_every", type=int, default=1000, help="How many stepss between save")
    parser.add_argument("--checkpoint_path", type=str, default=None, help = "Path for resuming from checkpoint")
    parser.add_argument("--dataset_reduce", type = int, default=None, help = "How many samples in the dataset for testing" )
    parser.add_argument("--smote", type=bool, default=False, help = "Use Synthetic Minority Oversampling")
    
    args = parser.parse_args()
    return args

args = get_args()

def main():
        # Create a multiview transform that returns three different augmentations of each image.
    transform_multispectral = get_transform(args, std_noise  =0.1, 
                                            brightness_factor=0.1 ,
                                            max_shift=0.1)
    tfs = [transform_multispectral for i in range(args.num_views)]

    transform_ms = MultiViewTransform(transforms=tfs)
    sampler = None

    pl.seed_everything(args.seed)

    if args.backbone == "pasiphae":
        batch_size = args.batch_size
        dataset_train_ms = MixedUAVDataset(
            root_dir = args.input_dir,
            transform = transform_ms,
            reduce_by = args.dataset_reduce
        )

        length_dataset = len(dataset_train_ms)
        print("Loaded dataset, dataset size: "+ str(length_dataset))
        args.dataset_size = length_dataset
        
        if args.smote:
            # Calculate appropriate oversampling factors based on dataset statistics
            rgb_count = dataset_train_ms.count_by_type['rgb_only']
            ms_count = dataset_train_ms.count_by_type['ms_only']
            aligned_count = dataset_train_ms.count_by_type['aligned']
            
            # Create balanced sampler
            sampler = BalancedSampler(
                dataset_train_ms, 
                batch_size=batch_size,
                oversample_factor_ms=4.5,
                oversample_factor_aligned=1,
                undersample_rgb=True,
                undersample_factor = 0.2,
                balance_mode="both"
            )
    

            # Get effective counts after balancing
            effective_counts = sampler.get_effective_counts()
            
            # Calculate dataset size per epoch
            epoch_size = effective_counts['total']
            
            # Ensure epoch size is divisible by batch size
            batches = epoch_size // batch_size
            if batches == 0:
                batches = 1
            epoch_size = batches * batch_size

            length_dataset = epoch_size
            print("Loaded dataset per Epoch after SMOTE, dataset size: "+ str(epoch_size))
                

        if sampler:
                
            dataloader_train_ms = torch.utils.data.DataLoader(
                dataset_train_ms,                            # Pass the dataset to the dataloader.
                batch_size=args.batch_size,         # A large batch size helps with learning.
                drop_last = True,
                sampler=sampler,
                num_workers=args.num_workers,
                collate_fn= multisensor_views_collate_fn
            )
        else: 
            dataloader_train_ms = torch.utils.data.DataLoader(
                dataset_train_ms,                            # Pass the dataset to the dataloader.
                batch_size=args.batch_size,         # A large batch size helps with learning.
                shuffle=True,                       # Shuffling is important!
                drop_last = True,
                sampler=sampler,
                num_workers=args.num_workers,
                collate_fn= multisensor_views_collate_fn
            )

    else:

        # Create a dataset from your image folder.
        dataset_train_ms = LightlyDataset(
            input_dir = args.input_dir,
            transform = transform_ms,
        )
        dataset_train_ms.dataset.loader = tifffile_loader

        length_dataset = len(dataset_train_ms)
        print("Loaded dataset, dataset size: "+ str(length_dataset))
        args.dataset_size = length_dataset

            
        # Build a PyTorch dataloader.
        dataloader_train_ms = torch.utils.data.DataLoader(
            dataset_train_ms,                            # Pass the dataset to the dataloader.
            batch_size=args.batch_size,         # A large batch size helps with learning.
            shuffle=True,                       # Shuffling is important!
            drop_last = True,
            num_workers=args.num_workers,

        )


    # Step 5: Load FastSiam Model with 4-channel support

    model = build_model(args)
    # Step 6: Checkpointer:

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath="checkpoints/",
        filename="{fs}-{epoch:02d}-{step:02d}-{train_loss_ssl:.4f}",
        save_top_k=3,  # Save the 3 best models
        monitor="train_loss_ssl",
        mode="min",  # Save based on lowest loss
        save_last=True,  # Also save the latest model
        every_n_train_steps = args.save_every
    )
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval = "step")
    if args.backbone == "pasiphae":
        from multissl.plots_pasiphae import ImageSaverCallback
        im_monitor = ImageSaverCallback(args =args)
    else:
        im_monitor = ImageSaverCallback(args =args)

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"

    wandb_logger = pl.loggers.WandbLogger(project="FastSiamUAV", log_model=True)


    #check if everythign went okay
    #plot_first_batch(dataloader_train_ms)

    trainer = pl.Trainer(max_epochs=args.epochs, 
                        devices=1, 
                        accelerator=accelerator,
                        callbacks=[lr_monitor, im_monitor,checkpoint_callback],
                        logger = wandb_logger,
                        strategy = "auto",
                        log_every_n_steps= 1 )
    

    trainer.fit(model=model, train_dataloaders=dataloader_train_ms, ckpt_path=args.checkpoint_path)
    


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high") 
    main()