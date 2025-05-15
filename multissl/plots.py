# Copyright 2025 Jurrian Doornbos
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0


import matplotlib.pyplot as plt
import torch
import numpy as np
# Load dataset (update path as needed)
import os
import pytorch_lightning as pl


def plot_pasiphae(batch, output_dir):
    return

    img_view = []

    if batch['rgb_only']['data']:
        num_views = len(batch['rgb_only']['data'])
        for view_idx in range(num_views):
            # Get the current view for all samples
            rgb_view = batch['rgb_only']['data'][view_idx][0]
            img_view.append({view_idx: tensor_to_image(rgb_view)})


    if batch['ms_only']['data']:
        num_views = len(batch['ms_only']['data'])
        for view_idx in range(num_views):
            # Get the current view for all samples
            rgb_view = batch['ms_only']['data'][view_idx][0]
            img_view.append({view_idx: tensor_to_image(rgb_view)})

    
    # 3. Aligned RGB+MS samples
    if batch['aligned']['rgb']:
        num_views = len(batch['aligned']['rgb'])

        for view_idx in range(num_views):
            # Get the current view for all samples
            rgb_view = batch['ms_only']['data'][view_idx][0]
            img_view.append({view_idx: tensor_to_image(rgb_view)})


    # 4 imgs, 16 batch, 5 channels, 224 height, 224 width:
        
    fig, axes = plt.subplots(3, 2, figsize=(9, 6))
    t=2

    for i in range(2):
        img_view0 = img_view[i]
        img_view1 = img_view[i]
        img_view2 = img_view[i]

        # Row 0: First view
        axes[0, i].imshow(img_view0)
        axes[0, i].axis("off")

        # Row 1: Second view (transformed version)
        axes[1, i].imshow(img_view1)
        axes[1, i].axis("off")

        # Row 2: Third view (transformed version)
        axes[2, i].imshow(img_view2)
        axes[2, i].axis("off")

    axes[0, 0].set_title("View 0 (Aug); {} views used".format(t))
    axes[1, 0].set_title("View 1 target (Aug)")
    axes[2, 0].set_title("View 2 target (Aug)")



    # Save figure
    save_path = os.path.join(output_dir, "lightly_multiview_batch.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()

# Convert tensor images for plotting
def tensor_to_image(tensor):
    """Convert a PyTorch tensor image to a NumPy array for visualization."""
    tensor = tensor.cpu().detach()  # Ensure it's detached from the computation graph
    tensor = tensor.permute(1, 2, 0)  # Convert from CxHxW to HxWxC
    image = tensor.numpy()
      # False Color Composite (NIR, RED, GREEN)
    false_color = np.stack([image[:, :, 2], image[:, :, 1], image[:, :, 0]], axis=-1)  # (H, W, 3)
    return false_color

def plot_first_batch(batch, output_dir,args = None):
    # Get first batch
    if args.backbone == "pasiphae":

        plot_pasiphae(batch, output_dir)


    else:
        views, targets, filenames = batch
    
        if len(views) == 3:
            view0, view1, view2 = views  # Unpacking two views from MultiViewCollate
            t = 3
        elif len(views) > 3:
            view0, view1, view2, _ = views  # Unpacking two views from MultiViewCollate
            t = len(views)
        # Plot 4 samples with two views each (3 rows, 4 columns)
        fig, axes = plt.subplots(3, t, figsize=(12, 6))

        for i in range(t):
            img_view0 = tensor_to_image(view0[i])
            img_view1 = tensor_to_image(view1[i])
            img_view2 = tensor_to_image(view2[i])

            # Row 0: First view
            axes[0, i].imshow(img_view0)
            axes[0, i].axis("off")

            # Row 1: Second view (transformed version)
            axes[1, i].imshow(img_view1)
            axes[1, i].axis("off")

            # Row 2: Third view (transformed version)
            axes[2, i].imshow(img_view2)
            axes[2, i].axis("off")

        axes[0, 0].set_title("View 0 (Aug); {} views used".format(t))
        axes[1, 0].set_title("View 1 target (Aug)")
        axes[2, 0].set_title("View 2 target (Aug)")

        # Save figure
        save_path = os.path.join(output_dir, "lightly_multiview_batch.png")
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.close()

class ImageSaverCallback(pl.Callback):
    def __init__(self, output_dir="saved_images", every_n_steps=100, args =None):
        """
        Args:
            output_dir (str): Directory to save images.
            every_n_epochs (int): Save images every n epochs.
        """
        super().__init__()
        self.output_dir = output_dir
        self.every_n_steps = every_n_steps
        self.args = args
        os.makedirs(self.output_dir, exist_ok=True)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Save images from the batch at the end of each epoch."""
        """Save images every n training steps."""
        
       
        # Run for batch 0 or every n steps
        if batch_idx == 0 or trainer.global_step % self.every_n_steps == 0:
            # Assuming batch contains (views, targets, filenames)
            plot_first_batch(batch, self.output_dir, args= self.args)
            if trainer.logger:
                trainer.logger.log_image(key="Batch viz", images=[os.path.join(self.output_dir, "lightly_multiview_batch.png")])