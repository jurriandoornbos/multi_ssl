import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

def rgb_visualize_batch(batch, nrow=8, padding=2, normalize=False, title=None, figsize=(12, 12)):
    """
    Visualize a batch of RGB images.
    
    Args:
        batch (torch.Tensor): Batch of images with shape (B, C, H, W)
        nrow (int): Number of images in each row of the grid
        padding (int): Amount of padding between images
        normalize (bool): If True, shift the image to the range (0, 1),
                         by the min and max values of the batch
        title (str): Title for the plot
        figsize (tuple): Figure size (width, height) in inches
        
    Returns:
        None (displays the plot)
    """
    # Check if batch is a torch tensor
    if not isinstance(batch, torch.Tensor):
        raise TypeError("Batch should be a torch.Tensor")
    
    # Check if it's a batch of RGB images
    if len(batch.shape) != 4:
        raise ValueError(f"Expected 4D tensor (B, C, H, W), got {batch.shape}")
    
    # Make sure we're dealing with RGB images (3 channels) or grayscale (1 channel)
    if batch.shape[1] not in [1, 3]:
        raise ValueError(f"Expected RGB (3 channels) or grayscale (1 channel), got {batch.shape[1]} channels")

    # Create a grid of images
    grid = make_grid(batch, nrow=nrow, padding=padding, normalize=normalize)
    
    # Convert to numpy and transpose from (C, H, W) to (H, W, C)
    grid_np = grid.cpu().numpy().transpose((1, 2, 0))
    
    # If single channel, repeat to make it RGB
    if batch.shape[1] == 1:
        grid_np = np.repeat(grid_np, 3, axis=2)
    
    # Create figure and display
    plt.figure(figsize=figsize)
    
    # Add title if provided
    if title:
        plt.title(title, fontsize=16)
    
    # Turn off axes
    plt.axis('off')
    
    # Plot the grid
    plt.imshow(grid_np)
    plt.tight_layout()
    plt.show()

# Example usage:
# batch = torch.randn(16, 3, 224, 224)  # Batch of 16 random RGB images
# visualize_batch(batch, nrow=4, normalize=True, title="Random RGB Images")