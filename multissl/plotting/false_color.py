import numpy as np
import torch 
import matplotlib.pyplot as plt

def create_false_color_image(tensor):

    """Convert a PyTorch tensor image to a NumPy array for visualization."""
    tensor = tensor.cpu().detach()  # Ensure it's detached from the computation graph
    tensor = tensor.permute(1, 2, 0)  # Convert from CxHxW to HxWxC
    image = tensor.numpy()
      # False Color Composite (NIR, RED, GREEN)
    false_color = np.stack([image[:, :, 3], image[:, :, 1], image[:, :, 0]], axis=-1)  # (H, W, 3)
    return false_color

def visualize_batch(batch, num_samples=4, class_names=None, num_classes=None):
    """Visualize a batch of images and their corresponding masks."""
    images, masks = batch
    
    if num_classes is None:
        num_classes = masks.max().item() + 1
    
    # Create a figure with two rows (images and masks)
    fig, axes = plt.subplots(2, num_samples, figsize=(15, 6))
    
    for i in range(num_samples):
        # Get current image and mask
        image = images[i]
        mask = masks[i]
        
        # Create false color image for visualization
        false_color = create_false_color_image(image)
        
        # Plot image
        axes[0, i].imshow(false_color)
        axes[0, i].set_title(f"Image {i}")
        axes[0, i].axis('off')
        
        # Plot mask with colormap
        if num_classes <= 10:
            cmap = plt.get_cmap('tab10', num_classes)
        else:
            cmap = plt.get_cmap('viridis', num_classes)
            
        im = axes[1, i].imshow(mask.cpu().numpy(), cmap=cmap, vmin=0, vmax=num_classes-1)
        axes[1, i].set_title(f"Mask {i}")
        axes[1, i].axis('off')
    
    # Add a colorbar if class names are provided
    if class_names:
        cbar = fig.colorbar(im, ax=axes[1,-1], shrink=0.7)
        cbar.set_ticks(np.arange(num_classes) + 0.5)
        cbar.set_ticklabels(class_names)
    
    plt.tight_layout()
    plt.show()

def visualize_predictions(model, dataloader, device, num_samples=4, class_names=None, num_classes=None):
    """Visualize model predictions on a batch of validation data."""
    # Get a batch from the dataloader
    batch = next(iter(dataloader))
    images, masks = batch
    images = images.to(device)
    
    # Make predictions
    with torch.no_grad():
        logits = model(images)
    
    # Get class predictions
    preds = torch.argmax(logits, dim=1)
    
    if num_classes is None:
        num_classes = masks.max().item() + 1
    
    # Create a figure with three rows (images, ground truth masks, predictions)
    fig, axes = plt.subplots(3, num_samples, figsize=(15, 9))
    
    for i in range(num_samples):
        # Get current image, mask, and prediction
        image = images[i]
        mask = masks[i]
        pred = preds[i]
        
        # Create false color image for visualization
        false_color = create_false_color_image(image)
        
        # Plot image
        axes[0, i].imshow(false_color)
        axes[0, i].set_title(f"Image {i}")
        axes[0, i].axis('off')
        
        # Plot ground truth mask
        if num_classes <= 10:
            cmap = plt.get_cmap('tab10', num_classes)
        else:
            cmap = plt.get_cmap('viridis', num_classes)
            
        axes[1, i].imshow(mask.cpu().numpy(), cmap=cmap, vmin=0, vmax=num_classes-1)
        axes[1, i].set_title(f"Ground Truth {i}")
        axes[1, i].axis('off')
        
        # Plot prediction
        im = axes[2, i].imshow(pred.cpu().numpy(), cmap=cmap, vmin=0, vmax=num_classes-1)
        axes[2, i].set_title(f"Prediction {i}")
        axes[2, i].axis('off')
    
    # Add a colorbar if class names are provided
    if class_names:
        cbar = fig.colorbar(im, ax=axes[1,-1], shrink=0.7)
        cbar.set_ticks(np.arange(num_classes) + 0.5)
        cbar.set_ticklabels(class_names)

    plt.show()