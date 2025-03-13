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
    
    model_type_name = model.__class__.__name__
    # Make predictions
    if model_type_name == "MeanTeacherSegmentation":
        print("Mean Teacher identified")
        with torch.no_grad():
            logits = model.teacher(images)
    else:
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

def visualize_batch_semi(batch, num_samples=4, class_names=None, num_classes=None, figsize=(15, 10)):
    """
    Visualize a semi-supervised batch containing both labeled and unlabeled data.
    
    Args:
        batch (tuple): Tuple of (labeled_batch, unlabeled_batch) where:
            - labeled_batch: Tuple of (images, masks) (may be empty tensors)
            - unlabeled_batch: Tensor of images (may be empty tensor)
        num_samples (int): Number of samples to visualize from each type
        class_names (list): List of class names for the mask visualization
        num_classes (int): Number of classes in the segmentation task
        figsize (tuple): Figure size
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    labeled_batch, unlabeled_batch = batch
    
    # Check if we have labeled data with non-empty tensors
    labeled_images, labeled_masks = labeled_batch
    has_labeled = labeled_images.size(0) > 0
    
    # Check if we have unlabeled data with non-empty tensors
    has_unlabeled = unlabeled_batch.size(0) > 0
    
    # Count samples of each type
    labeled_count = min(num_samples, labeled_images.size(0)) if has_labeled else 0
    unlabeled_count = min(num_samples, unlabeled_batch.size(0)) if has_unlabeled else 0
    
    # Determine number of rows in the plot
    num_rows = 0
    if has_labeled:
        num_rows += 2  # For image and mask
    if has_unlabeled:
        num_rows += 1  # For unlabeled images
    
    # Determine the total number of columns
    total_cols = max(labeled_count, unlabeled_count)
    
    # Create the figure and axes
    fig, axes = plt.subplots(num_rows, total_cols, figsize=figsize)
    
    # Convert to 2D array of axes if not already
    if num_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Keep track of the current row
    current_row = 0
    
    # Plot labeled samples if available
    if has_labeled:
        images, masks = labeled_batch
        
        if num_classes is None:
            num_classes = masks.max().item() + 1
        
        # Choose a colormap based on number of classes
        if num_classes <= 10:
            cmap = plt.get_cmap('tab10', num_classes)
        else:
            cmap = plt.get_cmap('viridis', num_classes)
        
        for i in range(labeled_count):
            # Get current image and mask
            image = images[i]
            mask = masks[i]
            
            # Create false color image for visualization
            false_color = create_false_color_image(image)
            
            # Plot image
            axes[current_row, i].imshow(false_color)
            axes[current_row, i].set_title(f"Labeled Image {i}")
            axes[current_row, i].axis('off')
            
            # Plot mask
            im = axes[current_row + 1, i].imshow(mask.cpu().numpy(), cmap=cmap, vmin=0, vmax=num_classes-1)
            axes[current_row + 1, i].set_title(f"Ground Truth {i}")
            axes[current_row + 1, i].axis('off')
        
        # Fill in empty plots in the labeled rows if needed
        for i in range(labeled_count, total_cols):
            axes[current_row, i].axis('off')
            axes[current_row + 1, i].axis('off')
        
        # Add a colorbar for masks if class names are provided
        if class_names and labeled_count > 0:
            cbar_ax = fig.add_axes([0.92, 0.6, 0.02, 0.3])  # [left, bottom, width, height]
            cbar = fig.colorbar(im, cax=cbar_ax)
            cbar.set_ticks(np.arange(num_classes) + 0.5)
            cbar.set_ticklabels(class_names)
        
        # Update current row
        current_row += 2
    
    # Plot unlabeled samples if available
    if has_unlabeled:
        for i in range(unlabeled_count):
            # Get current image
            image = unlabeled_batch[i]
            
            # Create false color image for visualization
            false_color = create_false_color_image(image)
            
            # Plot image
            axes[current_row, i].imshow(false_color)
            axes[current_row, i].set_title(f"Unlabeled Image {i}")
            axes[current_row, i].axis('off')
        
        # Fill in empty plots in the unlabeled row if needed
        for i in range(unlabeled_count, total_cols):
            axes[current_row, i].axis('off')
    
    plt.tight_layout()
    plt.subplots_adjust(right=0.9)  # Make room for colorbar
    
    return fig