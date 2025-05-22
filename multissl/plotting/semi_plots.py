import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import seaborn as sns
from pathlib import Path
import wandb
import pytorch_lightning as pl

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")


def plot_mixed_supervision_validation(
    images: torch.Tensor,
    targets: torch.Tensor,
    predictions: torch.Tensor,
    supervision_info: Dict[str, Any],
    class_names: Optional[List[str]] = None,
    ignore_index: int = 255,
    max_samples: int = 6,
    save_path: Optional[str] = None,
    logger: Optional[Any] = None,
    global_step: int = 0,
    colormap: str = 'tab10'
) -> plt.Figure:
    """
    Create comprehensive visualization for mixed supervision validation.
    
    Args:
        images: Input images [B, C, H, W]
        targets: Ground truth masks [B, H, W]
        predictions: Model predictions [B, C, H, W] (logits)
        supervision_info: Dictionary with supervision type information
        class_names: List of class names for legend
        ignore_index: Index for unlabeled pixels
        max_samples: Maximum number of samples to plot
        save_path: Optional path to save the plot
        logger: Optional logger (wandb, tensorboard, etc.)
        global_step: Current training step
        colormap: Matplotlib colormap for classes
        
    Returns:
        matplotlib Figure object
    """
    device = images.device
    batch_size = min(images.shape[0], max_samples)
    
    # Convert predictions to class predictions
    pred_masks = torch.argmax(predictions, dim=1)
    pred_probs = torch.softmax(predictions, dim=1)
    
    # Move to CPU for plotting
    images_cpu = images[:batch_size].cpu()
    targets_cpu = targets[:batch_size].cpu()
    pred_masks_cpu = pred_masks[:batch_size].cpu()
    pred_probs_cpu = pred_probs[:batch_size].cpu()
    
    # Get supervision types
    supervision_types = supervision_info.get('supervision_types', ['unknown'] * batch_size)
    full_indices = set(supervision_info.get('full_indices', []))
    partial_indices = set(supervision_info.get('partial_indices', []))
    
    # Set up the plot
    n_cols = 5  # Image, GT, Prediction, Confidence, Difference
    n_rows = batch_size
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Define colors for different classes
    colors = plt.cm.get_cmap(colormap)
    num_classes = predictions.shape[1]
    
    # Set up class names
    if class_names is None:
        class_names = [f'Class {i}' for i in range(num_classes)]
    
    for row in range(batch_size):
        img = images_cpu[row]
        target = targets_cpu[row]
        pred_mask = pred_masks_cpu[row]
        pred_prob = pred_probs_cpu[row]
        
        # Determine supervision type for this sample
        is_full = row in full_indices
        is_partial = row in partial_indices
        supervision_type = 'Full' if is_full else 'Partial' if is_partial else 'Unknown'
        
        # Prepare image for display
        if img.shape[0] == 3:  # RGB
            display_img = img.permute(1, 2, 0).numpy()
            display_img = np.clip(display_img, 0, 1)
        elif img.shape[0] > 3:  # Multispectral - create false color
            display_img = create_false_color_composite(img)
        else:  # Grayscale
            display_img = img[0].numpy()
        
        # 1. Original Image
        axes[row, 0].imshow(display_img, cmap='gray' if len(display_img.shape) == 2 else None)
        axes[row, 0].set_title(f'Image {row}\n({supervision_type} Supervision)', fontsize=10)
        axes[row, 0].axis('off')
        
        # 2. Ground Truth
        gt_colored = create_colored_mask(target.numpy(), colors, num_classes, ignore_index)
        axes[row, 1].imshow(gt_colored)
        axes[row, 1].set_title('Ground Truth', fontsize=10)
        axes[row, 1].axis('off')
        
        # Add indicators for unlabeled regions in partial supervision
        if is_partial:
            unlabeled_mask = (target == ignore_index)
            if unlabeled_mask.any():
                # Overlay unlabeled regions with hatching
                axes[row, 1].contour(unlabeled_mask.numpy(), levels=[0.5], colors=['red'], 
                                   linewidths=1, linestyles='--', alpha=0.7)
        
        # 3. Prediction
        pred_colored = create_colored_mask(pred_mask.numpy(), colors, num_classes, ignore_index)
        axes[row, 2].imshow(pred_colored)
        axes[row, 2].set_title('Prediction', fontsize=10)
        axes[row, 2].axis('off')
        
        # 4. Prediction Confidence
        # Show confidence as max probability across classes
        max_confidence = torch.max(pred_prob, dim=0)[0]
        confidence_plot = axes[row, 3].imshow(max_confidence.numpy(), cmap='viridis', vmin=0, vmax=1)
        axes[row, 3].set_title('Prediction Confidence', fontsize=10)
        axes[row, 3].axis('off')
        
        # Add colorbar for confidence
        cbar = plt.colorbar(confidence_plot, ax=axes[row, 3], fraction=0.046, pad=0.04)
        cbar.set_label('Confidence', fontsize=8)
        
        # 5. Prediction vs Ground Truth Difference
        difference_map = create_difference_map(
            target.numpy(), pred_mask.numpy(), ignore_index
        )
        diff_plot = axes[row, 4].imshow(difference_map, cmap='RdYlGn_r', vmin=0, vmax=2)
        axes[row, 4].set_title('Difference Map', fontsize=10)
        axes[row, 4].axis('off')
        
        # Add text annotations for metrics
        accuracy = compute_pixel_accuracy(target, pred_mask, ignore_index)
        iou = compute_mean_iou(target, pred_mask, num_classes, ignore_index)
        
        axes[row, 4].text(0.02, 0.98, f'Acc: {accuracy:.3f}\nIoU: {iou:.3f}', 
                         transform=axes[row, 4].transAxes, fontsize=8,
                         verticalalignment='top', bbox=dict(boxstyle='round', 
                         facecolor='wheat', alpha=0.8))
    
    # Create legend
    legend_elements = []
    for i, class_name in enumerate(class_names):
        color = colors(i / (num_classes - 1)) if num_classes > 1 else colors(0)
        legend_elements.append(mpatches.Patch(color=color, label=class_name))
    
    # Add special patches for supervision types
    legend_elements.append(mpatches.Patch(color='none', label=''))  # Spacer
    legend_elements.append(plt.Line2D([0], [0], color='red', linestyle='--', 
                                     label='Unlabeled regions'))
    
    # Add difference map legend
    legend_elements.append(mpatches.Patch(color='none', label=''))  # Spacer
    legend_elements.append(mpatches.Patch(color='darkgreen', label='Correct'))
    legend_elements.append(mpatches.Patch(color='red', label='Incorrect'))
    legend_elements.append(mpatches.Patch(color='gray', label='Ignored'))
    
    fig.legend(handles=legend_elements, loc='center', bbox_to_anchor=(0.5, 0.02), 
               ncol=min(len(legend_elements), 6), fontsize=10)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    
    # Save if path provided
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Validation plot saved to: {save_path}")
    
    # Log to wandb if available
    if logger and hasattr(logger, 'log_image'):
        logger.log_image(key="validation_predictions", images=[fig], step=global_step)
    
    return fig


def create_false_color_composite(multispectral_img: torch.Tensor) -> np.ndarray:
    """
    Create false color composite from multispectral image.
    Assumes channels are ordered as [R, G, B, NIR, ...] or similar.
    """
    img_np = multispectral_img.numpy()
    
    if img_np.shape[0] >= 4:
        # NIR-R-G composite (common for vegetation analysis)
        false_color = np.stack([
            img_np[3],  # NIR -> Red channel
            img_np[0],  # Red -> Green channel  
            img_np[1],  # Green -> Blue channel
        ], axis=-1)
    elif img_np.shape[0] >= 3:
        # Standard RGB
        false_color = img_np[:3].transpose(1, 2, 0)
    else:
        # Grayscale to RGB
        false_color = np.stack([img_np[0]] * 3, axis=-1)
    
    # Normalize to [0, 1]
    false_color = (false_color - false_color.min()) / (false_color.max() - false_color.min() + 1e-8)
    return np.clip(false_color, 0, 1)


def create_colored_mask(mask: np.ndarray, colormap, num_classes: int, ignore_index: int) -> np.ndarray:
    """Create colored visualization of segmentation mask"""
    colored_mask = np.zeros((*mask.shape, 3))
    
    for class_id in range(num_classes):
        class_mask = (mask == class_id)
        if class_mask.any():
            color = colormap(class_id / max(1, num_classes - 1))[:3]
            colored_mask[class_mask] = color
    
    # Handle ignore index with gray color
    ignore_mask = (mask == ignore_index)
    if ignore_mask.any():
        colored_mask[ignore_mask] = [0.5, 0.5, 0.5]  # Gray
    
    return colored_mask


def create_difference_map(target: np.ndarray, prediction: np.ndarray, ignore_index: int) -> np.ndarray:
    """
    Create difference map showing correct/incorrect predictions.
    
    Returns:
        0 = ignored pixels
        1 = correct predictions  
        2 = incorrect predictions
    """
    difference_map = np.zeros_like(target, dtype=np.float32)
    
    # Ignored pixels
    ignore_mask = (target == ignore_index)
    difference_map[ignore_mask] = 0
    
    # Valid pixels
    valid_mask = ~ignore_mask
    correct_mask = (target == prediction) & valid_mask
    incorrect_mask = (target != prediction) & valid_mask
    
    difference_map[correct_mask] = 1    # Correct (green)
    difference_map[incorrect_mask] = 2  # Incorrect (red)
    
    return difference_map


def compute_pixel_accuracy(target: torch.Tensor, prediction: torch.Tensor, ignore_index: int) -> float:
    """Compute pixel-wise accuracy excluding ignore_index"""
    valid_mask = (target != ignore_index)
    if valid_mask.sum() == 0:
        return 0.0
    
    correct = (target[valid_mask] == prediction[valid_mask]).sum().item()
    total = valid_mask.sum().item()
    return correct / total


def compute_mean_iou(target: torch.Tensor, prediction: torch.Tensor, num_classes: int, ignore_index: int) -> float:
    """Compute mean IoU across classes excluding ignore_index"""
    ious = []
    valid_mask = (target != ignore_index)
    
    for class_id in range(num_classes):
        target_class = (target == class_id) & valid_mask
        pred_class = (prediction == class_id) & valid_mask
        
        intersection = (target_class & pred_class).sum().item()
        union = (target_class | pred_class).sum().item()
        
        if union > 0:
            ious.append(intersection / union)
    
    return np.mean(ious) if ious else 0.0


def plot_supervision_statistics(
    supervision_info: Dict[str, Any],
    losses: Dict[str, torch.Tensor],
    metrics: Dict[str, float],
    epoch: int,
    logger: Optional[Any] = None
) -> plt.Figure:
    """
    Plot statistics about supervision types and their performance.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # 1. Supervision type distribution
    full_count = len(supervision_info.get('full_indices', []))
    partial_count = len(supervision_info.get('partial_indices', []))
    total_count = supervision_info.get('batch_size', full_count + partial_count)
    
    labels = ['Fully Supervised', 'Partially Supervised']
    sizes = [full_count, partial_count]
    colors = ['lightblue', 'lightcoral']
    
    axes[0, 0].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    axes[0, 0].set_title(f'Supervision Distribution\n(Total: {total_count} samples)')
    
    # 2. Loss breakdown
    loss_names = []
    loss_values = []
    for key, value in losses.items():
        if 'loss' in key.lower():
            loss_names.append(key.replace('_', ' ').title())
            loss_values.append(value.item() if hasattr(value, 'item') else value)
    
    if loss_values:
        bars = axes[0, 1].bar(range(len(loss_names)), loss_values, color='skyblue')
        axes[0, 1].set_xlabel('Loss Type')
        axes[0, 1].set_ylabel('Loss Value')
        axes[0, 1].set_title(f'Loss Breakdown (Epoch {epoch})')
        axes[0, 1].set_xticks(range(len(loss_names)))
        axes[0, 1].set_xticklabels(loss_names, rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, value in zip(bars, loss_values):
            height = bar.get_height()
            axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:.4f}', ha='center', va='bottom', fontsize=8)
    
    # 3. Metrics comparison
    metric_names = []
    metric_values = []
    for key, value in metrics.items():
        if any(x in key.lower() for x in ['accuracy', 'iou', 'f1']):
            metric_names.append(key.replace('_', ' ').title())
            metric_values.append(value)
    
    if metric_values:
        bars = axes[1, 0].bar(range(len(metric_names)), metric_values, color='lightgreen')
        axes[1, 0].set_xlabel('Metric')
        axes[1, 0].set_ylabel('Value')
        axes[1, 0].set_title('Performance Metrics')
        axes[1, 0].set_xticks(range(len(metric_names)))
        axes[1, 0].set_xticklabels(metric_names, rotation=45, ha='right')
        axes[1, 0].set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:.3f}', ha='center', va='bottom', fontsize=8)
    
    # 4. Class distribution (if available)
    axes[1, 1].text(0.5, 0.5, f'Epoch: {epoch}\n\nSupervision Summary:\n'
                    f'• Full: {full_count} samples\n'
                    f'• Partial: {partial_count} samples\n'
                    f'• Total Loss: {losses.get("total_loss", 0):.4f}',
                    ha='center', va='center', transform=axes[1, 1].transAxes,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                    fontsize=12)
    axes[1, 1].set_title('Summary Statistics')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    if logger and hasattr(logger, 'log_image'):
        logger.log_image(key="supervision_statistics", images=[fig], step=epoch)
    
    return fig

