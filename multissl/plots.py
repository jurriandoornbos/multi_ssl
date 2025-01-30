import matplotlib.pyplot as plt
import torch
import numpy as np
# Load dataset (update path as needed)


# Convert tensor images for plotting
def tensor_to_image(tensor):
    """Convert a PyTorch tensor image to a NumPy array for visualization."""
    tensor = tensor.cpu().detach()  # Ensure it's detached from the computation graph
    tensor = tensor.permute(1, 2, 0)  # Convert from CxHxW to HxWxC
    image = tensor.numpy()
      # False Color Composite (NIR, RED, GREEN)
    false_color = np.stack([image[:, :, 3], image[:, :, 1], image[:, :, 0]], axis=-1)  # (H, W, 3)
    return false_color

def plot_first_batch(dataloader):
    # Get first batch
    views, targets, filenames = next(iter(dataloader))
    view0, view1, view2 = views  # Unpacking two views from MultiViewCollate
    # Plot 4 samples with two views each (3 rows, 4 columns)
    fig, axes = plt.subplots(3, 4, figsize=(12, 6))
    print(view0.shape)

    for i in range(4):
        img_view0 = tensor_to_image(view0[i])
        img_view1 = tensor_to_image(view1[i])
        img_view2 = tensor_to_image(view2[i])

        # Row 0: First view
        axes[0, i].imshow(img_view0)
        axes[0, i].axis("off")
        axes[0, i].set_title(f"{filenames[i][:10]}")  # Show truncated filename

        # Row 1: Second view (transformed version)
        axes[1, i].imshow(img_view1)
        axes[1, i].axis("off")

        # Row 2: Third view (transformed version)
        axes[2, i].imshow(img_view2)
        axes[2, i].axis("off")

    axes[0, 0].set_title("View 0 (Original Augmented)")
    axes[1, 0].set_title("View 1 (Transformed)")
    axes[2, 0].set_title("View 2 (Transformed)")

    # Save figure
    save_path = "lightly_multiview_batch.png"
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()

    print(f"Saved visualization to {save_path}")
