import torch
from ..data import OrthoChipDataset
import xarray as xr
from torch.utils.data import DataLoader
import numpy as np

def predict_ortho(
    model: torch.nn.Module,
    ortho_dataset: OrthoChipDataset,
    ortho_template: xr.DataArray,
    batch_size: int = 8,
    num_workers: int = 4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    output_channels: int = 2  # Default for binary segmentation
) -> xr.DataArray:
    """
    Process an orthomosaic through a model chip by chip and reconstruct the full-sized orthomosaic
    with model predictions, maintaining exact spatial relationships without interpolation.
    
    Args:
        model: PyTorch model that takes image chips and produces predictions
        ortho_dataset: OrthoChipDataset containing the chips to process
        ortho_template: Template orthomosaic to use for reconstructing coordinates and projection
        batch_size: Batch size for processing
        num_workers: Number of workers for data loading
        device: Device to run model on ("cuda" or "cpu")
        output_channels: Number of output channels from the model
        
    Returns:
        xr.DataArray: Reconstructed orthomosaic with model predictions
    """
    # Set model to evaluation mode
    model.eval()
    model.to(device)
    
    # Create a DataLoader for efficient processing
    dataloader = DataLoader(
        ortho_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Determine model output resolution (we'll run inference on one chip to check)
    # This helps us set up the output array with the correct dimensions
    with torch.no_grad():
        sample_chip, _, sample_bounds, sample_img_coords = ortho_dataset[0]
        sample_chip = sample_chip.unsqueeze(0).to(device)  # Add batch dimension
        sample_output = model(sample_chip)
        
        # Handle different model output formats
        if isinstance(sample_output, tuple):
            sample_output = sample_output[0]  # Some models return (output, features)
            
        # If multi-class, get output shape
        if output_channels > 1:
            _, _, out_h, out_w = sample_output.shape
        else:
            _, out_h, out_w = sample_output.shape
            
        # Calculate scaling factor between input and output
        in_h, in_w = sample_chip.shape[2], sample_chip.shape[3]
        scale_h, scale_w = out_h / in_h, out_w / in_w
    
    # Get dimensions of the original orthomosaic
    template_shape = ortho_template.shape
    if len(template_shape) > 2:
        template_h, template_w = template_shape[1], template_shape[2]
    else:
        template_h, template_w = template_shape[0], template_shape[1]
    
    # Calculate dimensions of the output orthomosaic
    output_h = int(template_h * scale_h)
    output_w = int(template_w * scale_w)
    
    # Create prediction array with the scaled dimensions
    if output_channels > 1:
        # Multi-class case
        prediction_array = np.zeros((output_channels, output_h, output_w), dtype=np.float32)
    else:
        # Binary case
        prediction_array = np.zeros((output_h, output_w), dtype=np.float32)
    
    # Count array to track how many predictions are made for each pixel
    count_array = np.zeros((output_h, output_w), dtype=np.int32)
    
    # Process all chips
    with torch.no_grad():
        for batch in dataloader:
            # Unpack batch
            chips, _, bounds_list, img_coords_list = batch
            chips = chips.to(device)
            
            # Forward pass
            outputs = model(chips)
            outputs = outputs.permute(0,1,3,2)
            # Handle different model output formats
            if isinstance(outputs, tuple):
                outputs = outputs[0]  # Some models return (output, features)
            
            # Move outputs to CPU and convert to numpy
            outputs = outputs.cpu().numpy()
            if batch_size ==1:
                img_coords_list = [img_coords_list]
                bounds_list = [bounds_list]
            

            # For each chip in the batch
            for i, (img_coords, bounds) in enumerate(zip(img_coords_list, bounds_list)):
                # Convert img_coords from tensor to tuple if needed
                if isinstance(img_coords, torch.Tensor):
                    img_coords = tuple(img_coords.numpy())
                
                # Get the image coordinates from the original ortho
                row_min, col_min, row_max, col_max = [int(coord) for coord in img_coords]
                
                # Scale these coordinates to the output dimensions
                out_row_min = int(row_min * scale_h)
                out_col_min = int(col_min * scale_w)
                out_row_max = int(row_max * scale_h)
                out_col_max = int(col_max * scale_w)
                
                # Ensure they're within bounds of the output array
                out_row_min = max(0, out_row_min)
                out_col_min = max(0, out_col_min)
                out_row_max = min(output_h, out_row_max)
                out_col_max = min(output_w, out_col_max)
                
                # Skip if the region is invalid (this can happen at edges)
                if out_row_max <= out_row_min or out_col_max <= out_col_min:
                    continue
                
                # Get output for this chip
                chip_output = outputs[i]
                
                # Calculate target dimensions in the output array
                target_h = out_row_max - out_row_min
                target_w = out_col_max - out_col_min
                
                # Get dimensions of the model output
                if output_channels > 1:
                    pred_h, pred_w = chip_output.shape[1], chip_output.shape[2]
                else:
                    pred_h, pred_w = chip_output.shape
                
                # Handle cases where the chip output is larger than the target area
                if pred_h > target_h or pred_w > target_w:
                    # Create a view or slice of the chip output that matches the target dimensions
                    if output_channels > 1:
                        # For multi-channel output, take slices up to the target dimensions
                        chip_output_adjusted = chip_output[:, :target_h, :target_w]
                    else:
                        # For single-channel output
                        chip_output_adjusted = chip_output[:target_h, :target_w]
                        
                    # Add the adjusted output to the prediction array
                    if output_channels > 1:
                        prediction_array[:, out_row_min:out_row_max, out_col_min:out_col_max] += chip_output_adjusted
                    else:
                        prediction_array[out_row_min:out_row_max, out_col_min:out_col_max] += chip_output_adjusted
                        
                    # Update the count array
                    count_array[out_row_min:out_row_max, out_col_min:out_col_max] += 1
                
                # Handle cases where the chip output is smaller than the target area
                elif pred_h < target_h or pred_w < target_w:
                    # Calculate how much of the chip output we can use
                    use_h = min(pred_h, target_h)
                    use_w = min(pred_w, target_w)
                    
                    # Add partial chip output to prediction array
                    if output_channels > 1:
                        prediction_array[:, out_row_min:out_row_min+use_h, out_col_min:out_col_min+use_w] += \
                            chip_output[:, :use_h, :use_w]
                    else:
                        prediction_array[out_row_min:out_row_min+use_h, out_col_min:out_col_min+use_w] += \
                            chip_output[:use_h, :use_w]
                    
                    # Update count array for the partial region
                    count_array[out_row_min:out_row_min+use_h, out_col_min:out_col_min+use_w] += 1
                
                # Dimensions match exactly
                else:
                    if output_channels > 1:
                        prediction_array[:, out_row_min:out_row_max, out_col_min:out_col_max] += chip_output
                    else:
                        prediction_array[out_row_min:out_row_max, out_col_min:out_col_max] += chip_output
                    
                    # Update count array
                    count_array[out_row_min:out_row_max, out_col_min:out_col_max] += 1

    # Average predictions where there are overlaps
    mask = count_array > 0
    if output_channels > 1:
        for c in range(output_channels):
            prediction_array[c, mask] /= count_array[mask]
    else:
        prediction_array[mask] /= count_array[mask]
    
    # Create xarray DataArray with scaled dimensions but same extent as template
    # We need to create new coordinates that match the scaled dimensions but cover the same spatial extent
    
    # Get the spatial bounds from the template
    x_min, y_min, x_max, y_max = (
        ortho_template.x.min().item(),
        ortho_template.y.min().item(),
        ortho_template.x.max().item(),
        ortho_template.y.max().item()
    )
    
    # Create new coordinate arrays with the scaled dimensions
    new_x = np.linspace(x_min, x_max, output_w)
    new_y = np.linspace(y_max, y_min, output_h)  # Note: y is usually top to bottom
    
    # Create the xarray DataArray
    if output_channels > 1:
        # Multi-class case - add new dimension for classes
        class_coords = np.arange(output_channels)
        prediction_da = xr.DataArray(
            prediction_array,
            coords={
                'class': class_coords,
                'y': new_y,
                'x': new_x
            },
            dims=['class', 'y', 'x']
        )
    else:
        # Binary case
        prediction_da = xr.DataArray(
            prediction_array,
            coords={
                'y': new_y,
                'x': new_x
            },
            dims=['y', 'x']
        )
    
    # Copy CRS information
    prediction_da.rio.write_crs(ortho_template.rio.crs, inplace=True)
    
    return prediction_da