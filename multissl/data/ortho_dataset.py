import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import xarray as xr
import rioxarray as rxr
import geopandas as gpd
import uavgeo as ug
from typing import Optional, Dict, Tuple, List, Union
import os
import matplotlib.pyplot as plt

class OrthoChipDataset(Dataset):
    """
    Dataset that processes an orthomosaic into chips for deep learning.
    Can use predefined chips from a GeoDataFrame or generate them automatically.
    """
    def __init__(
        self,
        ortho: xr.DataArray,
        mask: Optional[xr.DataArray] = None,
        chip_gdf: Optional[gpd.GeoDataFrame] = None,
        img_size: Tuple[int, int] = (224, 224),
        transform=None,
        mask_transform=None,
        remove_empty_chips: bool = True,
        empty_threshold: float = 20.0,
        mask_empty_threshold: float = 99.0
    ):
        """
        Initialize the dataset with an orthomosaic and optional mask.
        
        Args:
            ortho: Input orthomosaic as xarray DataArray
            mask: Optional mask as xarray DataArray
            chip_gdf: Optional GeoDataFrame with predefined chips
            img_size: Tuple of (height, width) for chips if generating them
            transform: Optional transform for ortho chips
            mask_transform: Optional transform for mask chips
            remove_empty_chips: Whether to remove chips with too many zero/nodata pixels
            empty_threshold: Percentage threshold of zeros to consider a chip as empty
            mask_empty_threshold: Percentage threshold of zeros in mask to remove a chip
        """
        self.ortho = ortho
        self.mask = mask
        self.transform = transform
        self.mask_transform = mask_transform
        self.img_size = img_size
        
        # Generate or use provided chip GeoDataFrame
        if chip_gdf is None:
            input_dims = {"x": img_size[1], "y": img_size[0]}
            self.rio_gdf, self.im_gdf = self.generate_chips(ortho, input_dims)
        else:
            raise NotImplementedError()
            self.rio_gdf = chip_gdf.copy()
            self.im_gdf= chip_gdf.copy()
        
        # Remove empty/outlier chips if requested
        if remove_empty_chips:
            indices  = self.remove_outlier_chips(
                self.rio_gdf, ortho, perc=empty_threshold
            )

            self.rio_gdf = self.rio_gdf.loc[indices]
            self.im_gdf = self.im_gdf.loc[indices]
            if mask is not None:
                self.rio_gdf = self.remove_outlier_chips(
                    self.rio_gdf, mask, perc=mask_empty_threshold
                )
                
        # Cache the bounds for faster retrieval
        self.crsbounds = [row.geometry.bounds for i, row in self.rio_gdf.iterrows()]
        self.imbounds = [row.geometry.bounds for i, row in self.im_gdf.iterrows()]
        # Get band dimension for input
        self.n_bands = ortho.sizes['band'] if 'band' in ortho.dims else 1
        
    def generate_chips(self, raster, dims):
        """Generate a GeoDataFrame of chip geometries based on raster dimensions"""
        # Get the size of the total orthomosaic
        s_x = raster.shape[2] if len(raster.shape) > 2 else raster.shape[1]
        s_y = raster.shape[1] if len(raster.shape) > 2 else raster.shape[0]
        
        # Get CRS to apply to the GDF
        crs = raster.rio.crs
        
        # Create chip bounds
        gdf = ug.compute.create_chip_bounds_gdf(
            input_dims=dims, shape_x=s_x, shape_y=s_y, crs=crs
        )
        
        # Convert image coordinates to CRS coordinates
        gdf = ug.compute.imgref_to_crsref_boxes(raster=raster, gdf=gdf)
        
        # Set the geometry column
        gdf.set_geometry("c_geom", crs=crs, inplace=True)
        
        # Clean up duplicate geometry columns
        if 'geometry' in gdf.columns and gdf._geometry_column_name != 'geometry':
            gdf = gdf.drop(columns=['geometry'])
        chip_bounds = ug.compute.create_chip_bounds_gdf(
            input_dims=dims, shape_x=s_x, shape_y=s_y, crs=crs
        )
        return gdf.rename_geometry("geometry"), chip_bounds
    
    def remove_outlier_chips(self, gdf, ortho, perc=20):
        """Remove chips with too many zero/nodata pixels"""
        valid_indices = []
        
        for i, row in gdf.iterrows():
            # Clip the image to the geometry bounds
            clipped_img = ortho.rio.clip_box(
                minx=row.geometry.bounds[0],
                miny=row.geometry.bounds[1],
                maxx=row.geometry.bounds[2],
                maxy=row.geometry.bounds[3]
            )
            
            # Count zeros
            # For multi-band images, consider a pixel as zero if all bands are zero
            if len(clipped_img.data.shape) > 2:  # Multi-band case
                zeros_count = np.sum(np.all(clipped_img.data == 0, axis=0))
                total_pixels = clipped_img.data.shape[1] * clipped_img.data.shape[2]
            else:  # Single band case
                zeros_count = np.sum(clipped_img.data == 0)
                total_pixels = clipped_img.data.size
            
            # Calculate percentage of zeros
            zero_percentage = (zeros_count / total_pixels) * 100
            
            # Keep if less than threshold percentage of zeros
            if zero_percentage < perc:
                valid_indices.append(i)
        
        return valid_indices
    
    def __len__(self):
        """Return the number of chips in the dataset"""
        return len(self.rio_gdf)
    
    def __getitem__(self, idx):
        """Get a specific chip by index"""
        # Get the row from the GeoDataFrame for this idx
        row = self.rio_gdf.iloc[idx]
        rio_bounds = row.geometry.bounds
        im_row = self.im_gdf.iloc[idx]
        im_bounds = im_row.geometry.bounds
        
        
        # Clip ortho to bounds
        ortho_chip = self.ortho.rio.clip_box(
            minx=rio_bounds[0], miny=rio_bounds[1], maxx=rio_bounds[2], maxy=rio_bounds[3]
        )
        
        
        # Convert to numpy array with proper shape
        if len(ortho_chip.shape) > 2:
            # Multi-band case
            ortho_array = np.moveaxis(ortho_chip.values, 0, 2)  # From (C,H,W) to (H,W,C)
        else:
            # Single band case
            ortho_array = ortho_chip.values[:, :, np.newaxis]  # Add channel dimension
        
        # Apply transforms to ortho if provided
        if self.transform:
            ortho_tensor = self.transform(ortho_array)
        else:
            # Default conversion to tensor
            ortho_tensor = torch.from_numpy(ortho_array).float()
            
            # Normalize if uint8
            if ortho_tensor.dtype == torch.uint8:
                ortho_tensor = ortho_tensor / 255.0
                
            # Move channels to first dimension if needed
            if ortho_tensor.ndim == 3 and ortho_tensor.shape[2] <= self.n_bands:
                ortho_tensor = ortho_tensor.permute(2, 0, 1)  # From (H,W,C) to (C,H,W)
        
        # If mask is provided, process it too
        if self.mask is not None:
            mask_chip = self.mask.rio.clip_box(
                minx=rio_bounds[0], miny=rio_bounds[1], maxx=rio_bounds[2], maxy=rio_bounds[3]
            )
            
            # Get mask array
            mask_array = mask_chip.values
            
            # Handle multi-band masks
            if len(mask_array.shape) > 2:
                mask_array = mask_array[0]  # Usually masks are single channel
            
            # Apply transforms to mask if provided
            if self.mask_transform:
                mask_tensor = self.mask_transform(mask_array)
            else:
                # Default conversion to long tensor
                mask_tensor = torch.from_numpy(mask_array).long()
            
            return ortho_tensor, mask_tensor, rio_bounds, im_bounds
        
        return ortho_tensor, torch.tensor([0]), rio_bounds, im_bounds