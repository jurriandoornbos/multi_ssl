
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Dict, Any, Optional
import numpy as np

from .msrgb_convnext_upernet import MSRGBConvNeXtUPerNet

from ..plotting.semi_plots import plot_mixed_supervision_validation, plot_supervision_statistics
import matplotlib.pyplot as plt

class MSRGBConvNeXtUPerNetMixed(MSRGBConvNeXtUPerNet):
    """
    UPerNet model extended to handle mixed supervision training.
    
    Uses different loss weighting for fully vs partially labeled samples.
    """
    
    def __init__(
        self,
        # Base UPerNet parameters
        model_size='tiny',
        rgb_in_channels=3,
        ms_in_channels=5,
        num_classes=2,
        ignore_index=255,
        
        # Mixed supervision parameters
        full_supervision_weight=1.0,
        partial_supervision_weight=0.3,
        consistency_weight=0.1,
        use_consistency_loss=True,
        
        # Other parameters
        **kwargs
    ):
        """
        Args:
            full_supervision_weight: Weight for fully supervised loss
            partial_supervision_weight: Weight for partially supervised loss
            consistency_weight: Weight for consistency regularization
            use_consistency_loss: Whether to use consistency loss between predictions
        """
        super().__init__(
            model_size=model_size,
            rgb_in_channels=rgb_in_channels,
            ms_in_channels=ms_in_channels,
            num_classes=num_classes,
            ignore_index=ignore_index,
            **kwargs
        )
        
        # Mixed supervision parameters
        self.full_supervision_weight = full_supervision_weight
        self.partial_supervision_weight = partial_supervision_weight
        self.consistency_weight = consistency_weight
        self.use_consistency_loss = use_consistency_loss
        
        # Create separate loss functions for different supervision types
        self.full_criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.partial_criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
        
        # For consistency loss (comparing predictions from different augmentations)
        self.consistency_criterion = nn.KLDivLoss(reduction='mean')
        
    def compute_mixed_supervision_loss(
        self, 
        outputs: torch.Tensor, 
        targets: torch.Tensor,
        supervision_info: Dict[str, Any]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute loss for mixed supervision training.
        
        Args:
            outputs: Model predictions [B, C, H, W]
            targets: Ground truth masks [B, H, W]
            supervision_info: Dictionary with supervision type information
            
        Returns:
            Dictionary of losses
        """
        losses = {}
        total_loss = torch.tensor(0.0, device=outputs.device)
        
        # Get indices for different supervision types
        full_indices = supervision_info.get('full_indices', [])
        partial_indices = supervision_info.get('partial_indices', [])
        
        # Compute fully supervised loss
        if len(full_indices) > 0:
            full_outputs = outputs[full_indices]
            full_targets = targets[full_indices]
            
            full_loss = self.full_criterion(full_outputs, full_targets)
            losses['full_supervision_loss'] = full_loss
            total_loss += self.full_supervision_weight * full_loss
        else:
            losses['full_supervision_loss'] = torch.tensor(0.0, device=outputs.device)
        
        # Compute partially supervised loss
        if len(partial_indices) > 0:
            partial_outputs = outputs[partial_indices]
            partial_targets = targets[partial_indices]
            
            partial_loss = self.partial_criterion(partial_outputs, partial_targets)
            losses['partial_supervision_loss'] = partial_loss
            total_loss += self.partial_supervision_weight * partial_loss
        else:
            losses['partial_supervision_loss'] = torch.tensor(0.0, device=outputs.device)
        
        # Compute consistency loss (if enabled)
        if self.use_consistency_loss and len(partial_indices) > 0:
            consistency_loss = self.compute_consistency_loss(
                outputs[partial_indices], 
                partial_targets
            )
            losses['consistency_loss'] = consistency_loss
            total_loss += self.consistency_weight * consistency_loss
        else:
            losses['consistency_loss'] = torch.tensor(0.0, device=outputs.device)
        
        losses['total_loss'] = total_loss
        return losses
    
    def compute_consistency_loss(
        self, 
        outputs: torch.Tensor, 
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute consistency loss for partially labeled data.
        
        This encourages consistent predictions for pixels that are not labeled.
        """
        # Create mask for unlabeled pixels (ignore_index)
        unlabeled_mask = (targets == self.ignore_index)
        
        if not unlabeled_mask.any():
            return torch.tensor(0.0, device=outputs.device)
        
        # Get predictions for unlabeled pixels
        probs = F.softmax(outputs, dim=1)
        
        # Compute entropy loss for unlabeled pixels (encourage confident predictions)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1)
        unlabeled_entropy = entropy[unlabeled_mask]
        
        # Return mean entropy (lower is better - more confident predictions)
        return unlabeled_entropy.mean()
    
    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Training step with mixed supervision handling"""
        # Extract data
        images = batch['images']
        masks = batch['masks']
        
        # Get input based on available modalities
        rgb = images if images.shape[1] == 3 else None
        ms = images if images.shape[1] > 3 else None
        
        # Forward pass
        outputs = self(rgb=rgb, ms=ms)

        if torch.isnan(outputs).any():
            print("Model outputs contain NaNs!")
        
        # Handle auxiliary loss if present
        if self.use_aux_loss:
            main_outputs, aux_outputs = outputs
            
            # Compute main loss
            main_losses = self.compute_mixed_supervision_loss(
                main_outputs, masks, batch
            )
            
            # Compute auxiliary loss
            aux_losses = self.compute_mixed_supervision_loss(
                aux_outputs, masks, batch
            )
            
            # Combine losses
            total_loss = main_losses['total_loss'] + 0.4 * aux_losses['total_loss']
            
            # Log auxiliary losses
            for key, value in aux_losses.items():
                self.log(f'train_aux_{key}', value, on_step=True, on_epoch=True)
                
        else:
            main_losses = self.compute_mixed_supervision_loss(
                outputs, masks, batch
            )
            total_loss = main_losses['total_loss']
        
        # Log main losses
        for key, value in main_losses.items():
            self.log(f'train_{key}', value, on_step=True, on_epoch=True, prog_bar=True)
        
        # Log supervision type statistics
        full_count = len(batch.get('full_indices', []))
        partial_count = len(batch.get('partial_indices', []))
        total_count = batch['batch_size']
        
        self.log('train_full_ratio', full_count / total_count, on_step=True, on_epoch=True)
        self.log('train_partial_ratio', partial_count / total_count, on_step=True, on_epoch=True)
        
        return total_loss
    
    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Enhanced validation step with plotting functionality"""
        images = batch['images']
        masks = batch['masks']
        
        # Get input based on available modalities
        rgb = images if images.shape[1] == 3 else None
        ms = images if images.shape[1] > 3 else None
        
        # Forward pass
        outputs = self(rgb=rgb, ms=ms)
        
        # Handle auxiliary loss if present
        if self.use_aux_loss:
            outputs = outputs[0]  # Use main output for validation
        
        # Compute losses
        losses = self.compute_mixed_supervision_loss(outputs, masks, batch)
        
        # Log losses
        for key, value in losses.items():
            self.log(f'val_{key}', value, on_step=False, on_epoch=True, prog_bar=True)
        
        # Compute additional metrics
        metrics = self.compute_and_log_metrics(outputs, masks, batch, prefix='val_')
        
        # Plot validation results periodically
        if batch_idx == 0: # Plot every 5 epochs for first batch
            try:
                
                # Create validation plot
                fig = plot_mixed_supervision_validation(
                    images=images,
                    targets=masks,
                    predictions=outputs,
                    supervision_info=batch,
                    class_names=getattr(self, 'class_names', None),
                    ignore_index=self.ignore_index if hasattr(self, 'ignore_index') else 255,
                    max_samples=min(4, images.shape[0]),
                    logger=self.logger,
                    global_step=self.global_step
                )
                
                # Log to tensorboard/wandb if available
                if hasattr(self.logger, 'experiment'):
                    if hasattr(self.logger.experiment, 'add_figure'):  # TensorBoard
                        self.logger.experiment.add_figure(
                            'validation_predictions', fig, self.global_step
                        )
                    elif hasattr(self.logger.experiment, 'log'):  # Weights & Biases
                        import wandb
                        self.logger.experiment.log({
                            "validation_predictions": wandb.Image(fig),
                            "epoch": self.current_epoch
                        })
                
                # Close the figure to free memory
                plt.close(fig)
                
                # Create supervision statistics plot
                stats_fig = plot_supervision_statistics(
                    supervision_info=batch,
                    losses=losses,
                    metrics=metrics or {},
                    epoch=self.current_epoch,
                    logger=self.logger
                )
                
                # Log statistics plot
                if hasattr(self.logger, 'experiment'):
                    if hasattr(self.logger.experiment, 'add_figure'):  # TensorBoard
                        self.logger.experiment.add_figure(
                            'supervision_statistics', stats_fig, self.global_step
                        )
                    elif hasattr(self.logger.experiment, 'log'):  # Weights & Biases
                        import wandb
                        self.logger.experiment.log({
                            "supervision_statistics": wandb.Image(stats_fig),
                            "epoch": self.current_epoch
                        })
                
                plt.close(stats_fig)
                
            except Exception as e:
                print(f"Warning: Could not create validation plots: {e}")
        
        return losses

    
    def compute_and_log_metrics(
        self, 
        outputs: torch.Tensor, 
        targets: torch.Tensor,
        batch_info: Dict[str, Any],
        prefix: str = ''
    ):
        """Compute and log additional metrics"""
        with torch.no_grad():
            # Get predictions
            preds = outputs.argmax(dim=1)
            
            # Overall accuracy (excluding ignore_index)
            valid_mask = (targets != self.ignore_index)
            if valid_mask.sum() > 0:
                accuracy = (preds[valid_mask] == targets[valid_mask]).float().mean()
                self.log(f'{prefix}accuracy', accuracy, on_step=False, on_epoch=True)
            
            # Per-supervision-type metrics
            full_indices = batch_info.get('full_indices', [])
            partial_indices = batch_info.get('partial_indices', [])
            
            if len(full_indices) > 0:
                full_preds = preds[full_indices]
                full_targets = targets[full_indices]
                full_valid = (full_targets != self.ignore_index)
                
                if full_valid.sum() > 0:
                    full_acc = (full_preds[full_valid] == full_targets[full_valid]).float().mean()
                    self.log(f'{prefix}full_accuracy', full_acc, on_step=False, on_epoch=True)
            
            if len(partial_indices) > 0:
                partial_preds = preds[partial_indices]
                partial_targets = targets[partial_indices]
                partial_valid = (partial_targets != self.ignore_index)
                
                if partial_valid.sum() > 0:
                    partial_acc = (partial_preds[partial_valid] == partial_targets[partial_valid]).float().mean()
                    self.log(f'{prefix}partial_accuracy', partial_acc, on_step=False, on_epoch=True)
