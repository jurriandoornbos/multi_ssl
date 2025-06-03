import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from matplotlib.gridspec import GridSpec
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

class ImageSaverCallback(pl.Callback):
    def __init__(self, output_dir="saved_images", every_n_steps=1000, args=None, max_samples=4):
        """
        Args:
            output_dir (str): Directory to save images.
            every_n_steps (int): Save images every n steps.
            args: Additional arguments.
            max_samples (int): Maximum number of samples to visualize.
        """
        super().__init__()
        self.output_dir = output_dir
        self.every_n_steps = every_n_steps
        self.args = args
        self.max_samples = max_samples
        os.makedirs(self.output_dir, exist_ok=True)
        
        # For shared space visualization
        self.pca = None
        
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Save images every n training steps with hallucination visualization."""
        
        # Run for batch 0 or every n steps
        if batch_idx == 0 or trainer.global_step % self.every_n_steps == 0:
            self._visualize_hallucinations(trainer, pl_module, batch, batch_idx)
            
    def _visualize_hallucinations(self, trainer, pl_module, batch, batch_idx):
        """Create comprehensive visualization of hallucinations and shared space."""
        
        pl_module.eval()  # Set to eval mode for consistent results
        
        with torch.no_grad():
            
            # Process data by type
            for data_type in ['rgb_only', 'ms_only', 'aligned']:
                # Skip empty data types
                if data_type == 'rgb_only' and not batch[data_type]['data']:
                    continue
                if data_type == 'ms_only' and not batch[data_type]['data']:
                    continue
                if data_type == 'aligned' and not batch[data_type]['rgb']:
                    continue
                
                # Number of views
                num_views = len(batch[data_type]['data'] if data_type != 'aligned' else batch[data_type]['rgb'])
                
                # Process each view as a batch instead of individual samples
                for view_idx in range(num_views):
                    # Prepare batch inputs for this view
                    if data_type == 'rgb_only':
                        rgb_data = batch[data_type]['data'][view_idx]
                        input_dict = {'rgb': rgb_data}
                    elif data_type == 'ms_only':
                        ms_data = batch[data_type]['data'][view_idx]
                        input_dict = {'ms': ms_data}
                    else:  # aligned
                        rgb_data = batch[data_type]['rgb'][view_idx]
                        ms_data = batch[data_type]['ms'][view_idx]
                        input_dict = {'rgb': rgb_data, 'ms': ms_data}

            # Get device
            device = next(pl_module.parameters()).device
            
            # Ensure data is on correct device and get first few samples
            if rgb_data is not None:
                rgb_data = rgb_data[:self.max_samples].to(device)
            if ms_data is not None:
                ms_data = ms_data[:self.max_samples].to(device)
            
            # Get model's stem for hallucination extraction
            stem = pl_module.backbone.backbone.stem
            
            # Generate all combinations and hallucinations
            results = self._generate_hallucinations(stem, rgb_data, ms_data)
            
            # Create visualization
            self._create_visualization(results, trainer.global_step, batch_idx)
            
        pl_module.train()  # Return to training mode
        
    def _generate_hallucinations(self, stem, rgb_data, ms_data):
        """Generate hallucinations and shared representations."""
        results = {}
        
        # Case 1: Both RGB and MS present
        if rgb_data is not None and ms_data is not None:
            # Process through stems
            rgb_feat = stem.rgb_stem(rgb_data)
            ms_feat = stem.ms_stem(ms_data)
            
            # Get hallucinations
            rgb_to_ms_hall = stem.adapter.rgb_to_ms_hallucinator(rgb_feat)
            ms_to_rgb_hall = stem.adapter.ms_to_rgb_hallucinator(ms_feat)
            
            # Get confidence scores
            rgb_confidence = stem.adapter.hallucination_gate(rgb_feat)
            ms_confidence = stem.adapter.hallucination_gate(ms_feat)
            
            # Apply confidence gating
            rgb_to_ms_gated = rgb_to_ms_hall * rgb_confidence
            ms_to_rgb_gated = ms_to_rgb_hall * ms_confidence
            
            # Get shared representations
            shared_both = stem.adapter.cross_fusion(rgb_feat, ms_feat)
            shared_rgb_only = stem.adapter.cross_fusion(rgb_feat, rgb_to_ms_gated)
            shared_ms_only = stem.adapter.cross_fusion(ms_to_rgb_gated, ms_feat)
            
            results.update({
                'rgb_input': rgb_data,
                'ms_input': ms_data,
                'rgb_feat': rgb_feat,
                'ms_feat': ms_feat,
                'rgb_to_ms_hall': rgb_to_ms_hall,
                'ms_to_rgb_hall': ms_to_rgb_hall,
                'rgb_to_ms_gated': rgb_to_ms_gated,
                'ms_to_rgb_gated': ms_to_rgb_gated,
                'rgb_confidence': rgb_confidence,
                'ms_confidence': ms_confidence,
                'shared_both': shared_both,
                'shared_rgb_only': shared_rgb_only,
                'shared_ms_only': shared_ms_only
            })
            
        # Case 2: Only RGB
        elif rgb_data is not None:
            rgb_feat = stem.rgb_stem(rgb_data)
            rgb_to_ms_hall = stem.adapter.rgb_to_ms_hallucinator(rgb_feat)
            rgb_confidence = stem.adapter.hallucination_gate(rgb_feat)
            rgb_to_ms_gated = rgb_to_ms_hall * rgb_confidence
            shared_rgb_only = stem.adapter.cross_fusion(rgb_feat, rgb_to_ms_gated)
            
            results.update({
                'rgb_input': rgb_data,
                'rgb_feat': rgb_feat,
                'rgb_to_ms_hall': rgb_to_ms_hall,
                'rgb_to_ms_gated': rgb_to_ms_gated,
                'rgb_confidence': rgb_confidence,
                'shared_rgb_only': shared_rgb_only
            })
            
        # Case 3: Only MS
        elif ms_data is not None:
            ms_feat = stem.ms_stem(ms_data)
            ms_to_rgb_hall = stem.adapter.ms_to_rgb_hallucinator(ms_feat)
            ms_confidence = stem.adapter.hallucination_gate(ms_feat)
            ms_to_rgb_gated = ms_to_rgb_hall * ms_confidence
            shared_ms_only = stem.adapter.cross_fusion(ms_to_rgb_gated, ms_feat)
            
            results.update({
                'ms_input': ms_data,
                'ms_feat': ms_feat,
                'ms_to_rgb_hall': ms_to_rgb_hall,
                'ms_to_rgb_gated': ms_to_rgb_gated,
                'ms_confidence': ms_confidence,
                'shared_ms_only': shared_ms_only
            })
            
        return results
    
    def _create_visualization(self, results, global_step, batch_idx):
        """Create comprehensive visualization plot."""
        
        # Determine layout based on available data
        has_rgb = 'rgb_input' in results
        has_ms = 'ms_input' in results
        has_both = has_rgb and has_ms
        
        if has_both:
            fig = plt.figure(figsize=(20, 16))
            gs = GridSpec(4, 6, figure=fig, hspace=0.3, wspace=0.3)
        else:
            fig = plt.figure(figsize=(16, 12))
            gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
        
        # Row 1: Original Inputs
        if has_rgb:
            self._plot_rgb_images(fig, gs, results['rgb_input'], 0, 0, "RGB Input")
        if has_ms:
            col = 2 if has_both else 0
            self._plot_ms_images(fig, gs, results['ms_input'], 0, col, "MS Input")
            
        # Row 2: Feature Maps (stem outputs)
        if has_rgb:
            self._plot_feature_maps(fig, gs, results['rgb_feat'], 1, 0, "RGB Features")
        if has_ms:
            col = 2 if has_both else 0
            self._plot_feature_maps(fig, gs, results['ms_feat'], 1, col, "MS Features")
            
        if has_both:
            # Row 3: Hallucinations
            if 'rgb_to_ms_hall' in results:
                self._plot_feature_maps(fig, gs, results['rgb_to_ms_hall'], 2, 0, "RGB→MS Hallucination")
                self._plot_confidence_maps(fig, gs, results['rgb_confidence'], 2, 1, "RGB Confidence")
                
            if 'ms_to_rgb_hall' in results:
                self._plot_feature_maps(fig, gs, results['ms_to_rgb_hall'], 2, 2, "MS→RGB Hallucination")
                self._plot_confidence_maps(fig, gs, results['ms_confidence'], 2, 3, "MS Confidence")
            
            # Row 4: Shared Representations
            shared_keys = ['shared_both', 'shared_rgb_only', 'shared_ms_only']
            shared_titles = ['Shared (Both)', 'Shared (RGB Only)', 'Shared (MS Only)']
            
            for i, (key, title) in enumerate(zip(shared_keys, shared_titles)):
                if key in results:
                    self._plot_shared_space(fig, gs, results[key], 3, i*2, title)
        else:
            # Single modality case - simpler layout
            if 'rgb_to_ms_hall' in results:
                self._plot_feature_maps(fig, gs, results['rgb_to_ms_hall'], 2, 0, "RGB→MS Hallucination")
                self._plot_confidence_maps(fig, gs, results['rgb_confidence'], 2, 1, "RGB Confidence")
                self._plot_shared_space(fig, gs, results['shared_rgb_only'], 2, 2, "Shared (RGB Only)")
                
            if 'ms_to_rgb_hall' in results:
                self._plot_feature_maps(fig, gs, results['ms_to_rgb_hall'], 2, 0, "MS→RGB Hallucination")
                self._plot_confidence_maps(fig, gs, results['ms_confidence'], 2, 1, "MS Confidence")
                self._plot_shared_space(fig, gs, results['shared_ms_only'], 2, 2, "Shared (MS Only)")
        
        # Add global title
        modality_str = "RGB+MS" if has_both else ("RGB" if has_rgb else "MS")
        fig.suptitle(f'Hallucination Visualization - Step {global_step} - Batch {batch_idx} ({modality_str})', 
                    fontsize=16, fontweight='bold')
        
        # Save figure
        filename = f"hallucination_viz_step_{global_step}_batch_{batch_idx}.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved hallucination visualization: {filepath}")
        
    def _plot_rgb_images(self, fig, gs, rgb_data, row, col, title):
        """Plot RGB images."""
        ax = fig.add_subplot(gs[row, col:col+2])
        
        # Convert to numpy and handle normalization
        rgb_np = rgb_data.cpu().numpy()
        
        # Create grid of images
        n_samples = min(self.max_samples, rgb_np.shape[0])
        grid_img = self._create_image_grid(rgb_np[:n_samples], is_rgb=True)
        
        ax.imshow(grid_img)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.axis('off')
        
    def _plot_ms_images(self, fig, gs, ms_data, row, col, title):
        """Plot MS images (using false color composite)."""
        ax = fig.add_subplot(gs[row, col:col+2])
        
        # Convert to numpy
        ms_np = ms_data.cpu().numpy()
        
        # Create false color composite (use first 3 bands)
        n_samples = min(self.max_samples, ms_np.shape[0])
        grid_img = self._create_image_grid(ms_np[:n_samples, :3], is_rgb=False)
        
        ax.imshow(grid_img)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.axis('off')
        
    def _plot_feature_maps(self, fig, gs, features, row, col, title):
        """Plot feature maps as heatmaps."""
        ax = fig.add_subplot(gs[row, col])
        
        # Average across channels and create visualization
        feat_np = features.cpu().numpy()
        feat_avg = np.mean(feat_np, axis=1)  # Average across channel dimension
        
        # Create grid
        grid_feat = self._create_feature_grid(feat_avg)
        
        im = ax.imshow(grid_feat, cmap='viridis', aspect='auto')
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
    def _plot_confidence_maps(self, fig, gs, confidence, row, col, title):
        """Plot confidence maps."""
        ax = fig.add_subplot(gs[row, col])
        
        conf_np = confidence.cpu().numpy()
        conf_avg = np.mean(conf_np, axis=1)  # Average across channel dimension
        
        grid_conf = self._create_feature_grid(conf_avg)
        
        im = ax.imshow(grid_conf, cmap='Reds', vmin=0, vmax=1, aspect='auto')
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
    def _plot_shared_space(self, fig, gs, shared_features, row, col, title):
        """Plot shared space using PCA/t-SNE visualization."""
        ax = fig.add_subplot(gs[row, col:col+2])
        
        # Flatten features for dimensionality reduction
        shared_np = shared_features.cpu().numpy()
        B, C, H, W = shared_np.shape
        
        # Reshape to (B*H*W, C) for PCA
        shared_flat = shared_np.transpose(0, 2, 3, 1).reshape(-1, C)
        
        # Sample points if too many
        if shared_flat.shape[0] > 1000:
            indices = np.random.choice(shared_flat.shape[0], 1000, replace=False)
            shared_flat = shared_flat[indices]
        
        # Apply PCA
        if self.pca is None or shared_flat.shape[1] != self.pca.n_features_in_:
            self.pca = PCA(n_components=min(3, shared_flat.shape[1]))
            
        try:
            shared_pca = self.pca.fit_transform(shared_flat)
            
            if shared_pca.shape[1] >= 3:
                # 3D scatter plot
                ax.remove()
                ax = fig.add_subplot(gs[row, col:col+2], projection='3d')
                scatter = ax.scatter(shared_pca[:, 0], shared_pca[:, 1], shared_pca[:, 2], 
                                   c=np.arange(len(shared_pca)), cmap='tab10', alpha=0.6)
                ax.set_xlabel('PC1')
                ax.set_ylabel('PC2')
                ax.set_zlabel('PC3')
            else:
                # 2D scatter plot
                scatter = ax.scatter(shared_pca[:, 0], shared_pca[:, 1], 
                                   c=np.arange(len(shared_pca)), cmap='tab10', alpha=0.6)
                ax.set_xlabel('PC1')
                ax.set_ylabel('PC2')
                
            ax.set_title(f'{title} (PCA)', fontsize=10, fontweight='bold')
            
        except Exception as e:
            # Fallback: show feature statistics
            ax.text(0.5, 0.5, f'PCA failed: {str(e)}', transform=ax.transAxes, 
                   ha='center', va='center')
            ax.set_title(title, fontsize=10, fontweight='bold')
            
    def _create_image_grid(self, images, is_rgb=True):
        """Create a grid of images for visualization."""
        B, C, H, W = images.shape
        grid_size = int(np.ceil(np.sqrt(B)))
        
        if is_rgb and C >= 3:
            # RGB images
            grid = np.zeros((grid_size * H, grid_size * W, 3))
            images = images[:, :3]  # Take first 3 channels
            
            # Normalize to [0, 1]
            images = (images - images.min()) / (images.max() - images.min() + 1e-8)
            
        else:
            # Single channel or false color
            grid = np.zeros((grid_size * H, grid_size * W, 3))
            if C == 1:
                images = np.repeat(images, 3, axis=1)
            elif C >= 3:
                images = images[:, :3]
            
            # Normalize
            images = (images - images.min()) / (images.max() - images.min() + 1e-8)
        
        for i in range(B):
            row = i // grid_size
            col = i % grid_size
            if row < grid_size and col < grid_size:
                img = images[i].transpose(1, 2, 0)
                grid[row*H:(row+1)*H, col*W:(col+1)*W] = img
                
        return grid
    
    def _create_feature_grid(self, features):
        """Create a grid of feature maps."""
        B, H, W = features.shape
        grid_size = int(np.ceil(np.sqrt(B)))
        grid = np.zeros((grid_size * H, grid_size * W))
        
        for i in range(B):
            row = i // grid_size
            col = i % grid_size
            if row < grid_size and col < grid_size:
                grid[row*H:(row+1)*H, col*W:(col+1)*W] = features[i]
                
        return grid