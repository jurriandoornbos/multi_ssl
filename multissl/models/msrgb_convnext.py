import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union, Dict

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample when applied in main path of residual blocks."""
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample during training"""
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class ConvNeXtBlock(nn.Module):
    """ConvNeXt Block with depthwise/pointwise convolutions and layer scaling"""
    def __init__(
        self, 
        dim: int, 
        drop_path: float = 0., 
        layer_scale_init_value: float = 1e-6
    ):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim), 
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        x = input + self.drop_path(x)
        return x

class ConvNeXtDownsample(nn.Module):
    """Downsampling layer for ConvNeXt"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.norm = nn.LayerNorm(out_channels)
        
    def forward(self, x):
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        return x

class ModalityAdapter(nn.Module):
    """
    Learned adapter that converts modality-specific features to shared representation.
    Enables cross-modal knowledge transfer even when only one modality is present.
    """
    def __init__(self, shared_dim: int, use_residual: bool = True):
        super().__init__()
        self.shared_dim = shared_dim
        self.use_residual = use_residual
        
        # RGB to shared representation adapter
        self.rgb_adapter = nn.Sequential(
            nn.Conv2d(shared_dim, shared_dim, kernel_size=1),
            nn.BatchNorm2d(shared_dim),
            nn.GELU(),
            nn.Conv2d(shared_dim, shared_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(shared_dim)
        )
        
        # MS to shared representation adapter
        self.ms_adapter = nn.Sequential(
            nn.Conv2d(shared_dim, shared_dim, kernel_size=1),
            nn.BatchNorm2d(shared_dim),
            nn.GELU(),
            nn.Conv2d(shared_dim, shared_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(shared_dim)
        )
        
        # Cross-modal hallucination/completion modules using efficient convolutions
        # These learn to "imagine" the missing modality from the present one
        self.rgb_to_ms_hallucinator = nn.Sequential(
            # Depthwise separable convolution for efficiency
            nn.Conv2d(shared_dim, shared_dim, kernel_size=3, padding=1, groups=shared_dim),  # Depthwise
            nn.Conv2d(shared_dim, shared_dim, kernel_size=1),  # Pointwise
            nn.BatchNorm2d(shared_dim),
            nn.GELU(),
            # Feature transformation with channel mixing
            nn.Conv2d(shared_dim, shared_dim // 2, kernel_size=1),
            nn.BatchNorm2d(shared_dim // 2),
            nn.GELU(),
            nn.Conv2d(shared_dim // 2, shared_dim, kernel_size=1),
            nn.BatchNorm2d(shared_dim)
        )
        
        self.ms_to_rgb_hallucinator = nn.Sequential(
            # Depthwise separable convolution for efficiency
            nn.Conv2d(shared_dim, shared_dim, kernel_size=3, padding=1, groups=shared_dim),  # Depthwise
            nn.Conv2d(shared_dim, shared_dim, kernel_size=1),  # Pointwise
            nn.BatchNorm2d(shared_dim),
            nn.GELU(),
            # Feature transformation with channel mixing
            nn.Conv2d(shared_dim, shared_dim // 2, kernel_size=1),
            nn.BatchNorm2d(shared_dim // 2),
            nn.GELU(),
            nn.Conv2d(shared_dim // 2, shared_dim, kernel_size=1),
            nn.BatchNorm2d(shared_dim)
        )
        
        # Cross-modal fusion using efficient channel attention
        self.cross_fusion = ChannelCrossModalFusion(shared_dim)
        
        # Confidence weighting for hallucinated features
        self.hallucination_gate = nn.Sequential(
            nn.Conv2d(shared_dim, shared_dim // 4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(shared_dim // 4, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Final projection
        self.output_proj = nn.Sequential(
            nn.Conv2d(shared_dim, shared_dim, kernel_size=1),
            nn.BatchNorm2d(shared_dim)
        )
        
    def forward(self, rgb_feat=None, ms_feat=None):
        """
        Adapt modality features to shared representation with cross-modal knowledge transfer
        
        Args:
            rgb_feat: RGB features or None
            ms_feat: MS features or None
            
        Returns:
            Shared representation tensor
        """
        if rgb_feat is None and ms_feat is None:
            raise ValueError("At least one modality must be provided")
        
        # Case 1: Both modalities present - standard cross-modal attention
        if rgb_feat is not None and ms_feat is not None:
            rgb_adapted = self.rgb_adapter(rgb_feat)
            if self.use_residual:
                rgb_adapted = rgb_adapted + rgb_feat
                
            ms_adapted = self.ms_adapter(ms_feat)
            if self.use_residual:
                ms_adapted = ms_adapted + ms_feat
            
            output = self.cross_fusion(rgb_adapted, ms_adapted)
        
        # Case 2: Only RGB present - hallucinate MS features
        elif rgb_feat is not None:
            rgb_adapted = self.rgb_adapter(rgb_feat)
            if self.use_residual:
                rgb_adapted = rgb_adapted + rgb_feat
            
            # Hallucinate MS features from RGB
            hallucinated_ms = self.rgb_to_ms_hallucinator(rgb_adapted)
            
            # Generate confidence score for hallucinated features
            confidence = self.hallucination_gate(rgb_adapted)
            hallucinated_ms = hallucinated_ms * confidence
            
            # Apply cross-modal fusion with hallucinated MS features
            output = self.cross_fusion(rgb_adapted, hallucinated_ms)
        
        # Case 3: Only MS present - hallucinate RGB features
        else:  # ms_feat is not None
            ms_adapted = self.ms_adapter(ms_feat)
            if self.use_residual:
                ms_adapted = ms_adapted + ms_feat
            
            # Hallucinate RGB features from MS  
            hallucinated_rgb = self.ms_to_rgb_hallucinator(ms_adapted)
            
            # Generate confidence score for hallucinated features
            confidence = self.hallucination_gate(ms_adapted)
            hallucinated_rgb = hallucinated_rgb * confidence
            
            # Apply cross-modal fusion with hallucinated RGB features
            output = self.cross_fusion(hallucinated_rgb, ms_adapted)
        
        return self.output_proj(output)

class ChannelCrossModalFusion(nn.Module):
    """
    Lightweight channel-wise cross-modal fusion
    O(C) complexity instead of O(H²W²) attention
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        
        # Channel-wise attention - much lighter than spatial attention
        self.rgb_channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Global average pooling: B,C,H,W -> B,C,1,1
            nn.Conv2d(dim, dim // 8, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim // 8, dim, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.ms_channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 8, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim // 8, dim, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Final output projection
        self.output_proj = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.BatchNorm2d(dim)
        )
        
    def forward(self, rgb_feat, ms_feat):
        """
        Channel attention fusion - each modality guides the other's channel importance
        
        Args:
            rgb_feat: RGB features [B, C, H, W]
            ms_feat: MS features [B, C, H, W]
            
        Returns:
            Fused features [B, C, H, W]
        """
        # MS features guide RGB channel attention
        rgb_weights = self.rgb_channel_attn(ms_feat)  # B, C, 1, 1
        rgb_enhanced = rgb_feat * rgb_weights  # Broadcast multiply
        
        # RGB features guide MS channel attention  
        ms_weights = self.ms_channel_attn(rgb_feat)  # B, C, 1, 1
        ms_enhanced = ms_feat * ms_weights  # Broadcast multiply
        
        # Simple addition fusion
        fused = rgb_enhanced + ms_enhanced
        
        return self.output_proj(fused)

class AdaptiveStem(nn.Module):
    """
    Adaptive stem that converts different input modalities to shared representation
    """
    def __init__(self, rgb_in_channels=3, ms_in_channels=5, shared_dim=96):
        super().__init__()
        self.shared_dim = shared_dim
        
        # Separate stems for each modality
        self.rgb_stem = nn.Sequential(
            nn.Conv2d(rgb_in_channels, shared_dim, kernel_size=4, stride=4),
            nn.BatchNorm2d(shared_dim),
            nn.GELU()
        )
        
        self.ms_stem = nn.Sequential(
            nn.Conv2d(ms_in_channels, shared_dim, kernel_size=4, stride=4),
            nn.BatchNorm2d(shared_dim),
            nn.GELU()
        )
        
        # Modality adapter for fusion
        self.adapter = ModalityAdapter(shared_dim)
        
    def forward(self, rgb=None, ms=None):
        """
        Process inputs through stems and adapt to shared representation
        """
        rgb_feat = self.rgb_stem(rgb) if rgb is not None else None
        ms_feat = self.ms_stem(ms) if ms is not None else None
        
        # Adapt to shared representation
        shared_feat = self.adapter(rgb_feat, ms_feat)
        
        return shared_feat

class ConvNeXtStage(nn.Module):
    """ConvNeXt stage with hierarchical fusion using direct cross-modal operations"""
    def __init__(
        self, 
        in_dim: int, 
        out_dim: int,
        depth: int, 
        drop_paths: List[float], 
        layer_scale_init_value: float = 1e-6,
        downsample: bool = True,
        fusion_interval: int = 2
    ):
        super().__init__()
        self.depth = depth
        self.fusion_interval = fusion_interval
        
        # Downsampling
        if downsample:
            self.downsample = ConvNeXtDownsample(in_dim, out_dim)
        else:
            self.downsample = nn.Identity() if in_dim == out_dim else nn.Conv2d(in_dim, out_dim, kernel_size=1)
        
        # Main processing blocks
        self.blocks = nn.ModuleList([
            ConvNeXtBlock(
                dim=out_dim,
                drop_path=drop_paths[i],
                layer_scale_init_value=layer_scale_init_value
            ) for i in range(depth)
        ])
        
        # Direct fusion modules (no adapter needed)
        self.fusion_modules = nn.ModuleList([
            ChannelCrossModalFusion(out_dim)
            for _ in range(depth // fusion_interval)
        ])
        
    def forward(self, shared_feat):
        """
        Simplified forward pass - just process the shared representation
        
        Args:
            shared_feat: Current shared representation
        """
        # Apply downsampling
        shared_feat = self.downsample(shared_feat)
        
        # Process through blocks 
        for i, block in enumerate(self.blocks):
            shared_feat = block(shared_feat)
            
            # Note: Hierarchical fusion would happen here if we had separate streams
            # But since we already have shared representation, we just continue processing
        
        return shared_feat

class MSRGBConvNeXt(nn.Module):
    """
    Simplified ConvNeXt with learned adapters for MS+RGB inputs
    Uses only hierarchical fusion with attention
    """
    def __init__(
        self,
        rgb_in_channels: int = 3,
        ms_in_channels: int = 5,
        depths: List[int] = [3, 3, 9, 3],
        dims: List[int] = [96, 192, 384, 768],
        drop_path_rate: float = 0.,
        layer_scale_init_value: float = 1e-6,
        out_indices: List[int] = [0, 1, 2, 3],
    ):
        super().__init__()
        self.depths = depths
        self.out_indices = out_indices
        
        # Stochastic depth decay
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        # Adaptive stem
        self.stem = AdaptiveStem(
            rgb_in_channels=rgb_in_channels,
            ms_in_channels=ms_in_channels,
            shared_dim=dims[0]
        )
        
        # Build stages
        self.stages = nn.ModuleList()
        curr_idx = 0
        
        for i in range(len(depths)):
            in_dim = dims[i-1] if i > 0 else dims[0]
            out_dim = dims[i]
            
            stage_dpr = dpr[curr_idx:curr_idx + depths[i]]
            curr_idx += depths[i]
            
            stage = ConvNeXtStage(
                in_dim=in_dim,
                out_dim=out_dim,
                depth=depths[i],
                drop_paths=stage_dpr,
                layer_scale_init_value=layer_scale_init_value,
                downsample=(i > 0),
                fusion_interval=2
            )
            
            self.stages.append(stage)
        
        # Output normalization
        self.norms = nn.ModuleList([
            nn.LayerNorm(dims[i]) for i in out_indices
        ])
        
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
                
    def forward(self, rgb=None, ms=None):
        """Forward pass through the network"""
        # Process through adaptive stem (handles all fusion logic)
        x = self.stem(rgb=rgb, ms=ms)
        
        # Collect outputs
        outs = []
        
        for i, stage in enumerate(self.stages):
            x = stage(x)
            
            if i in self.out_indices:
                # Apply normalization
                feat_norm = x.permute(0, 2, 3, 1)
                feat_norm = self.norms[self.out_indices.index(i)](feat_norm)
                feat_norm = feat_norm.permute(0, 3, 1, 2)
                outs.append(feat_norm)
        
        return outs

class MSRGBConvNeXtFeatureExtractor(nn.Module):
    """Feature extractor wrapper for semantic segmentation tasks"""
    def __init__(
        self,
        model_name: str = 'tiny',
        rgb_in_channels: int = 3,
        ms_in_channels: int = 5,
        drop_path_rate: float = 0.1,
        **kwargs
    ):
        super().__init__()
        
        model_configs = {
            'nano': {
                'depths': [2, 2, 6, 2],
                'dims': [48, 96, 192, 384],
            },
            'tiny': {
                'depths': [3, 3, 9, 3],
                'dims': [96, 192, 384, 768]
            },
            'small': {
                'depths': [3, 3, 27, 3],
                'dims': [96, 192, 384, 768]
            },
            'base': {
                'depths': [3, 3, 27, 3],
                'dims': [128, 256, 512, 1024]
            },
            'large': {
                'depths': [3, 3, 27, 3],
                'dims': [192, 384, 768, 1536]
            }
        }
        
        config = model_configs.get(model_name)
        if not config:
            raise ValueError(f"Unknown model size: {model_name}")
        
        self.backbone = MSRGBConvNeXt(
            rgb_in_channels=rgb_in_channels,
            ms_in_channels=ms_in_channels,
            depths=config['depths'],
            dims=config['dims'],
            drop_path_rate=drop_path_rate,
            **kwargs
        )
        
        self.feature_dims = {
            f'layer{i+1}': config['dims'][i] for i in range(len(config['dims']))
        }
    
    def forward(self, rgb=None, ms=None):
        """Forward pass returning feature dictionary"""
        features = self.backbone(rgb=rgb, ms=ms)
        
        feature_dict = {
            f'layer{i+1}': features[i] for i in range(len(features))
        }
        
        if features:
            global_feat = F.adaptive_avg_pool2d(features[-1], (1, 1))
            feature_dict['flat'] = global_feat.squeeze(-1).squeeze(-1)
        
        return feature_dict