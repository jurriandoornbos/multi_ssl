import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union, Dict

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample when applied in main path of residual blocks.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
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
    """
    ConvNeXt Block with depthwise/pointwise convolutions and layer scaling
    """
    def __init__(
        self, 
        dim: int, 
        drop_path: float = 0., 
        layer_scale_init_value: float = 1e-6
    ):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = nn.LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
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
    def __init__(self, in_channels, out_channels, norm_layer=nn.LayerNorm):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.norm = norm_layer(out_channels)
        
    def forward(self, x):
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        return x

class ModalityFusion(nn.Module):
    """Fusion module to combine features from RGB and MS streams"""
    def __init__(self, dim, fusion_type='concat'):
        super().__init__()
        self.fusion_type = fusion_type
        
        if fusion_type == 'concat':
            self.fusion = nn.Sequential(
                nn.Conv2d(dim * 2, dim, kernel_size=1),
                nn.BatchNorm2d(dim),
                nn.GELU()
            )
        elif fusion_type == 'add':
            self.fusion_rgb = nn.Conv2d(dim, dim, kernel_size=1)
            self.fusion_ms = nn.Conv2d(dim, dim, kernel_size=1)
        elif fusion_type == 'attention':
            self.query = nn.Conv2d(dim, dim, kernel_size=1)
            self.key = nn.Conv2d(dim, dim, kernel_size=1)
            self.value = nn.Conv2d(dim, dim, kernel_size=1)
            self.fusion = nn.Conv2d(dim, dim, kernel_size=1)
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")
    
    def forward(self, rgb_feat, ms_feat):
        if self.fusion_type == 'concat':
            fused = torch.cat([rgb_feat, ms_feat], dim=1)
            return self.fusion(fused)
        elif self.fusion_type == 'add':
            return self.fusion_rgb(rgb_feat) + self.fusion_ms(ms_feat)
        elif self.fusion_type == 'attention':
            # Cross-attention between RGB and MS features
            q = self.query(rgb_feat)
            k = self.key(ms_feat)
            v = self.value(ms_feat)
            
            # Reshape for attention
            b, c, h, w = q.shape
            q = q.view(b, c, -1).permute(0, 2, 1)  # B, HW, C
            k = k.view(b, c, -1)  # B, C, HW
            v = v.view(b, c, -1).permute(0, 2, 1)  # B, HW, C
            
            # Scaled dot-product attention
            attn = F.softmax(torch.bmm(q, k) / (c ** 0.5), dim=-1)
            out = torch.bmm(attn, v).permute(0, 2, 1).view(b, c, h, w)
            
            return self.fusion(out)

class DualModalityStem(nn.Module):
    """
    Stem module that processes RGB and MS inputs separately with optional early fusion
    """
    def __init__(
        self, 
        rgb_in_channels=3, 
        ms_in_channels=5, 
        out_channels=96, 
        fusion_type='concat',
        early_fusion=True
    ):
        super().__init__()
        self.early_fusion = early_fusion
        
        # RGB stem
        self.rgb_stem = nn.Sequential(
            nn.Conv2d(rgb_in_channels, out_channels, kernel_size=4, stride=4),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        
        # MS stem - handle variable channel counts
        self.ms_stem = nn.Sequential(
            nn.Conv2d(ms_in_channels, out_channels, kernel_size=4, stride=4),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        
        # Fusion module to combine RGB and MS features if using early fusion
        if early_fusion:
            self.fusion = ModalityFusion(out_channels, fusion_type)
        
    def forward(self, rgb=None, ms=None):
        # Process available inputs
        rgb_features = self.rgb_stem(rgb) if rgb is not None else None
        ms_features = self.ms_stem(ms) if ms is not None else None
        
        # Fusion based on available inputs and early_fusion setting
        if self.early_fusion and rgb_features is not None and ms_features is not None:
            # Apply fusion
            fused = self.fusion(rgb_features, ms_features)
            return fused, {'rgb': rgb_features, 'ms': ms_features, 'fused': fused}
        elif rgb_features is not None and ms_features is not None:
            # No early fusion, but keep both streams
            return None, {'rgb': rgb_features, 'ms': ms_features}
        elif rgb_features is not None:
            # Only RGB available
            return rgb_features, {'rgb': rgb_features, 'ms': None}
        elif ms_features is not None:
            # Only MS available
            return ms_features, {'rgb': None, 'ms': ms_features}
        else:
            raise ValueError("At least one of RGB or MS input must be provided")

class ConvNeXtStage(nn.Module):
    """
    ConvNeXt stage with optional fusion between RGB and MS streams
    """
    def __init__(
        self, 
        dim: int, 
        out_dim: int,
        depth: int, 
        drop_paths: List[float], 
        layer_scale_init_value: float = 1e-6,
        downsample: bool = True,
        apply_fusion: bool = False,
        fusion_type: str = 'concat',
        fusion_interval: int = 1  # Fuse every N blocks
    ):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.apply_fusion = apply_fusion
        self.fusion_interval = fusion_interval
        
        # Downsampling layer
        if downsample:
            self.downsample = ConvNeXtDownsample(dim, out_dim)
        else:
            self.downsample = nn.Identity() if dim == out_dim else nn.Conv2d(dim, out_dim, kernel_size=1)
        
        # Process streams separately if not using only fused path
        # RGB stream
        self.rgb_blocks = nn.ModuleList([
            ConvNeXtBlock(
                dim=out_dim,
                drop_path=drop_paths[i],
                layer_scale_init_value=layer_scale_init_value
            ) for i in range(depth)
        ])
        
        # MS stream
        self.ms_blocks = nn.ModuleList([
            ConvNeXtBlock(
                dim=out_dim,
                drop_path=drop_paths[i],
                layer_scale_init_value=layer_scale_init_value
            ) for i in range(depth)
        ])
        
        # Fused stream (if using fusion)
        self.fused_blocks = nn.ModuleList([
            ConvNeXtBlock(
                dim=out_dim,
                drop_path=drop_paths[i],
                layer_scale_init_value=layer_scale_init_value
            ) for i in range(depth)
        ])
        
        # Fusion modules
        if apply_fusion:
            self.fusion_layers = nn.ModuleList([
                ModalityFusion(out_dim, fusion_type) 
                for _ in range(depth // fusion_interval)
            ])
    
    def forward(self, x=None, modality_features=None):
        """
        Forward pass for the stage
        
        Args:
            x: Current fused features or None if using separate streams
            modality_features: Dict containing separate RGB and MS streams
        """
        # Apply downsampling to all streams
        if x is not None:
            x = self.downsample(x)
        
        rgb_feat = modality_features.get('rgb')
        ms_feat = modality_features.get('ms')
        
        if rgb_feat is not None:
            rgb_feat = self.downsample(rgb_feat)
        
        if ms_feat is not None:
            ms_feat = self.downsample(ms_feat)
        
        # Process through blocks with optional fusion
        fusion_idx = 0
        for i in range(len(self.rgb_blocks)):
            # Process each stream independently
            if rgb_feat is not None:
                rgb_feat = self.rgb_blocks[i](rgb_feat)
            
            if ms_feat is not None:
                ms_feat = self.ms_blocks[i](ms_feat)
            
            if x is not None:
                x = self.fused_blocks[i](x)
            
            # Apply fusion if needed
            if self.apply_fusion and (i + 1) % self.fusion_interval == 0:
                if rgb_feat is not None and ms_feat is not None:
                    fused = self.fusion_layers[fusion_idx](rgb_feat, ms_feat)
                    fusion_idx += 1
                    
                    if x is None:
                        # First fusion
                        x = fused
                    else:
                        # Update fused stream
                        x = fused
        
        # Update modality features
        modality_features = {
            'rgb': rgb_feat,
            'ms': ms_feat,
            'fused': x
        }
        
        return x, modality_features

class MSRGBConvNeXt(nn.Module):
    """
    ConvNeXt model with dual MS+RGB modality inputs and configurable fusion strategy
    """
    def __init__(
        self,
        rgb_in_channels: int = 3,
        ms_in_channels: int = 5,
        depths: List[int] = [3, 3, 9, 3],
        dims: List[int] = [96, 192, 384, 768],
        drop_path_rate: float = 0.,
        layer_scale_init_value: float = 1e-6,
        fusion_strategy: str = 'hierarchical',  # 'early', 'late', 'hierarchical', 'none'
        fusion_type: str = 'attention',  # 'concat', 'add', 'attention'
        out_indices: List[int] = [0, 1, 2, 3],
    ):
        super().__init__()
        self.depths = depths
        self.fusion_strategy = fusion_strategy
        self.out_indices = out_indices
        
        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        # Stem with early fusion option
        self.stem = DualModalityStem(
            rgb_in_channels=rgb_in_channels,
            ms_in_channels=ms_in_channels,
            out_channels=dims[0],
            fusion_type=fusion_type,
            early_fusion=(fusion_strategy == 'early')
        )
        
        # Configure fusion for stages based on strategy 
        stage_fusion = [False, False, False, False]
        
        if fusion_strategy == 'early':
            # Only fuse at the stem, keep streams separate in stages
            stage_fusion = [False, False, False, False]
        elif fusion_strategy == 'late':
            # Only fuse at the final stage
            stage_fusion = [False, False, False, True]
        elif fusion_strategy == 'hierarchical':
            # Fuse at every stage
            stage_fusion = [True, True, True, True]
        elif fusion_strategy == 'progressive':
            # Gradually increase fusion
            stage_fusion = [False, True, True, True]
        
        # Build stages
        self.stages = nn.ModuleList()
        curr_idx = 0
        
        for i in range(len(depths)):
            # Define input and output dimensions
            if i == 0:
                in_dim = dims[0]
            else:
                in_dim = dims[i-1]
            
            out_dim = dims[i]
            
            # Collect drop paths for this stage
            stage_dpr = dpr[curr_idx:curr_idx + depths[i]]
            curr_idx += depths[i]
            
            # Create stage
            stage = ConvNeXtStage(
                dim=in_dim,
                out_dim=out_dim,
                depth=depths[i],
                drop_paths=stage_dpr,
                layer_scale_init_value=layer_scale_init_value,
                downsample=(i > 0),  # No downsampling for first stage
                apply_fusion=stage_fusion[i],
                fusion_type=fusion_type,
                fusion_interval=2  # Fuse every 2 blocks for efficiency
            )
            
            self.stages.append(stage)
        
        # Normalization for feature maps
        self.norms = nn.ModuleList([
            nn.LayerNorm(dims[i]) for i in out_indices
        ])
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
                

    def forward_features(self, rgb=None, ms=None):
        """
        Forward pass through the feature extraction layers
        """
        # Initial stem processing
        x, modality_features = self.stem(rgb=rgb, ms=ms)
        
        # Store outputs from each stage
        outs = []
        
        # Process through stages
        for i, stage in enumerate(self.stages):
            # Forward through stage
            x, modality_features = stage(x, modality_features)
            
            # Collect features from appropriate stream
            if i in self.out_indices:
                # Correctly select features based on availability
                # Fix: Use explicit conditional checks instead of boolean operators on tensors
                if 'fused' in modality_features and modality_features['fused'] is not None:
                    feat = modality_features['fused']
                elif 'rgb' in modality_features and modality_features['rgb'] is not None:
                    feat = modality_features['rgb']
                elif 'ms' in modality_features and modality_features['ms'] is not None:
                    feat = modality_features['ms']
                else:
                    feat = None
                
                if feat is not None:
                    # Apply norm
                    feat_norm = feat.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
                    feat_norm = self.norms[self.out_indices.index(i)](feat_norm)
                    feat_norm = feat_norm.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
                    outs.append(feat_norm)
        
        return outs
    
    def forward(self, rgb=None, ms=None):
        """Forward pass"""
        return self.forward_features(rgb=rgb, ms=ms)
    
class MSRGBConvNeXtFeatureExtractor(nn.Module):
    """
    Feature extractor based on ConvNeXt for semantic segmentation tasks
    with RGB and MS input support
    """
    def __init__(
        self,
        model_name: str = 'tiny',  # 'tiny', 'small', 'base', 'large'
        rgb_in_channels: int = 3,
        ms_in_channels: int = 5,
        fusion_strategy: str = 'hierarchical',
        fusion_type: str = 'attention',
        drop_path_rate: float = 0.1,
        **kwargs
    ):
        super().__init__()
        
        # Model configurations
        model_configs = {
            'nano': {
                 "depths": [2, 2, 6, 2],  # Reduced depths
                 "dims":[48, 96, 192, 384],  # Reduced width
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
        
        # Create backbone
        self.backbone = MSRGBConvNeXt(
            rgb_in_channels=rgb_in_channels,
            ms_in_channels=ms_in_channels,
            depths=config['depths'],
            dims=config['dims'],
            fusion_strategy=fusion_strategy,
            fusion_type=fusion_type,
            drop_path_rate=drop_path_rate,
            **kwargs
        )
        
        # Store feature dimensions for segmentation heads
        self.feature_dims = {
            f'layer{i+1}': config['dims'][i] for i in range(len(config['dims']))
        }
    
    def forward(self, rgb=None, ms=None):
        """
        Forward pass with support for dual modality inputs
        
        Returns:
            Dictionary of features compatible with segmentation heads
        """
        # Extract multi-scale features
        features = self.backbone(rgb=rgb, ms=ms)
        
        # Create feature dictionary for compatibility with segmentation heads
        feature_dict = {
            f'layer{i+1}': features[i] for i in range(len(features))
        }
        
        # Add a flat representation for global features
        if features:
            global_feat = F.adaptive_avg_pool2d(features[-1], (1, 1))
            feature_dict['flat'] = global_feat.squeeze(-1).squeeze(-1)
        
        return feature_dict