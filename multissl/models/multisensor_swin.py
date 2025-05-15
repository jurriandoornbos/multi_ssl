import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from typing import Dict, Optional, Tuple, List
from collections import OrderedDict


class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: nn.Module = nn.GELU,
        drop: float = 0.
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
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
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


def window_partition(x, window_size):
    """
    Args:
        x: (B, N, H, W, C) where N is number of bands
        window_size (int): window size
    Returns:
        windows: (num_windows*B, N, window_size, window_size, C)
    """
    B, N, H, W, C = x.shape
    x = x.view(B, N, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 2, 4, 1, 3, 5, 6).contiguous().view(-1, N, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, num_bands, H, W):
    """
    Args:
        windows: (num_windows*B, N, window_size, window_size, C)
        window_size (int): Window size
        num_bands (int): Number of spectral bands
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, N, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, num_bands, window_size, window_size, -1)
    x = x.permute(0, 3, 1, 4, 2, 5, 6).contiguous().view(B, num_bands, H, W, -1)
    return x


class UAVChannelGroupTokenizer(nn.Module):
    """
    Tokenizer for UAV imagery supporting RGB and Multispectral data
    with Galileo-inspired channel grouping
    """
    def __init__(
        self,
        embed_dim: int = 128,
        patch_size: int = 4,
        norm_layer: Optional[nn.Module] = nn.LayerNorm,
        use_channel_embeds: bool = True,
        project_per_band: bool = True
    ):
        super().__init__()
        
        # Define channel groups for UAV data
        self.CHANNEL_GROUPS = OrderedDict({
            'RGB_R': [0],      # RGB Red
            'RGB_G': [1],      # RGB Green 
            'RGB_B': [2],      # RGB Blue
            'MS_B': [0],       # Multispectral Blue (may be missing)
            'MS_G': [1],       # Multispectral Green
            'MS_R': [2],       # Multispectral Red
            'MS_RE': [3],      # Red Edge
            'MS_NIR': [4],     # Near Infrared
        })
        
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.project_per_band = project_per_band
        
        # Create separate projections for each channel group
        if project_per_band:
            self.rgb_projections = nn.ModuleDict({
                f'RGB_{band}': nn.Conv2d(1, embed_dim, kernel_size=patch_size, stride=patch_size)
                for band in ['R', 'G', 'B']
            })
            
            self.ms_projections = nn.ModuleDict({
                f'MS_{band}': nn.Conv2d(1, embed_dim, kernel_size=patch_size, stride=patch_size)
                for band in ['B', 'G', 'R', 'RE', 'NIR']
            })
        
        # Layer norm
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        
        # Channel embeddings
        if use_channel_embeds:
            self.rgb_channel_embed = nn.Parameter(
                torch.zeros(3, embed_dim), requires_grad=True
            )
            self.ms_channel_embed = nn.Parameter(
                torch.zeros(5, embed_dim), requires_grad=True
            )
            nn.init.trunc_normal_(self.rgb_channel_embed, std=0.02)
            nn.init.trunc_normal_(self.ms_channel_embed, std=0.02)
        else:
            self.register_buffer('rgb_channel_embed', torch.zeros(3, embed_dim))
            self.register_buffer('ms_channel_embed', torch.zeros(5, embed_dim))
    
    def _detect_ms_configuration(self, ms: torch.Tensor) -> Tuple[bool, List[str], List[int]]:
        """Detect if MS data is 4 or 5 channel and return configuration"""
        c_ms = ms.shape[1]
        
        if c_ms == 5:
            has_blue = True
            band_names = ['B', 'G', 'R', 'RE', 'NIR']
            channel_indices = [0, 1, 2, 3, 4]
        elif c_ms == 4:
            has_blue = False
            band_names = ['G', 'R', 'RE', 'NIR']
            channel_indices = [1, 2, 3, 4]
        else:
            raise ValueError(f"MS data must have 4 or 5 channels, got {c_ms}")
        
        return has_blue, band_names, channel_indices
    
    def forward(
        self,
        rgb: Optional[torch.Tensor] = None,
        ms: Optional[torch.Tensor] = None,
        return_dict: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for the tokenizer"""
        output_tokens = {}
        
        if rgb is not None:
            b, c, h, w = rgb.shape
            assert c == 3, f"RGB must have 3 channels, got {c}"
            
            for i, band in enumerate(['R', 'G', 'B']):
                band_data = rgb[:, i:i+1, :, :]
                tokens = self.rgb_projections[f'RGB_{band}'](band_data)
                tokens = rearrange(tokens, 'b d h w -> b (h w) d')
                tokens = self.norm(tokens + self.rgb_channel_embed[i])
                output_tokens[f'RGB_{band}'] = tokens
        
        if ms is not None:
            b, c_ms, h, w = ms.shape
            has_blue, band_names, channel_indices = self._detect_ms_configuration(ms)
            
            for data_idx, (band_name, embed_idx) in enumerate(zip(band_names, channel_indices)):
                band_data = ms[:, data_idx:data_idx+1, :, :]
                tokens = self.ms_projections[f'MS_{band_name}'](band_data)
                tokens = rearrange(tokens, 'b d h w -> b (h w) d')
                tokens = self.norm(tokens + self.ms_channel_embed[embed_idx])
                output_tokens[f'MS_{band_name}'] = tokens
        
        return output_tokens


class PatchMerging(nn.Module):
    """
    Patch Merging Layer adapted for variable bands
    """
    def __init__(self, dim: int, norm_layer: nn.Module = nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(2 * dim)

    def forward(self, x: torch.Tensor, band_info: Dict):
        """
        x: B, L, C where L = num_bands * H * W
        """
        B, L, C = x.shape
        num_bands = band_info['num_bands']
        H, W = band_info['spatial_size']
        
        assert L == num_bands * H * W
        assert H % 2 == 0 and W % 2 == 0
        
        x = rearrange(x, 'b (n h w) c -> b n h w c', n=num_bands, h=H, w=W)
        
        # Merge patches
        x0 = x[:, :, 0::2, 0::2, :]  # B N H/2 W/2 C
        x1 = x[:, :, 1::2, 0::2, :]
        x2 = x[:, :, 0::2, 1::2, :]
        x3 = x[:, :, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)  # B N H/2 W/2 4*C
        
        x = rearrange(x, 'b n h w c -> b (n h w) c')
        x = self.reduction(x)
        x = self.norm(x)
        
        # Update band info
        band_info = band_info.copy()
        band_info['spatial_size'] = (H // 2, W // 2)
        
        return x, band_info


class VariableBandWindowAttention(nn.Module):
    """Window-based multi-head self attention with support for variable band inputs"""
    def __init__(
        self,
        dim: int,
        window_size: int,
        num_heads: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.,
        proj_drop: float = 0.,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)
            attn = attn.masked_fill(mask == 0, float('-inf'))

        attn = self.attn_drop(attn.softmax(dim=-1))
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class VariableBandSwinBlock(nn.Module):
    """
    Swin Transformer Block that handles variable band inputs
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int = 7,
        shift_size: int = 0,
        mlp_ratio: float = 4.,
        qkv_bias: bool = True,
        drop: float = 0.,
        attn_drop: float = 0.,
        drop_path: float = 0.,
        norm_layer: nn.Module = nn.LayerNorm,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        self.attn = VariableBandWindowAttention(
            dim,
            window_size=window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x: torch.Tensor, band_info: Dict[str, torch.Tensor]):
        """
        Args:
            x: Input tokens [B, L, C] where L = H/P * W/P * num_bands
            band_info: Dictionary containing band arrangement info
        """
        B, L, C = x.shape
        num_bands = band_info['num_bands']
        H_p, W_p = band_info['spatial_size']
        
        # Reshape to spatial-band structure
        x = rearrange(x, 'b (n h w) c -> b n h w c', n=num_bands, h=H_p, w=W_p)
        
        # Window partition
        if self.shift_size > 0:
            # Shift tokens for SW-MSA
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(2, 3))
        
        # Partition windows
        x_windows = window_partition(x, self.window_size)  # [B*num_windows, num_bands, window_size, window_size, C]
        x_windows = rearrange(x_windows, 'bw n h w c -> bw (n h w) c')
        
        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows)
        
        # Merge windows
        attn_windows = rearrange(
            attn_windows, 'bw (n h w) c -> bw n h w c', 
            n=num_bands, h=self.window_size, w=self.window_size
        )
        x = window_reverse(attn_windows, self.window_size, num_bands, H_p, W_p)
        
        # Reverse shift
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(2, 3))
        
        # Reshape back
        x = rearrange(x, 'b n h w c -> b (n h w) c')
        
        # FFN
        x = x + self.drop_path(self.norm1(x))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        return x

class VariableBandSwinTransformer(nn.Module):
    """
    Swin Transformer V2 adapted for variable band inputs,
    With default size of  Swin-T at 2, 2, 6, 2 and 96 embed dim
    """
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 4,
        embed_dim: int = 96,
        depths: Tuple[int] = (2, 2,6, 2),
        num_heads: Tuple[int] = (3, 6, 12, 24),
        window_size: int = 7,
        mlp_ratio: float = 4.,
        qkv_bias: bool = True,
        drop_rate: float = 0.,
        attn_drop_rate: float = 0.,
        drop_path_rate: float = 0.1,
        norm_layer: nn.Module = nn.LayerNorm,
        **kwargs
    ):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        
        # UAV-specific tokenizer
        self.patch_embed = UAVChannelGroupTokenizer(
            embed_dim=embed_dim,
            patch_size=patch_size,
            norm_layer=norm_layer,
            use_channel_embeds=True,
            project_per_band=True
        )
        
        # Stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        # Build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
            )
            self.layers.append(layer)
        
        self.norm = norm_layer(self.num_features)
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def forward(self, rgb: Optional[torch.Tensor] = None, ms: Optional[torch.Tensor] = None):
        """
        Forward pass handling variable band inputs
        """
        # Get tokens for each band
        tokens_dict = self.patch_embed(rgb=rgb, ms=ms, return_dict=True)
        
        # Prepare band information
        band_names = list(tokens_dict.keys())
        num_bands = len(band_names)
        
        # Stack all band tokens
        x = torch.cat([tokens_dict[name] for name in band_names], dim=1)
        B, L_total, C = x.shape
        L_per_band = L_total // num_bands
        H_p = W_p = int(np.sqrt(L_per_band))
        
        band_info = {
            'num_bands': num_bands,
            'spatial_size': (H_p, W_p),
            'band_names': band_names
        }
        
        # Process through layers
        for layer in self.layers:
            x, band_info = layer(x, band_info)
            
        x = self.norm(x)
        
        # Reshape to extract band and spatial dimensions
        H_final, W_final = band_info['spatial_size']
        x = rearrange(x, 'b (n h w) c -> b n (h w) c', n=num_bands, h=H_final, w=W_final)
        
        # Pool across spatial dimension for each band
        x_band_features = torch.mean(x, dim=2)  # [B, num_bands, C]
        # Return both the features and band information
        return x_band_features, band_info


class BasicLayer(nn.Module):
    """
    A basic Swin Transformer layer for one stage
    """
    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float = 4.,
        qkv_bias: bool = True,
        drop: float = 0.,
        attn_drop: float = 0.,
        drop_path: float = 0.,
        norm_layer: nn.Module = nn.LayerNorm,
        downsample: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.depth = depth
        
        # Build blocks
        self.blocks = nn.ModuleList([
            VariableBandSwinBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
            )
            for i in range(depth)
        ])
        
        # Patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None
    
    def forward(self, x: torch.Tensor, band_info: Dict):
        """Forward function."""
        for blk in self.blocks:
            x = blk(x, band_info)
            
        if self.downsample is not None:
            x, band_info = self.downsample(x, band_info)
            
        return x, band_info
