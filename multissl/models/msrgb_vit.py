import collections.abc
import itertools
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import Tensor, vmap


def to_2tuple(x: Any) -> Tuple:
    """Convert x to a tuple of length 2."""
    if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
        return tuple(x)
    return tuple(itertools.repeat(x, 2))


class FlexiPatchEmbed(nn.Module):
    """
    Flexible patch embedding that can handle different patch sizes
    during inference than what was used during training.
    """
    def __init__(
        self,
        patch_size: Union[int, Tuple[int, int]],
        in_chans: int = 3,
        embed_dim: int = 128,
        norm_layer: Optional[nn.Module] = None,
        bias: bool = True,
        patch_size_seq: Sequence[int] = (1, 2, 3, 4, 5, 6, 7, 8),
        interpolation: str = "bicubic",
        antialias: bool = True,
    ) -> None:
        super().__init__()

        self.patch_size = to_2tuple(patch_size)
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        # For BCHW input, this is already in the right format
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=bias,
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

        # Flexi specific attributes
        self.interpolation = interpolation
        self.antialias = antialias
        self.patch_size_seq = patch_size_seq

        # Pre-calculate pinvs
        self.pinvs = self._cache_pinvs()

    def _cache_pinvs(self) -> dict:
        """Pre-calculate all pinv matrices"""
        pinvs = {}
        for ps in self.patch_size_seq:
            tuple_ps = to_2tuple(ps)
            pinvs[tuple_ps] = self._calculate_pinv(self.patch_size, tuple_ps)
        return pinvs

    def _resize(self, x: Tensor, shape: Tuple[int, int]) -> Tensor:
        """Resize a tensor to the given shape"""
        x_resized = F.interpolate(
            x[None, None, ...],
            shape,
            mode=self.interpolation,
            antialias=self.antialias,
        )
        return x_resized[0, 0, ...]

    def _calculate_pinv(self, old_shape: Tuple[int, int], new_shape: Tuple[int, int]) -> Tensor:
        """Calculate the pseudo-inverse matrix for patch resizing"""
        mat = []
        for i in range(np.prod(old_shape)):
            basis_vec = torch.zeros(old_shape)
            basis_vec[np.unravel_index(i, old_shape)] = 1.0
            mat.append(self._resize(basis_vec, new_shape).reshape(-1))
        resize_matrix = torch.stack(mat)
        return torch.linalg.pinv(resize_matrix)

    def resize_patch_embed(self, patch_embed: Tensor, new_patch_size: Tuple[int, int]):
        """Resize patch_embed to target resolution via pseudo-inverse resizing"""
        # Return original kernel if no resize is necessary
        if self.patch_size == new_patch_size:
            return patch_embed

        # Calculate pseudo-inverse of resize matrix
        if new_patch_size not in self.pinvs:
            self.pinvs[new_patch_size] = self._calculate_pinv(self.patch_size, new_patch_size)
        pinv = self.pinvs[new_patch_size]
        pinv = pinv.to(patch_embed.device)

        def resample_patch_embed(patch_embed: Tensor):
            h, w = new_patch_size
            resampled_kernel = pinv @ patch_embed.reshape(-1)
            return rearrange(resampled_kernel, "(h w) -> h w", h=h, w=w)

        v_resample_patch_embed = vmap(vmap(resample_patch_embed, 0, 0), 1, 1)
        return v_resample_patch_embed(patch_embed)

    def forward(
        self,
        x: Tensor,
        patch_size: Optional[Union[int, Tuple[int, int]]] = None,
    ) -> Tensor:
        """
        Forward pass for the FlexiPatchEmbed module.
        Args:
            x: Input tensor of shape [B, C, H, W]
            patch_size: Optional patch size to use, defaults to self.patch_size
        Returns:
            Embedded patches tensor
        """
        # x already in [B, C, H, W] format
        
        # Use base patch size if not specified
        if not patch_size:
            patch_size = self.patch_size
        patch_size = to_2tuple(patch_size)

        # Resize conv weights if needed
        if patch_size == self.patch_size:
            weight = self.proj.weight
        else:
            weight = self.resize_patch_embed(self.proj.weight, patch_size)
            
        # Apply conv with resized weights
        x = F.conv2d(x, weight, bias=self.proj.bias, stride=patch_size)
        
        # Convert to [B, H, W, C] format for the ViT
        x = rearrange(x, "b c h w -> b h w c")
        x = self.norm(x)
        
        return x


class MSRGBTokenizer(nn.Module):
    """
    Tokenizer for MS and RGB imagery with Galileo-inspired channel grouping
    """
    def __init__(
        self,
        embed_dim: int = 128,
        patch_size: int = 4,
        norm_layer: Optional[nn.Module] = nn.LayerNorm,
        use_channel_embeds: bool = True,
    ):
        super().__init__()
        
        # Define channel indices for each band
        self.CHANNEL_GROUPS = {
            # MS Channels (5 total bands)
            "MS_BLUE": [0],      # MS Blue 
            "MS_GREEN": [1],     # MS Green
            "MS_RED": [2],       # MS Red
            "MS_RE": [3],        # Red Edge
            "MS_NIR": [4],       # Near Infrared
            
            # RGB Channels (3 total bands)
            "RGB_R": [0],        # RGB Red
            "RGB_G": [1],        # RGB Green
            "RGB_B": [2],        # RGB Blue
        }
        
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        
        # Create patch embeddings for each channel group
        self.ms_embed = nn.ModuleDict({
            group_name: FlexiPatchEmbed(
                in_chans=len(channels),
                embed_dim=embed_dim,
                patch_size=patch_size,
                norm_layer=norm_layer
            )
            for group_name, channels in self.CHANNEL_GROUPS.items()
            if group_name.startswith("MS_")
        })
        
        self.rgb_embed = nn.ModuleDict({
            group_name: FlexiPatchEmbed(
                in_chans=len(channels),
                embed_dim=embed_dim,
                patch_size=patch_size,
                norm_layer=norm_layer
            )
            for group_name, channels in self.CHANNEL_GROUPS.items()
            if group_name.startswith("RGB_")
        })
        
        # Channel embeddings to distinguish between different bands
        if use_channel_embeds:
            self.ms_channel_embed = nn.Parameter(
                torch.zeros(len([g for g in self.CHANNEL_GROUPS if g.startswith("MS_")]), embed_dim // 4),
                requires_grad=True
            )
            self.rgb_channel_embed = nn.Parameter(
                torch.zeros(len([g for g in self.CHANNEL_GROUPS if g.startswith("RGB_")]), embed_dim // 4),
                requires_grad=True
            )
            nn.init.normal_(self.ms_channel_embed, std=0.02)
            nn.init.normal_(self.rgb_channel_embed, std=0.02)
        else:
            self.register_buffer('ms_channel_embed', torch.zeros(len([g for g in self.CHANNEL_GROUPS if g.startswith("MS_")]), embed_dim // 4))
            self.register_buffer('rgb_channel_embed', torch.zeros(len([g for g in self.CHANNEL_GROUPS if g.startswith("RGB_")]), embed_dim // 4))
            
        # Positional embeddings to encode spatial information
        self.spatial_embed = nn.Parameter(
            torch.zeros(1, embed_dim // 2),
            requires_grad=True
        )
        nn.init.normal_(self.spatial_embed, std=0.02)

    def extract_single_channels(self, x: torch.Tensor, prefix: str) -> Dict[str, torch.Tensor]:
        """
        Extract single-channel tensors from multi-channel input
        Args:
            x: Input tensor of shape [B, C, H, W]
            prefix: 'MS_' or 'RGB_' to indicate which type
        Returns:
            Dict of band name to single-channel tensor
        """
        result = {}
        groups = [name for name in self.CHANNEL_GROUPS.keys() if name.startswith(prefix)]
        
        for group_name in groups:
            channel_idx = self.CHANNEL_GROUPS[group_name][0]
            if prefix == 'MS_' and channel_idx < x.shape[1]:
                result[group_name] = x[:, channel_idx:channel_idx+1]  # Keep dim for conv
            elif prefix == 'RGB_' and channel_idx < x.shape[1]:
                result[group_name] = x[:, channel_idx:channel_idx+1]  # Keep dim for conv
                
        return result

    def get_2d_sincos_pos_embed(self, embed_dim, grid_size):
        """
        Create 2D sinusoidal positional embeddings for spatial positions
        """
        device = self.spatial_embed.device  # Get device from a parameter tensor
        
        grid_h = torch.arange(grid_size, dtype=torch.float32, device=device)
        grid_w = torch.arange(grid_size, dtype=torch.float32, device=device)
        grid = torch.meshgrid(grid_h, grid_w, indexing="ij")
        grid = torch.stack(grid, dim=0)
        
        grid = grid.reshape([2, 1, grid_size, grid_size])
        pos_embed = self._get_2d_sincos_pos_embed_from_grid(embed_dim, grid[0], grid[1])
        return pos_embed
        
    def _get_2d_sincos_pos_embed_from_grid(self, embed_dim, grid_h, grid_w):
        assert embed_dim % 2 == 0
        
        # Use half of dimensions to encode height
        emb_h = self._get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid_h)
        # Use half of dimensions to encode width
        emb_w = self._get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid_w)
        
        emb = torch.cat([emb_h, emb_w], dim=-1)
        return emb
    
    def _get_1d_sincos_pos_embed_from_grid(self, embed_dim, pos):
        """
        pos: [H, W]
        output: [H, W, embed_dim]
        """
        device = pos.device  # Get device from input position tensor
        
        assert embed_dim % 2 == 0
        omega = torch.arange(embed_dim // 2, dtype=torch.float32, device=device) / embed_dim
        omega = 1.0 / (10000 ** omega)  # [embed_dim//2]
        
        pos = pos.flatten()  # [H*W]
        out = torch.einsum('i,j->ij', pos, omega)  # outer product
        
        emb_sin = torch.sin(out)
        emb_cos = torch.cos(out)
        
        emb = torch.cat([emb_sin, emb_cos], dim=-1)
        return emb
    
    def apply_positional_encoding(self, tokens, token_type, h, w):
        """
        Apply positional encoding to tokens
        Args:
            tokens: Token embeddings
            token_type: 'ms' or 'rgb'
            h, w: Spatial dimensions after patching
        """
        b = tokens.shape[0]
        channels = tokens.shape[-2]
        device = tokens.device  # Get the device from the input tokens
        
        # Get channel embeddings
        if token_type == 'ms':
            channel_emb = repeat(self.ms_channel_embed, 'c d -> b h w c d', b=b, h=h, w=w)
        else:
            channel_emb = repeat(self.rgb_channel_embed, 'c d -> b h w c d', b=b, h=h, w=w)
            
        # Get spatial embeddings and ensure they're on the correct device
        spatial_emb = self.get_2d_sincos_pos_embed(self.embed_dim // 2, h)
        spatial_emb = rearrange(spatial_emb, '(h w) d -> h w d', h=h, w=w)
        spatial_emb = repeat(spatial_emb, 'h w d -> b h w c d', b=b, c=channels)
        
        # Move spatial_emb to the same device as tokens
        spatial_emb = spatial_emb.to(device)
        
        # Create zeros_pad tensor directly on the correct device
        zeros_pad = torch.zeros(
            b, h, w, channels, 
            self.embed_dim - channel_emb.shape[-1] - spatial_emb.shape[-1], 
            device=device
        )
        
        # Now all tensors are on the same device
        combined_emb = torch.cat([channel_emb, spatial_emb, zeros_pad], dim=-1)
        return tokens + combined_emb
        
    def forward(
        self,
        ms: Optional[torch.Tensor] = None,
        rgb: Optional[torch.Tensor] = None,
        patch_size: Optional[int] = None,
        return_dict: bool = False
    ) -> Union[Dict[str, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Tokenize MS and/or RGB inputs
        
        Args:
            ms: MS input tensor of shape [B, 5, H, W]
            rgb: RGB input tensor of shape [B, 3, H, W]
            patch_size: Optional patch size override
            return_dict: If True, return a dictionary of tokens; if False, return combined tokens
            
        Returns:
            If return_dict=True: Dict of token name to token tensor
            If return_dict=False: (ms_tokens, rgb_tokens, mask)
        """
        patch_size = patch_size or self.patch_size
        tokens_dict = {}
        
        if ms is not None:
            ms_channels = self.extract_single_channels(ms, 'MS_')
            ms_tokens_list = []
            
            for idx, (band_name, band_data) in enumerate(ms_channels.items()):
                tokens = self.ms_embed[band_name](band_data, patch_size=patch_size)
                if return_dict:
                    tokens_dict[band_name] = tokens
                else:
                    ms_tokens_list.append(tokens)
            
            if not return_dict:
                # Get spatial dimensions after patching
                h, w = ms_tokens_list[0].shape[1:3]
                
                # Stack tokens along channel dimension
                ms_tokens = torch.stack(ms_tokens_list, dim=3)  # [B, H, W, C, D]
                
                # Apply positional and channel embeddings
                ms_tokens = self.apply_positional_encoding(ms_tokens, 'ms', h, w)
        
        if rgb is not None:
            rgb_channels = self.extract_single_channels(rgb, 'RGB_')
            rgb_tokens_list = []
            
            for idx, (band_name, band_data) in enumerate(rgb_channels.items()):
                tokens = self.rgb_embed[band_name](band_data, patch_size=patch_size)
                if return_dict:
                    tokens_dict[band_name] = tokens
                else:
                    rgb_tokens_list.append(tokens)
            
            if not return_dict:
                # Get spatial dimensions after patching
                h, w = rgb_tokens_list[0].shape[1:3]
                
                # Stack tokens along channel dimension
                rgb_tokens = torch.stack(rgb_tokens_list, dim=3)  # [B, H, W, C, D]
                
                # Apply positional and channel embeddings
                rgb_tokens = self.apply_positional_encoding(rgb_tokens, 'rgb', h, w)
        
        if return_dict:
            return tokens_dict
        else:
            # Create masks indicating which modalities are present
            ms_mask = torch.zeros((1,), dtype=torch.bool) if ms is None else torch.ones((1,), dtype=torch.bool)
            rgb_mask = torch.zeros((1,), dtype=torch.bool) if rgb is None else torch.ones((1,), dtype=torch.bool)
            mask = torch.cat([ms_mask, rgb_mask])
            
            # Return tensors for each modality (or None if not present)
            ms_output = ms_tokens if ms is not None else None
            rgb_output = rgb_tokens if rgb is not None else None
            
            return ms_output, rgb_output, mask

import torch.nn as nn
from einops import rearrange
import torch.utils.checkpoint as checkpoint

class Attention(nn.Module):
    """
    Multi-head attention with optional cross-attention capability
    """
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
        cross_attn=False,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.cross_attn = cross_attn

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, y=None, attn_mask=None):
        """
        Args:
            x: Input tensor for queries
            y: Optional tensor for keys and values (for cross-attention)
            attn_mask: Optional attention mask
        """
        B, N, C = x.shape

        q = self.q(x)
        
        if y is None:
            assert not self.cross_attn
            k = self.k(x)
            v = self.v(x)
        else:
            assert self.cross_attn
            k = self.k(y)
            v = self.v(y)

        q = rearrange(q, "b n (h d) -> b h n d", h=self.num_heads)
        k = rearrange(k, "b n (h d) -> b h n d", h=self.num_heads)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.num_heads)

        # Compute attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        if attn_mask is not None:
            # Expand mask for multi-head attention
            attn_mask = attn_mask.unsqueeze(1).unsqueeze(1)
            attn = attn.masked_fill(~attn_mask, float('-inf'))
            
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v

        x = rearrange(x, "b h n d -> b n (h d)")
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    """MLP block used in Vision Transformer"""
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
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


class ViTBlock(nn.Module):
    """Vision Transformer Block with support for cross-attention"""
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop=0.0,
        attn_drop=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        cross_attn=False,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            cross_attn=cross_attn,
        )
        
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x, y=None, attn_mask=None):
        x = x + self.attn(self.norm1(x), y, attn_mask)
        x = x + self.mlp(self.norm2(x))
        return x


class MSRGBViT(nn.Module):
    """
    Vision Transformer for MS and RGB data with flexible patch embedding
    """
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        norm_layer=nn.LayerNorm,
        use_channel_embeds=True,
        pool_type="mean",
    ):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.pool_type = pool_type
        
        # Tokenizer for MS and RGB data
        self.tokenizer = MSRGBTokenizer(
            embed_dim=embed_dim,
            patch_size=patch_size,
            norm_layer=norm_layer,
            use_channel_embeds=use_channel_embeds,
        )
        
        # Transformer encoder blocks
        self.blocks = nn.ModuleList([
            ViTBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                norm_layer=norm_layer,
            )
            for _ in range(depth)
        ])
        
        self.norm = norm_layer(embed_dim)
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def _prepare_tokens(self, ms_tokens, rgb_tokens, mask):
        """
        Prepare tokens from different modalities for joint processing
        
        Args:
            ms_tokens: MS tokens of shape [B, H, W, C_ms, D] or None
            rgb_tokens: RGB tokens of shape [B, H, W, C_rgb, D] or None
            mask: Boolean mask indicating which modalities are present
            
        Returns:
            tokens: Combined tokens for processing
            attn_mask: Attention mask for valid tokens
        """
        tokens_list = []
        batch_size = None
        
        # Add present modalities to the tokens list
        if ms_tokens is not None:
            batch_size = ms_tokens.shape[0]
            tokens_list.append(rearrange(ms_tokens, "b h w c d -> b (h w c) d"))
            
        if rgb_tokens is not None:
            batch_size = rgb_tokens.shape[0]
            tokens_list.append(rearrange(rgb_tokens, "b h w c d -> b (h w c) d"))
        
        # Concatenate tokens from all modalities
        tokens = torch.cat(tokens_list, dim=1)
        
        # Create attention mask that allows tokens to attend to all tokens
        # from present modalities
        attn_mask = torch.ones((batch_size, tokens.shape[1]), dtype=torch.bool, device=tokens.device)
        
        return tokens, attn_mask
    
    def forward_features(self, ms=None, rgb=None, patch_size=None):
        """
        Forward pass through the feature extraction part of the model
        
        Args:
            ms: MS input tensor of shape [B, 5, H, W]
            rgb: RGB input tensor of shape [B, 3, H, W]
            patch_size: Optional patch size override
            
        Returns:
            features: Extracted features
        """
        # Get tokens from tokenizer
        ms_tokens, rgb_tokens, mask = self.tokenizer(ms=ms, rgb=rgb, patch_size=patch_size)
        
        # Prepare tokens for the transformer
        tokens, attn_mask = self._prepare_tokens(ms_tokens, rgb_tokens, mask)
        
        # Pass through transformer blocks
        for block in self.blocks:
            tokens = block(tokens, attn_mask=attn_mask)
            
        # Apply final normalization
        tokens = self.norm(tokens)
        
        # Pool tokens based on specified strategy
        if self.pool_type == "mean":
            # Global average pooling
            features = tokens.mean(dim=1)
        elif self.pool_type == "cls":
            # Use first token as class token (not implemented in this version)
            features = tokens[:, 0]
        else:
            raise ValueError(f"Unsupported pooling type: {self.pool_type}")
            
        return features
    
    def forward(self, ms=None, rgb=None, patch_size=None):
        """
        Forward pass through the model
        
        Args:
            ms: MS input tensor of shape [B, 5, H, W]
            rgb: RGB input tensor of shape [B, 3, H, W]
            patch_size: Optional patch size override
            
        Returns:
            features: Extracted features for downstream tasks
        """
        features = self.forward_features(ms=ms, rgb=rgb, patch_size=patch_size)
        return features