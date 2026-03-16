"""
SVTR (Scene Text Recognition with a Single Visual model) Neck
Khối Transformer nhỏ gọn để trộn thông tin toàn cục và cục bộ
Thay thế BiLSTM nhưng nhẹ và hiệu quả hơn
"""

import torch
import torch.nn as nn
import math


class ConvBNLayer(nn.Module):
    """Conv + BatchNorm + Activation"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, 
                 padding=0, groups=1, act='gelu'):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, 
                             padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        
        if act == 'gelu':
            self.act = nn.GELU()
        elif act == 'relu':
            self.act = nn.ReLU(inplace=True)
        else:
            self.act = nn.Identity()
    
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class MixingBlock(nn.Module):
    """
    Mixing Block trong SVTR - kết hợp local và global information
    Sử dụng attention mechanism nhẹ
    """
    def __init__(self, dim, num_heads=8, mixer='Global', 
                 local_k=(1, 11), dropout=0.1):  # (1, k) cho sequence 1D
        super().__init__()
        
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # Mixing layer (Attention hoặc Conv)
        if mixer == 'Global':
            self.mixer = GlobalAttention(dim, num_heads, dropout)
        elif mixer == 'Local':
            self.mixer = LocalAttention(dim, num_heads, local_k, dropout)
        else:
            self.mixer = ConvMixing(dim, local_k)
        
        # Feed forward
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        # x: (N, L, C)
        x = x + self.mixer(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class GlobalAttention(nn.Module):
    """
    Global Multi-head Self-Attention
    """
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        assert dim % num_heads == 0
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)
    
    def forward(self, x):
        # x: (N, L, C)
        N, L, C = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(N, L, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, N, H, L, D)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(N, L, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class LocalAttention(nn.Module):
    """
    Local windowed attention để tập trung vào local patterns
    Optimized cho sequence 1D: kernel (1, k) thay vì (h, k)
    """
    def __init__(self, dim, num_heads=8, window_size=(1, 11), dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        
        # Depthwise conv cho sequence 1D: kernel (1, 11), padding (0, 5)
        self.dwconv = nn.Conv2d(dim, dim, window_size, 1, 
                               (0, window_size[1]//2),  # Padding (0, k//2) cho 1D
                               groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.pwconv = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)
    
    def forward(self, x):
        # x: (N, L, C) - sequence 1D sau khi flatten height
        N, L, C = x.shape
        
        # Reshape về (N, C, 1, L) cho Conv2d - height=1 vì là sequence 1D
        H = 1
        W = L
        
        x_2d = x.transpose(1, 2).reshape(N, C, H, W)  # (N, C, 1, L)
        x_2d = self.dwconv(x_2d)  # Conv2d với kernel (1, 11)
        x_2d = x_2d.reshape(N, C, L).transpose(1, 2)  # Back to (N, L, C)
        
        x = self.norm(x_2d)
        x = self.pwconv(x)
        x = self.drop(x)
        
        return x


class ConvMixing(nn.Module):
    """
    Convolution-based mixing (nhẹ nhất)
    Optimized cho sequence 1D: kernel (1, k)
    """
    def __init__(self, dim, kernel_size=(1, 11)):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size, 1, 
                               (0, kernel_size[1]//2),  # Padding (0, k//2) cho 1D
                               groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.pwconv = nn.Linear(dim, dim)
    
    def forward(self, x):
        # x: (N, L, C) - sequence 1D
        N, L, C = x.shape
        H = 1  # Height = 1 vì sequence 1D
        W = L  # Width = sequence length
        
        x_2d = x.transpose(1, 2).reshape(N, C, H, W)  # (N, C, 1, L)
        x_2d = self.dwconv(x_2d)  # Conv2d kernel (1, 11)
        x_2d = x_2d.reshape(N, C, L).transpose(1, 2)  # (N, L, C)
        
        x = self.norm(x_2d)
        x = self.pwconv(x)
        
        return x


class SVTRTinyNeck(nn.Module):
    """
    SVTR-Tiny Neck: khối xử lý sau CNN backbone
    Kết hợp local và global information qua các Mixing blocks
    
    CRITICAL: Neck xử lý SEQUENCE 1D (height đã flatten vào channels ở backbone)
    - Input từ backbone: (W, N, C) với W = sequence length
    - Không có dimension height riêng biệt
    - Local convolution dùng kernel (1, k) thay vì (h, k)
    
    Input: (W, N, C) từ backbone (height đã flatten)
    Output: (W, N, C) cho CTC head
    """
    def __init__(self, in_channels=256, out_channels=256, 
                 depth=2, num_heads=8, mixer_types=['Local', 'Global'],
                 dropout=0.1):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Input projection nếu cần
        if in_channels != out_channels:
            self.input_proj = nn.Linear(in_channels, out_channels)
        else:
            self.input_proj = nn.Identity()
        
        # Các Mixing blocks
        self.blocks = nn.ModuleList([
            MixingBlock(
                dim=out_channels,
                num_heads=num_heads,
                mixer=mixer_types[i % len(mixer_types)],
                dropout=dropout
            )
            for i in range(depth)
        ])
        
        self.norm = nn.LayerNorm(out_channels)
    
    def forward(self, x):
        """
        Args:
            x: (W, N, C) từ backbone - sequence 1D (height đã flatten)
        Returns:
            out: (W, N, C)
        """
        # Validate input - Neck chỉ xử lý sequence 1D
        if x.dim() == 4:
            raise ValueError(
                f"SVTRTinyNeck expects 3D input (W, N, C), got 4D tensor {x.shape}. "
                f"Height should be flattened into channels before passing to neck!"
            )
        
        assert x.dim() == 3, f"Expected 3D input (W, N, C), got {x.dim()}D tensor"
        
        # x shape: (W, N, C) -> (N, W, C) cho các blocks
        x = x.transpose(0, 1)  # (N, W, C)
        
        # Input projection
        x = self.input_proj(x)
        
        # Apply mixing blocks
        for block in self.blocks:
            x = block(x)
        
        # Final norm
        x = self.norm(x)
        
        # Back to (W, N, C)
        x = x.transpose(0, 1)
        
        return x


def svtr_tiny_neck(in_channels=256, out_channels=256, depth=2, 
                  num_heads=8, dropout=0.1):
    """
    Factory function cho SVTR-Tiny neck
    
    Args:
        in_channels: số channel đầu vào từ backbone
        out_channels: số channel đầu ra
        depth: số lượng mixing blocks (2-4 cho tiny)
        num_heads: số attention heads
        dropout: dropout rate
    """
    return SVTRTinyNeck(
        in_channels=in_channels,
        out_channels=out_channels,
        depth=depth,
        num_heads=num_heads,
        mixer_types=['Local', 'Global'],  # Xen kẽ Local và Global
        dropout=dropout
    )
