"""
MobileNetV3-Large backbone với stride tùy chỉnh cho OCR tên tiếng Việt
- Early layers: stride (2, 2) - giảm kích thước nhanh
- Middle layers: stride (2, 1) - giữ chiều rộng cho sequence
- Final layers: stride (1, 1) - giữ nguyên độ phân giải
"""

import torch
from torch import nn
import torch.nn.functional as F


def _make_divisible(v, divisor, min_value=None):
    """
    Hàm helper để đảm bảo số channel chia hết cho divisor
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class SqueezeExcitation(nn.Module):
    """SE (Squeeze and Excitation) module"""
    def __init__(self, in_channels, squeeze_channels):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, squeeze_channels, 1)
        self.fc2 = nn.Conv2d(squeeze_channels, in_channels, 1)
    
    def forward(self, x):
        scale = self.avgpool(x)
        scale = F.relu(self.fc1(scale), inplace=True)
        scale = torch.sigmoid(self.fc2(scale))
        return x * scale


class InvertedResidual(nn.Module):
    """
    MobileNetV3 Inverted Residual Block với SE module
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, 
                 expand_ratio, use_se=False, activation='relu'):
        super().__init__()
        
        self.stride = stride
        self.use_res_connect = stride == (1, 1) and in_channels == out_channels
        
        hidden_dim = int(round(in_channels * expand_ratio))
        
        layers = []
        
        # Expand
        if expand_ratio != 1:
            layers.append(nn.Conv2d(in_channels, hidden_dim, 1, 1, 0, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            if activation == 'relu':
                layers.append(nn.ReLU(inplace=True))
            else:  # hardswish
                layers.append(nn.Hardswish(inplace=True))
        
        # Depthwise
        padding = (kernel_size - 1) // 2
        layers.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, 
                               padding, groups=hidden_dim, bias=False))
        layers.append(nn.BatchNorm2d(hidden_dim))
        if activation == 'relu':
            layers.append(nn.ReLU(inplace=True))
        else:
            layers.append(nn.Hardswish(inplace=True))
        
        # SE
        if use_se:
            squeeze_channels = _make_divisible(hidden_dim // 4, 8)
            layers.append(SqueezeExcitation(hidden_dim, squeeze_channels))
        
        # Project
        layers.append(nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV3Large(nn.Module):
    """
    MobileNetV3-Large với stride tùy chỉnh cho OCR
    Input: (N, 3, H, W)
    Output: (W', N, C) where W' là chiều rộng sau khi giảm
    """
    
    def __init__(self, hidden=256, pretrained=False, dropout=0.3):
        super().__init__()
        
        # Cấu hình: [in, out, kernel, stride, expand, se, activation]
        # Chia thành 3 giai đoạn với stride khác nhau
        
        # Stage 1: Early layers - stride (2,2) để giảm nhanh
        stage1_config = [
            [16, 16, 3, (1,1), 1, False, 'relu'],      # layer 0
            [16, 24, 3, (2,2), 4, False, 'relu'],      # layer 1 - stride (2,2)
            [24, 24, 3, (1,1), 3, False, 'relu'],      # layer 2
        ]
        
        # Stage 2: Middle layers - stride (2,1) giữ chiều rộng
        stage2_config = [
            [24, 40, 5, (2,1), 3, True, 'relu'],       # layer 3 - stride (2,1)
            [40, 40, 5, (1,1), 3, True, 'relu'],       # layer 4
            [40, 40, 5, (1,1), 3, True, 'relu'],       # layer 5
            [40, 80, 3, (2,1), 6, False, 'hardswish'], # layer 6 - stride (2,1)
            [80, 80, 3, (1,1), 2.5, False, 'hardswish'],# layer 7
        ]
        
        # Stage 3: Final layers - stride (1,1) giữ nguyên
        stage3_config = [
            [80, 80, 3, (1,1), 2.3, False, 'hardswish'],# layer 8
            [80, 80, 3, (1,1), 2.3, False, 'hardswish'],# layer 9
            [80, 112, 3, (1,1), 6, True, 'hardswish'],  # layer 10
            [112, 112, 3, (1,1), 6, True, 'hardswish'], # layer 11
            [112, 160, 5, (1,1), 6, True, 'hardswish'], # layer 12
            [160, 160, 5, (1,1), 6, True, 'hardswish'], # layer 13
            [160, 160, 5, (1,1), 6, True, 'hardswish'], # layer 14
        ]
        
        # First conv
        first_channels = 16
        self.first_conv = nn.Sequential(
            nn.Conv2d(3, first_channels, 3, (2,2), 1, bias=False),
            nn.BatchNorm2d(first_channels),
            nn.Hardswish(inplace=True)
        )
        
        # Build stages
        self.stage1 = self._make_stage(stage1_config)
        self.stage2 = self._make_stage(stage2_config)
        self.stage3 = self._make_stage(stage3_config)
        
        # Final conv layers
        last_channels = 960
        self.conv_last = nn.Sequential(
            nn.Conv2d(160, last_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(last_channels),
            nn.Hardswish(inplace=True)
        )
        
        # Projection to hidden dimension
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Conv2d(last_channels, hidden, 1, 1, 0)
        
        self._initialize_weights()
    
    def _make_stage(self, config):
        """Tạo một stage từ config"""
        layers = []
        for in_channels, out_channels, kernel, stride, expand, se, act in config:
            layers.append(
                InvertedResidual(in_channels, out_channels, kernel, 
                               stride, expand, se, act)
            )
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Khởi tạo weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """
        Args:
            x: (N, 3, H, W)
        Returns:
            features: (N, hidden, H', W') - raw feature map
        """
        # First conv: H/2, W/2
        x = self.first_conv(x)
        
        # Stage 1: stride (2,2) -> H/4, W/4
        x = self.stage1(x)
        
        # Stage 2: stride (2,1) -> H/16, W/4
        x = self.stage2(x)
        
        # Stage 3: stride (1,1) -> H/16, W/4
        x = self.stage3(x)
        
        # Final conv
        x = self.conv_last(x)
        x = self.dropout(x)
        x = self.proj(x)  # (N, hidden, H', W')
        
        # Trả về raw feature map để xử lý đúng cách ở model level
        # Height sẽ được flatten vào channels, không vào sequence length
        return x


def mobilenetv3_large(hidden=256, pretrained=False, dropout=0.3):
    """
    Factory function để tạo MobileNetV3-Large backbone
    
    Args:
        hidden: số chiều ẩn đầu ra
        pretrained: có dùng pretrained weights không (chưa implement)
        dropout: dropout rate
    
    Returns:
        MobileNetV3Large model
    """
    return MobileNetV3Large(hidden=hidden, pretrained=pretrained, dropout=dropout)
