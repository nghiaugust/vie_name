"""
MobileNet-SVTR-CTC Model
Kiến trúc tối ưu cho nhận diện tên tiếng Việt:
- Backbone: MobileNetV3-Large (nhanh, nhẹ)
- Neck: SVTR-Tiny (mixing local & global info)
- Head: CTC (alignment-free)

Ưu điểm:
- Nhanh: MobileNetV3 rất nhẹ
- Chính xác: SVTR trộn thông tin tốt, CTC không cần alignment
- Phù hợp tên tiếng Việt: stride tùy chỉnh giữ được thông tin sequence
"""

import torch
from torch import nn
from vietocr.model.backbone.mobilenetv3 import MobileNetV3Large
from vietocr.model.backbone.svtr_neck import SVTRTinyNeck
from vietocr.model.seqmodel.ctc import CTCSeqModel


class MobileNetSVTRCTC(nn.Module):
    """
    Complete model: MobileNetV3 + SVTR Neck + CTC Head
    """
    
    def __init__(
        self,
        vocab_size,
        backbone_hidden=256,
        neck_hidden=256,
        svtr_depth=2,
        svtr_heads=8,
        dropout=0.1,
        pretrained=False,
        **kwargs
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        
        # Backbone: MobileNetV3-Large
        self.backbone = MobileNetV3Large(
            hidden=backbone_hidden,
            pretrained=pretrained,
            dropout=dropout
        )
        
        # Tính toán H_feature dựa trên stride pattern:
        # First conv: stride (2,2) -> H/2
        # Stage1 layer1: stride (2,2) -> H/4  
        # Stage2 layer3: stride (2,1) -> H/8
        # Stage2 layer6: stride (2,1) -> H/16
        # => Với input H=32, output H'=2
        self.H_feature = 2  # 32 // 16 = 2
        
        # Channel sau khi flatten height: C * H'
        self.neck_in_channels = backbone_hidden * self.H_feature
        
        # Projector: Giảm chiều từ C*H' xuống neck_hidden
        self.projector = nn.Sequential(
            nn.Linear(self.neck_in_channels, neck_hidden),
            nn.LayerNorm(neck_hidden),
            nn.Dropout(dropout)
        )
        
        # Neck: SVTR-Tiny
        self.neck = SVTRTinyNeck(
            in_channels=neck_hidden,
            out_channels=neck_hidden,
            depth=svtr_depth,
            num_heads=svtr_heads,
            dropout=dropout
        )
        
        # Head: CTC
        self.head = CTCSeqModel(
            vocab_size=vocab_size,
            d_model=neck_hidden,
            dropout=dropout
        )
    
    def forward(self, img, tgt_input=None, tgt_key_padding_mask=None):
        """
        Forward pass
        
        Args:
            img: (N, C, H, W) - input images
            tgt_input: (T, N) - target (not used in CTC, kept for compatibility)
            tgt_key_padding_mask: (N, T) - padding mask (not used)
        
        Returns:
            logits: (N, T, vocab_size)
        """
        # 1. Backbone: Extract features
        features = self.backbone(img)  # (N, C, H', W')
        
        # 2. Flatten Height into Channels (CRITICAL for CTC)
        # Shape: (N, C, H, W) -> (N, W, C*H)
        # Giữ W làm sequence length, gộp H vào channels
        N, C, H, W = features.shape
        features = features.permute(0, 3, 1, 2).contiguous()  # (N, W, C, H)
        features = features.view(N, W, C * H)  # (N, W, C*H)
        
        # 3. Projection: Giảm chiều C*H xuống neck_hidden
        features = self.projector(features)  # (N, W, neck_hidden)
        
        # 4. Chuẩn bị cho SVTR: (N, W, C) -> (W, N, C)
        features = features.transpose(0, 1)  # (W, N, neck_hidden)
        
        # 5. SVTR Neck: Mix local & global info
        features = self.neck(features)  # (W, N, neck_hidden)
        
        # 6. CTC Head: Predict logits
        logits = self.head(features)  # (N, W, vocab_size)
        
        return logits
    
    def forward_encoder(self, img):
        """
        Chỉ chạy encoder (để inference hoặc extract features)
        
        Args:
            img: (N, C, H, W)
        
        Returns:
            features: (W, N, C)
        """
        features = self.backbone(img)
        features = self.neck(features)
        return features
    
    def decode(self, logits, method='greedy', beam_width=5):
        """
        Decode logits to sequences
        
        Args:
            logits: (N, T, vocab_size)
            method: 'greedy' or 'beam_search'
            beam_width: beam width for beam search
        
        Returns:
            decoded: list of decoded sequences
        """
        return self.head.decode(logits, method=method, beam_width=beam_width)
    
    def compute_loss(self, logits, targets, input_lengths, target_lengths):
        """
        Compute CTC loss
        
        Args:
            logits: (N, T, vocab_size)
            targets: (N, S) - target sequences
            input_lengths: (N,) - actual input lengths
            target_lengths: (N,) - actual target lengths
        
        Returns:
            loss: scalar
        """
        return self.head.compute_loss(logits, targets, input_lengths, target_lengths)


class MobileNetSVTRCTCLight(nn.Module):
    """
    Phiên bản nhẹ hơn với ít parameters hơn
    Dùng cho deployment hoặc training nhanh
    """
    
    def __init__(
        self,
        vocab_size,
        backbone_hidden=128,  # Giảm từ 256
        neck_hidden=128,
        svtr_depth=1,  # Giảm từ 2
        svtr_heads=4,  # Giảm từ 8
        dropout=0.1,
        **kwargs
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        
        # Backbone
        self.backbone = MobileNetV3Large(
            hidden=backbone_hidden,
            pretrained=False,
            dropout=dropout
        )
        
        # Tính H_feature (giống standard version)
        self.H_feature = 2  # 32 // 16 = 2
        self.neck_in_channels = backbone_hidden * self.H_feature
        
        # Projector
        self.projector = nn.Sequential(
            nn.Linear(self.neck_in_channels, neck_hidden),
            nn.LayerNorm(neck_hidden),
            nn.Dropout(dropout)
        )
        
        # Neck - shallow SVTR
        self.neck = SVTRTinyNeck(
            in_channels=neck_hidden,
            out_channels=neck_hidden,
            depth=svtr_depth,
            num_heads=svtr_heads,
            dropout=dropout
        )
        
        # Head - CTC
        self.head = CTCSeqModel(
            vocab_size=vocab_size,
            d_model=neck_hidden,
            dropout=dropout
        )
    
    def forward(self, img, tgt_input=None, tgt_key_padding_mask=None):
        # 1. Backbone
        features = self.backbone(img)  # (N, C, H', W')
        
        # 2. Flatten Height into Channels
        N, C, H, W = features.shape
        features = features.permute(0, 3, 1, 2).contiguous()  # (N, W, C, H)
        features = features.view(N, W, C * H)  # (N, W, C*H)
        
        # 3. Projection
        features = self.projector(features)  # (N, W, neck_hidden)
        
        # 4. Format cho SVTR
        features = features.transpose(0, 1)  # (W, N, neck_hidden)
        
        # 5. SVTR Neck
        features = self.neck(features)  # (W, N, neck_hidden)
        
        # 6. CTC Head
        logits = self.head(features)  # (N, W, vocab_size)
        
        return logits
    
    def forward_encoder(self, img):
        features = self.backbone(img)
        features = self.neck(features)
        return features
    
    def decode(self, logits, method='greedy', beam_width=5):
        return self.head.decode(logits, method=method, beam_width=beam_width)
    
    def compute_loss(self, logits, targets, input_lengths, target_lengths):
        return self.head.compute_loss(logits, targets, input_lengths, target_lengths)


def mobilenet_svtr_ctc(vocab_size, hidden=256, svtr_depth=2, svtr_heads=8, 
                       dropout=0.1, pretrained=False, **kwargs):
    """
    Factory function cho MobileNet-SVTR-CTC model (standard)
    
    Args:
        vocab_size: kích thước vocabulary
        hidden: số hidden units
        svtr_depth: số SVTR blocks
        svtr_heads: số attention heads
        dropout: dropout rate
        pretrained: sử dụng pretrained backbone
    
    Returns:
        MobileNetSVTRCTC model
    """
    return MobileNetSVTRCTC(
        vocab_size=vocab_size,
        backbone_hidden=hidden,
        neck_hidden=hidden,
        svtr_depth=svtr_depth,
        svtr_heads=svtr_heads,
        dropout=dropout,
        pretrained=pretrained,
        **kwargs
    )


def mobilenet_svtr_ctc_light(vocab_size, hidden=128, svtr_depth=1, svtr_heads=4,
                             dropout=0.1, **kwargs):
    """
    Factory function cho MobileNet-SVTR-CTC model (light version)
    
    Phiên bản nhẹ hơn cho:
    - Training nhanh hơn
    - Inference nhanh hơn
    - Ít memory hơn
    
    Args:
        vocab_size: kích thước vocabulary
        hidden: số hidden units (default 128)
        svtr_depth: số SVTR blocks (default 1)
        svtr_heads: số attention heads (default 4)
        dropout: dropout rate
    
    Returns:
        MobileNetSVTRCTCLight model
    """
    return MobileNetSVTRCTCLight(
        vocab_size=vocab_size,
        backbone_hidden=hidden,
        neck_hidden=hidden,
        svtr_depth=svtr_depth,
        svtr_heads=svtr_heads,
        dropout=dropout,
        **kwargs
    )
