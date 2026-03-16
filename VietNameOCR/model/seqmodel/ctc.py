"""
CTC (Connectionist Temporal Classification) Head cho OCR
Không cần alignment giữa input và output
Phù hợp cho nhận diện text với độ dài biến đổi
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CTCHead(nn.Module):
    """
    CTC Head cho sequence recognition
    Input: (T, N, C) - sequence features từ encoder/neck
    Output: (T, N, vocab_size) - logits cho mỗi time step
    """
    
    def __init__(self, in_channels, vocab_size, dropout=0.1):
        super().__init__()
        
        self.in_channels = in_channels
        self.vocab_size = vocab_size
        
        # Optional: thêm layer processing trước CTC
        self.pre_proj = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # CTC projection layer
        self.ctc_proj = nn.Linear(in_channels, vocab_size)
        
        self._init_weights()
    
    def _init_weights(self):
        """Khởi tạo weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """
        Args:
            x: (T, N, C) - sequence features
        Returns:
            logits: (T, N, vocab_size)
        """
        # Pre-processing
        x = self.pre_proj(x)
        
        # CTC projection
        logits = self.ctc_proj(x)
        
        return logits
    
    def decode_greedy(self, logits, blank_idx=0):
        """
        Greedy decoding cho CTC
        
        Args:
            logits: (T, N, vocab_size)
            blank_idx: index của blank token
        
        Returns:
            decoded: list of decoded sequences
        """
        # Get predictions
        preds = torch.argmax(logits, dim=-1)  # (T, N)
        preds = preds.transpose(0, 1)  # (N, T)
        
        # Decode each sequence in batch
        decoded = []
        for pred in preds:
            # Remove consecutive duplicates and blanks
            decoded_seq = []
            prev = None
            for token in pred:
                token = token.item()
                if token != blank_idx and token != prev:
                    decoded_seq.append(token)
                prev = token
            decoded.append(decoded_seq)
        
        return decoded
    
    def decode_beam_search(self, logits, beam_width=5, blank_idx=0):
        """
        Beam search decoding cho CTC (tối ưu hơn greedy)
        
        Args:
            logits: (T, N, vocab_size)
            beam_width: số lượng beams
            blank_idx: index của blank token
        
        Returns:
            decoded: list of decoded sequences
        """
        # Log probabilities
        log_probs = F.log_softmax(logits, dim=-1)  # (T, N, V)
        T, N, V = log_probs.shape
        
        # Decode each sequence
        decoded = []
        for n in range(N):
            seq_log_probs = log_probs[:, n, :]  # (T, V)
            best_seq = self._beam_search_single(seq_log_probs, beam_width, blank_idx)
            decoded.append(best_seq)
        
        return decoded
    
    def _beam_search_single(self, log_probs, beam_width, blank_idx):
        """
        Beam search cho một sequence
        
        Args:
            log_probs: (T, V)
            beam_width: số beams
            blank_idx: blank index
        
        Returns:
            best_seq: decoded sequence
        """
        T, V = log_probs.shape
        
        # Initialize beams: (prefix, log_prob)
        beams = [('', 0.0)]  # Start with empty sequence
        
        for t in range(T):
            new_beams = {}
            
            for prefix, log_prob in beams:
                # Get top-k tokens at this timestep
                topk_log_probs, topk_indices = torch.topk(log_probs[t], min(beam_width, V))
                
                for token_log_prob, token_idx in zip(topk_log_probs, topk_indices):
                    token_idx = token_idx.item()
                    token_log_prob = token_log_prob.item()
                    
                    if token_idx == blank_idx:
                        # Blank: không thêm gì
                        new_prefix = prefix
                    else:
                        # Non-blank: thêm token
                        if len(prefix) > 0 and prefix[-1] == token_idx:
                            # Duplicate: merge
                            new_prefix = prefix
                        else:
                            new_prefix = prefix + str(token_idx) + ','
                    
                    # Update beam
                    new_log_prob = log_prob + token_log_prob
                    if new_prefix not in new_beams or new_beams[new_prefix] < new_log_prob:
                        new_beams[new_prefix] = new_log_prob
            
            # Keep top beams
            beams = sorted(new_beams.items(), key=lambda x: x[1], reverse=True)[:beam_width]
        
        # Get best beam
        best_prefix, _ = beams[0]
        if best_prefix:
            best_seq = [int(x) for x in best_prefix.rstrip(',').split(',') if x]
        else:
            best_seq = []
        
        return best_seq


class CTCSeqModel(nn.Module):
    """
    Wrapper cho CTC model - tương thích với VietOCR framework
    """
    
    def __init__(self, vocab_size, d_model=256, dropout=0.1, **kwargs):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        # CTC head
        self.ctc_head = CTCHead(d_model, vocab_size, dropout)
        
        # CTC Loss
        self.ctc_loss = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
    
    def forward(self, src, tgt_input=None, tgt_key_padding_mask=None):
        """
        Forward pass
        
        Args:
            src: (W, N, C) - features từ backbone/neck
            tgt_input: (T, N) - target sequence (for training, not used in CTC)
            tgt_key_padding_mask: (N, T) - padding mask (not used in CTC)
        
        Returns:
            logits: (N, T, vocab_size) - theo format của VietOCR
        """
        # CTC head
        logits = self.ctc_head(src)  # (W, N, vocab_size)
        
        # Reshape to match VietOCR format: (N, T, vocab_size)
        logits = logits.transpose(0, 1)  # (N, W, vocab_size)
        
        return logits
    
    def compute_loss(self, logits, targets, input_lengths, target_lengths):
        """
        Tính CTC loss
        
        Args:
            logits: (N, T, vocab_size)
            targets: (N, S) - target sequences
            input_lengths: (N,) - độ dài thực của input sequences
            target_lengths: (N,) - độ dài thực của target sequences
        
        Returns:
            loss: scalar
        """
        # CTC cần format: (T, N, vocab_size)
        logits = logits.transpose(0, 1)  # (T, N, vocab_size)
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Flatten targets (remove padding)
        targets_flat = []
        for i in range(targets.shape[0]):
            targets_flat.extend(targets[i, :target_lengths[i]].tolist())
        targets_flat = torch.tensor(targets_flat, dtype=torch.long, device=targets.device)
        
        # CTC loss
        loss = self.ctc_loss(log_probs, targets_flat, input_lengths, target_lengths)
        
        return loss
    
    def decode(self, logits, method='greedy', beam_width=5):
        """
        Decode logits thành sequences
        
        Args:
            logits: (N, T, vocab_size)
            method: 'greedy' hoặc 'beam_search'
            beam_width: beam width cho beam search
        
        Returns:
            decoded: list of sequences
        """
        # Convert to (T, N, vocab_size)
        logits = logits.transpose(0, 1)
        
        if method == 'greedy':
            return self.ctc_head.decode_greedy(logits)
        elif method == 'beam_search':
            return self.ctc_head.decode_beam_search(logits, beam_width)
        else:
            raise ValueError(f"Unknown decoding method: {method}")
