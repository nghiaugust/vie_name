"""
CTC-compatible Vocab class
Khác VietOCR Vocab: blank token ở index 0, không có sos/eos
"""


class VocabCTC:
    """
    Vocab cho CTC models
    Index 0: blank token (dành riêng cho CTC)
    Index 1+: actual characters
    """
    def __init__(self, chars):
        self.blank = 0  # CTC blank token ở index 0
        self.chars = chars
        
        # Character to index: start from 1 (0 is blank)
        self.c2i = {c: i + 1 for i, c in enumerate(chars)}
        
        # Index to character
        self.i2c = {i + 1: c for i, c in enumerate(chars)}
        self.i2c[0] = "<blank>"
        
    def encode(self, chars):
        """
        Encode string to indices
        CTC không cần sos/eos tokens
        
        Args:
            chars: string
        Returns:
            list of indices (không bao gồm blank)
        """
        return [self.c2i[c] for c in chars]
    
    def decode(self, ids):
        """
        Decode indices to string
        Tự động bỏ qua blank tokens (index 0)
        
        Args:
            ids: list of indices
        Returns:
            decoded string
        """
        chars = []
        for i in ids:
            if i == 0:  # Skip blank
                continue
            if i in self.i2c:
                chars.append(self.i2c[i])
        return "".join(chars)
    
    def __len__(self):
        """Vocab size bao gồm blank token"""
        return len(self.chars) + 1  # +1 for blank
    
    def batch_decode(self, arr):
        """Decode batch of sequences"""
        return [self.decode(ids) for ids in arr]
    
    def __str__(self):
        return f"<blank>{self.chars}"
