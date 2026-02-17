"""
CTC-specific Collator
Đơn giản hóa cho CTC: không cần tgt_input, masked_language_model
"""

import numpy as np
import torch


class CollatorCTC:
    """
    Collator đơn giản cho CTC models
    
    Khác VietOCR Collator:
    - Không tạo tgt_input (CTC không có decoder/teacher forcing)
    - Không masked language model
    - Chỉ padding targets về cùng length, không roll
    """
    
    def __init__(self):
        pass
    
    def __call__(self, batch):
        """
        Args:
            batch: list of samples from OCRDataset
                   [{img, word, img_path}, ...]
        
        Returns:
            dict: {
                'img': (N, C, H, W),
                'tgt_output': (N, max_len) - padded with 0,
                'tgt_lengths': (N,) - actual lengths,
                'filenames': list of paths
            }
        """
        filenames = []
        img = []
        tgt_output = []
        tgt_lengths = []
        
        # Find max label length
        max_label_len = max(len(sample["word"]) for sample in batch)
        
        # Find max image width in batch
        max_width = max(sample["img"].shape[2] for sample in batch)
        
        for sample in batch:
            # Get image (C, H, W)
            image = sample["img"]
            filenames.append(sample["img_path"])
            
            # Pad image to max_width with 0 (black padding)
            c, h, w = image.shape
            if w < max_width:
                # Pad on the right side
                pad_width = max_width - w
                padded_img = np.zeros((c, h, max_width), dtype=np.float32)
                padded_img[:, :, :w] = image
                img.append(padded_img)
            else:
                img.append(image)
            
            # Pad label với 0 (blank token trong CTC vocab)
            label = sample["word"]
            label_len = len(label)
            tgt = np.concatenate(
                (label, np.zeros(max_label_len - label_len, dtype=np.int32))
            )
            tgt_output.append(tgt)
            tgt_lengths.append(label_len)
        
        # Now all images have same shape, safe to stack
        img = np.array(img, dtype=np.float32)
        tgt_output = np.array(tgt_output, dtype=np.int64)
        tgt_lengths = np.array(tgt_lengths, dtype=np.int32)
        
        rs = {
            "img": torch.FloatTensor(img),
            "tgt_output": torch.LongTensor(tgt_output),
            "tgt_lengths": torch.LongTensor(tgt_lengths),
            "filenames": filenames,
        }
        
        return rs
