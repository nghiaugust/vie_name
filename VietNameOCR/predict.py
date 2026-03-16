import yaml
import torch
import numpy as np
from PIL import Image

# Thay đổi import phù hợp với cấu trúc standalone mới
from model.mobilenet_svtr_ctc import mobilenet_svtr_ctc
from model.vocab_ctc import VocabCTC

class VietNameOCR:
    def __init__(self, config_path, weights_path, device=None):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        if device is None:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        print(f"Loading VietNameOCR on {self.device}...")
        
        # 1. Khởi tạo vocab
        self.vocab = VocabCTC(self.config['vocab'])
        self.blank_idx = 0
        
        # 2. Khởi tạo model
        model_config = self.config['model']
        self.model = mobilenet_svtr_ctc(
            vocab_size=len(self.vocab),
            hidden=model_config.get('backbone_hidden', 256),
            svtr_depth=model_config.get('svtr_depth', 2),
            svtr_heads=model_config.get('svtr_heads', 8),
            dropout=model_config.get('dropout', 0.1),
            pretrained=False
        )
        
        # 3. Load trọng số
        checkpoint = torch.load(weights_path, map_location=self.device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
            
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Cấu hình tiền xử lý ảnh
        self.img_height = self.config['train']['image_height']
        self.img_width = self.config['train']['image_width']

    def preprocess_image(self, img):
        """Tiền xử lý ảnh giống như lúc train"""
        w, h = img.size
        # Resize giữ đúng tỉ lệ
        new_w = int(self.img_height * w / h)
        new_w = min(new_w, self.img_width)
        img = img.resize((new_w, self.img_height), Image.Resampling.LANCZOS)
        
        img = np.array(img)
        # Nếu ảnh xám (2D) thì convert sang 3D
        if len(img.shape) == 2:
            img = np.stack([img, img, img], axis=2)
            
        # Pytorch cần format (C, H, W) thay vì (H, W, C)
        img = img.transpose(2, 0, 1)
        img = img.astype(np.float32) / 255.0
        
        # Pad pixel 0 ở phần bên phải ảnh nếu chiều rộng chưa đủ
        c, h, w = img.shape
        if w < self.img_width:
            pad_width = self.img_width - w
            pad = np.zeros((c, h, pad_width), dtype=np.float32)
            img = np.concatenate([img, pad], axis=2)
            
        return img
    
    def decode_greedy(self, logits):
        """Giải mã dự đoán của mô hình CTC"""
        pred = torch.argmax(logits, dim=-1)
        decoded_seq = []
        prev = None
        for token in pred.tolist():
            if token != self.blank_idx and token != prev:
                decoded_seq.append(token)
            prev = token
        return self.vocab.decode(decoded_seq)
    
    def predict(self, image_path_or_pil):
        """Hàm API gọi từ bên ngoài để dự đoán text"""
        if isinstance(image_path_or_pil, str):
            img = Image.open(image_path_or_pil).convert('RGB')
        else:
            img = image_path_or_pil.convert('RGB')
            
        with torch.no_grad():
            img_tensor = self.preprocess_image(img)
            img_tensor = torch.FloatTensor(img_tensor).unsqueeze(0)
            img_tensor = img_tensor.to(self.device)
            
            logits = self.model(img_tensor)
            text = self.decode_greedy(logits[0])
            
        return text

if __name__ == "__main__":
    # Cấu hình file path
    config_file = 'config_mobilenet_svtr_ctc.yml'
    weights_file = 'weights/mobilenet_svtr_ctc.pth'
    
    import sys
    import os
    if not os.path.exists(config_file) or not os.path.exists(weights_file):
        print(f"Không tìm thấy file config hoặc trọng số! Hãy kiểm tra lại thư mục.")
        sys.exit(1)
        
    print("Khởi tạo model OCR...")
    ocr_engine = VietNameOCR(config_file, weights_file)
    print("Mô hình đã sẵn sàng để kiểm tra!")