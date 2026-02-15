# Training VietOCR với Dataset Tên Tiếng Việt

## 📋 Tổng Quan

Dự án này training lại mô hình VietOCR để nhận dạng tên tiếng Việt từ ảnh.

**Dataset:**
- 145,962 ảnh tên người Việt Nam
- Chiều cao ảnh: 32px
- Format: `đường_dẫn_ảnh\ttên_tiếng_việt`

**Model:**
- Architecture: VGG19-BN + Transformer
- Pretrained: Có (từ VietOCR)
- Training time: ~2-4 giờ (tùy GPU)

## 🚀 Quick Start

**⚠️ LƯU Ý:** Trước tiên, cd vào thư mục `custom_vietocr`:
```bash
cd custom_vietocr
```

### Bước 1: Chuẩn bị Dataset

```bash
python prepare_dataset.py
```

Kết quả:
- `../dataset/train_annotation.txt`: 131,366 mẫu (90%)
- `../dataset/val_annotation.txt`: 14,596 mẫu (10%)

### Bước 2: Training

```bash
# Train với pretrained weights (khuyến nghị)
python train_vietnamese_names.py

# Hoặc train từ đầu
python train_vietnamese_names.py --from-scratch

# Resume từ checkpoint
python train_vietnamese_names.py --checkpoint ../checkpoint/vietnamese_names_checkpoint.pth
```

### Bước 3: Test Model

```bash
# Test 1 ảnh
python test_model.py --image ../dataset/images/1.jpg

# Test nhiều ảnh
python test_model.py --folder ../dataset/images --limit 10

# Test accuracy trên validation set
python test_model.py --test --limit 100
```

## 📁 Cấu Trúc Files

```
VietOCR/
├── custom_vietocr/                 # ← Thư mục làm việc (cd vào đây)
│   ├── prepare_dataset.py          # Script chia dataset train/val
│   ├── train_vietnamese_names.py   # Script training
│   ├── test_model.py               # Script test/predict
│   ├── visualize_results.py        # Script visualize kết quả
│   ├── config_vietnamese_names.yml # File config training
│   └── README_TRAINING.md          # File này
│
├── dataset/
│   ├── images/                     # 145,962 ảnh
│   ├── label.txt                   # Annotation gốc
│   ├── train_annotation.txt        # Training data (auto-generated)
│   └── val_annotation.txt          # Validation data (auto-generated)
│
├── weights/
│   └── vietnamese_names_best.pth   # Model tốt nhất (sau training)
│
├── checkpoint/
│   └── vietnamese_names_checkpoint.pth  # Checkpoint để resume
│
├── logs/
│   └── vietnamese_names_train.log  # Training logs
│
└── visualization/
    └── *.jpg                        # Kết quả visualize (nếu có)
```

## ⚙️ Cấu Hình Training

File: `config_vietnamese_names.yml`

### Điều chỉnh theo GPU

**GPU nhỏ (< 6GB):**
```yaml
trainer:
  batch_size: 16  # Giảm từ 32
```

**Không có GPU:**
```yaml
device: cpu
trainer:
  batch_size: 8
```

### Điều chỉnh số lượng training

```yaml
trainer:
  iters: 50000    # Mặc định ~12 epochs
  # iters: 100000 # ~24 epochs (chính xác hơn)
```

### Thay đổi kiến trúc

**Seq2Seq (nhanh hơn 7x):**
```yaml
seq_modeling: seq2seq
```

**Model nhỏ hơn:**
```yaml
transformer:
  d_model: 128
  num_encoder_layers: 3
  num_decoder_layers: 3
```

## 📊 Kết Quả Mong Đợi

**Chỉ số tốt cho dataset tên tiếng Việt:**
- Accuracy full sequence: > 85% (tốt), > 90% (rất tốt), > 95% (xuất sắc)
- Accuracy per character: > 95% (tốt), > 98% (rất tốt)

**Thời gian training:**
- GPU (RTX 3060): ~2-3 giờ (50k iterations)
- GPU (GTX 1060): ~4-6 giờ (50k iterations)
- CPU: ~2-3 ngày (KHÔNG khuyến nghị)

## 🔧 Sử Dụng Model Sau Khi Train

```python
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from PIL import Image

# Load config
config = Cfg.load_config_from_file('config_vietnamese_names.yml')
config['weights'] = '../weights/vietnamese_names_best.pth'
config['device'] = 'cuda:0'  # hoặc 'cpu'

# Tạo predictor
predictor = Predictor(config)

# Predict
img = Image.open('../dataset/images/test_image.jpg')
text = predictor.predict(img)
print(text)  # "Nguyễn Văn A"
```

## 🐛 Troubleshooting

### CUDA out of memory
```yaml
trainer:
  batch_size: 16  # hoặc 8
```

### Training quá chậm
1. Kiểm tra đang dùng GPU: `nvidia-smi`
2. Tăng num_workers:
```yaml
dataloader:
  num_workers: 4
```
3. Đổi sang seq2seq

### Accuracy không tăng
1. Tăng số iterations
2. Thử learning rate khác
3. Kiểm tra data có đúng không

## 📚 Tài Liệu Tham Khảo

- **TRAINING_GUIDE.txt**: Hướng dẫn chi tiết tất cả tùy chọn
- **VietOCR GitHub**: https://github.com/pbcquoc/vietocr
- **Documentation**: https://pbcquoc.github.io/vietocr

## 📝 Notes

- **Luôn cd vào `custom_vietocr` trước khi chạy**: `cd custom_vietocr`
- File annotation format: `đường_dẫn_ảnh\tnhãn` (tab-separated)
- Checkpoint tự động lưu mỗi lần validate
- Model tốt nhất được lưu khi accuracy cao hơn
- Nhấn Ctrl+C để dừng training an toàn

## 🎯 Workflow Summary

```
cd custom_vietocr → prepare_dataset.py → train_vietnamese_names.py → test_model.py
       ↓                   ↓                        ↓                         ↓
  Vào thư mục         Chia train/val          Training model         Test accuracy
```

Chúc bạn training thành công! 🚀
