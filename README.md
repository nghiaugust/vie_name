# Vietnamese Name Recognition using OCR Models

## Introduction

This project builds and evaluates OCR models for recognizing Vietnamese names, specifically for application in Vietnamese ballot name recognition. The research compares 3 different model architectures in terms of accuracy and processing speed.

## Models

### 1. VGG Transformer (VietOCR)
- Architecture: VGG backbone + Transformer decoder
- Source: VietOCR library
- Fine-tuned on custom dataset

### 2. VGG Seq2Seq (VietOCR)  
- Architecture: VGG backbone + ConvSeq2Seq decoder
- Source: VietOCR library
- Fine-tuned on custom dataset

### 3. VietNameOCR (Proposed)
- Architecture: MobileNetV3 backbone + SVTR + CTC decoder
- Proposed model optimized for speed
- Lightweight design for practical deployment

## Dataset

**Total samples:** 145,961 Vietnamese name images
- Training set: 116,768 images (80%)
- Validation set: 14,596 images (10%)
- Test set: 14,597 images (10%)

**Characteristics:**
- Name images constructed from combinations of common Vietnamese surnames, middle names, and given names
- Diverse image quality
- Includes Vietnamese characters

## Evaluation Results

Evaluation on test set (14,597 samples):

| Model | Exact Match | Word Acc | Char Acc | Speed (ms/img) |
|-------|-------------|----------|----------|----------------|
| **VGG Seq2Seq** | **99.52%** | **99.82%** | **99.94%** | 16.41 |
| **VGG Transformer** | **99.21%** | **99.73%** | **99.91%** | 116.36 |
| **VietNameOCR** | **98.65%** | **99.51%** | **99.83%** | **5.42** |
| VGG Transformer (Pretrained) | 64.43% | 83.00% | 93.43% | 116.41 |
| VGG Seq2Seq (Pretrained) | 54.09% | 76.19% | 88.40% | 15.87 |

### Analysis

**Accuracy:**
- **VGG Seq2Seq** achieves the highest accuracy with 99.52% exact match
- All 3 fine-tuned models achieve >98.6% exact match accuracy
- Fine-tuning significantly improves performance compared to pretrained models (~45% exact match improvement)

**Speed:**
- **VietNameOCR** is the fastest: **5.42ms/image** (3x faster than Seq2Seq, 21x faster than Transformer)
- Suitable for real-time deployment with 98.65% accuracy

## Directory Structure

```
VietOCR/
├── custom_vietocr/           # Training scripts and configs
├── dataset/                  # Dataset and annotations
├── weights/                  # Model checkpoints
│   └── best/                # Best models
├── evaluate/                     # Evaluation scripts
│   └── evaluate_test/            # Test results
└── vietocr/                  # VietOCR library
```

## Training

```bash
# VGG Transformer
python VietOCR/custom_vietocr/train_vietnamese_names.py --config config_vgg_transformer.yml

# VGG Seq2Seq  
python VietOCR/custom_vietocr/train_vietnamese_names.py --config config_vgg_seq2seq.yml

# VietNameOCR
python VietOCR/custom_vietocr/train_mobilenet_svtr_ctc.py
```

## Requirements

- Python 3.8+
- PyTorch 1.9+
- CUDA 11.0+ (for GPU)
- VietOCR library
- See `requirements.txt` for full dependencies

## References

- VietOCR: [https://github.com/pbcquoc/vietocr](https://github.com/pbcquoc/vietocr)
- SVTR: Scene Text Recognition with a Single Visual Model

## Contact

For more information, please contact: **domanhnghiaforwork@gmail.com**
