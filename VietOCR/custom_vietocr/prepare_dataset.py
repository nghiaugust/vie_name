"""
Script chuẩn bị dataset cho training VietOCR
Chia dataset thành train/validation (90/10)
"""

import os
import random
from pathlib import Path

def split_dataset(
    input_annotation_file,
    output_train_file,
    output_val_file,
    train_ratio=0.9,
    seed=42
):
    """
    Chia dataset thành train và validation
    
    Args:
        input_annotation_file: File annotation gốc
        output_train_file: File annotation cho training
        output_val_file: File annotation cho validation
        train_ratio: Tỷ lệ dữ liệu training (0.9 = 90%)
        seed: Random seed để reproducible
    """
    print(f"Đang đọc file annotation: {input_annotation_file}")
    
    # Đọc toàn bộ annotations
    with open(input_annotation_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"Tổng số mẫu: {len(lines)}")
    
    # Sửa đường dẫn ảnh (strip ./dataset/ hoặc dataset/)
    fixed_lines = []
    for line in lines:
        parts = line.strip().split('\t')
        if len(parts) >= 2:
            img_path = parts[0]
            # Loại bỏ ./dataset/ hoặc dataset/ ở đầu đường dẫn
            img_path = img_path.replace('./dataset/', '').replace('dataset/', '')
            fixed_line = img_path + '\t' + '\t'.join(parts[1:]) + '\n'
            fixed_lines.append(fixed_line)
        else:
            fixed_lines.append(line)
    
    lines = fixed_lines
    print(f"✓ Đã sửa đường dẫn ảnh (loại bỏ ./dataset/ prefix)")
    
    # Shuffle với seed cố định
    random.seed(seed)
    random.shuffle(lines)
    
    # Tính số lượng train/val
    num_train = int(len(lines) * train_ratio)
    num_val = len(lines) - num_train
    
    train_lines = lines[:num_train]
    val_lines = lines[num_train:]
    
    print(f"Số mẫu training: {num_train}")
    print(f"Số mẫu validation: {num_val}")
    
    # Tạo thư mục output nếu chưa có
    os.makedirs(os.path.dirname(output_train_file), exist_ok=True)
    os.makedirs(os.path.dirname(output_val_file), exist_ok=True)
    
    # Ghi file train
    with open(output_train_file, 'w', encoding='utf-8') as f:
        f.writelines(train_lines)
    print(f"✓ Đã tạo file training: {output_train_file}")
    
    # Ghi file validation
    with open(output_val_file, 'w', encoding='utf-8') as f:
        f.writelines(val_lines)
    print(f"✓ Đã tạo file validation: {output_val_file}")


def verify_dataset(annotation_file, data_root, max_check=100):
    """
    Kiểm tra xem các file ảnh trong annotation có tồn tại không
    
    Args:
        annotation_file: File annotation cần kiểm tra
        data_root: Thư mục gốc chứa ảnh
        max_check: Số lượng file tối đa cần kiểm tra
    """
    print(f"\nKiểm tra dataset: {annotation_file}")
    
    with open(annotation_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    missing_files = []
    checked = 0
    
    for i, line in enumerate(lines):
        if checked >= max_check and max_check > 0:
            break
            
        parts = line.strip().split('\t')
        if len(parts) < 2:
            print(f"Cảnh báo: Dòng {i+1} không đúng format: {line.strip()}")
            continue
        
        img_path = parts[0]
        # Xử lý đường dẫn tương đối
        if img_path.startswith('./'):
            img_path = img_path[2:]
        
        full_path = os.path.join(data_root, img_path)
        
        if not os.path.exists(full_path):
            missing_files.append(full_path)
        
        checked += 1
    
    if missing_files:
        print(f"⚠ Cảnh báo: Tìm thấy {len(missing_files)} file không tồn tại (trong {checked} file đã kiểm tra)")
        print("Ví dụ:")
        for f in missing_files[:5]:
            print(f"  - {f}")
    else:
        print(f"✓ Tất cả {checked} file đã kiểm tra đều tồn tại")


def analyze_dataset(annotation_file):
    """
    Phân tích dataset: độ dài text, phân bố ký tự, v.v.
    
    Args:
        annotation_file: File annotation cần phân tích
    """
    print(f"\nPhân tích dataset: {annotation_file}")
    
    with open(annotation_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    text_lengths = []
    all_chars = set()
    
    for line in lines:
        parts = line.strip().split('\t')
        if len(parts) >= 2:
            text = parts[1]
            text_lengths.append(len(text))
            all_chars.update(text)
    
    if text_lengths:
        print(f"Số mẫu: {len(text_lengths)}")
        print(f"Độ dài text:")
        print(f"  - Min: {min(text_lengths)} ký tự")
        print(f"  - Max: {max(text_lengths)} ký tự")
        print(f"  - Trung bình: {sum(text_lengths)/len(text_lengths):.1f} ký tự")
        print(f"Tổng số ký tự unique: {len(all_chars)}")
        
        # Hiển thị một số ký tự
        sample_chars = sorted(list(all_chars))[:50]
        print(f"Ví dụ các ký tự: {''.join(sample_chars)}")


if __name__ == "__main__":
    # Cấu hình đường dẫn
    BASE_DIR = Path(__file__).parent.parent  # Lên thư mục cha (VietOCR)
    DATA_ROOT = BASE_DIR / "dataset"
    
    INPUT_ANNOTATION = DATA_ROOT / "label.txt"
    OUTPUT_TRAIN = DATA_ROOT / "train_annotation.txt"
    OUTPUT_VAL = DATA_ROOT / "val_annotation.txt"
    
    print("=" * 60)
    print("CHUẨN BỊ DATASET CHO TRAINING VIETOCR")
    print("=" * 60)
    
    # Bước 1: Phân tích dataset gốc
    if os.path.exists(INPUT_ANNOTATION):
        analyze_dataset(INPUT_ANNOTATION)
    else:
        print(f"❌ Không tìm thấy file: {INPUT_ANNOTATION}")
        exit(1)
    
    # Bước 2: Chia dataset
    split_dataset(
        input_annotation_file=str(INPUT_ANNOTATION),
        output_train_file=str(OUTPUT_TRAIN),
        output_val_file=str(OUTPUT_VAL),
        train_ratio=0.9,
        seed=42
    )
    
    # Bước 3: Kiểm tra dataset
    print("\nKiểm tra file training:")
    verify_dataset(str(OUTPUT_TRAIN), str(DATA_ROOT), max_check=100)
    
    print("\nKiểm tra file validation:")
    verify_dataset(str(OUTPUT_VAL), str(DATA_ROOT), max_check=100)
    
    print("\n" + "=" * 60)
    print("✓ HOÀN TẤT CHUẨN BỊ DATASET")
    print("=" * 60)
    print(f"\nFile training: {OUTPUT_TRAIN}")
    print(f"File validation: {OUTPUT_VAL}")
    print("\nBạn có thể bắt đầu training ngay!")
