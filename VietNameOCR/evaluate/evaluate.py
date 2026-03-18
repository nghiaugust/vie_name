import sys
import os
import time
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# Thêm thư mục VietNameOCR vào sys.path để import predict
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

from predict import VietNameOCR

def levenshtein_distance(s1, s2):
    """Tính khoảng cách Levenshtein (CER/WER)"""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

def calculate_metrics(pred, label):
    """Tính 3 metrics: Exact Match, Word Acc (1-WER), Char Acc (1-CER)"""
    pred = pred.strip()
    label = label.strip()
    
    # 1. Exact Match
    exact_match = 1 if pred == label else 0
    
    # 2. Word-level Accuracy (1 - WER)
    pred_words = pred.split()
    label_words = label.split()
    
    if len(label_words) == 0:
        word_acc = 1.0 if len(pred_words) == 0 else 0.0
    else:
        word_distance = levenshtein_distance(pred_words, label_words)
        word_acc = max(0.0, 1.0 - (word_distance / len(label_words)))
    
    # 3. Character-level Accuracy (1 - CER)
    if len(label) == 0:
        char_acc = 1.0 if len(pred) == 0 else 0.0
    else:
        char_distance = levenshtein_distance(pred, label)
        char_acc = max(0.0, 1.0 - (char_distance / len(label)))
    
    return exact_match, word_acc, char_acc

def load_annotations(annotation_path):
    """Đọc file annotation.txt, danh sách 15 tên (dòng 1 -> dòng 15)"""
    labels = {}
    with open(annotation_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f, start=1):
            name = line.strip()
            if name:
                labels[idx] = name
    return labels

def get_image_samples(image_dir, labels):
    """Lấy danh sách các ảnh và nối nhãn dựa vào ID đuôi (VD: *_*_X.jpg -> dòng X)"""
    samples = []
    valid_extensions = {'.jpg', '.jpeg', '.png'}
    
    for entry in image_dir.iterdir():
        if entry.is_file() and entry.suffix.lower() in valid_extensions:
            # Bỏ file không đúng format
            # File có dạng XXX_Y.jpg, chia cắt bằng "_" và lấy số cuối
            parts = entry.stem.split('_')
            try:
                line_idx = int(parts[-1])
                # Lấy nhãn tương ứng ID
                if line_idx in labels:
                    samples.append((entry, labels[line_idx]))
                else:
                    print(f"⚠️  File {entry.name}: Không có dòng {line_idx} trong annotation.txt")
            except ValueError:
                print(f"⚠️  File {entry.name}: Không parse được số thứ tự dòng.")
                
    return samples

def evaluate_images(model, samples):
    total_samples = len(samples)
    if total_samples == 0:
        print("Không có ảnh để đánh giá!")
        return None
        
    exact_matches = 0
    total_word_acc = 0.0
    total_char_acc = 0.0
    errors = []
    
    start_time = time.time()
    
    for test_idx, (img_path, label) in enumerate(tqdm(samples, desc="🔥 Predicting", ncols=80), 1):
        try:
            pred = model.predict(str(img_path))
            exact, w_acc, c_acc = calculate_metrics(pred, label)
            
            exact_matches += exact
            total_word_acc += w_acc
            total_char_acc += c_acc
            
            if exact == 0:
                errors.append({
                    "image": img_path.name,
                    "label": label,
                    "pred": pred,
                    "word_acc": w_acc,
                    "char_acc": c_acc
                })
        except Exception as e:
            print(f"Lỗi khi dự đoán ảnh {img_path.name}: {e}")

    elapsed_time = time.time() - start_time
    avg_time_per_image = (elapsed_time / total_samples) * 1000

    exact_match_acc = (exact_matches / total_samples) * 100
    avg_word_acc = (total_word_acc / total_samples) * 100
    avg_char_acc = (total_char_acc / total_samples) * 100

    return {
        'total_samples': total_samples,
        'exact_matches': exact_matches,
        'exact_match_acc': exact_match_acc,
        'avg_word_acc': avg_word_acc,
        'avg_char_acc': avg_char_acc,
        'elapsed_time': elapsed_time,
        'avg_time_per_image': avg_time_per_image,
        'errors': errors
    }

def main():
    # 1. Đường dẫn thư mục & file cần thiết
    evaluate_dir = Path(__file__).parent
    image_dir = evaluate_dir / "image_name"
    annotation_file = image_dir / "annotation.txt"
    report_file = evaluate_dir / "ket_qua_danh_gia.txt"
    
    config_file = str(project_root / "config_mobilenet_svtr_ctc.yml")
    weights_file = str(project_root / "weights" / "mobilenet_svtr_ctc.pth")
    
    print("="*60)
    print("ĐÁNH GIÁ MÔ HÌNH VIE_NAME (MobileNet-SVTR-CTC)")
    print("="*60)
    
    # 2. Xử lý Ground Truth (Labels)
    if not annotation_file.exists():
        print(f"Không tìm thấy file annotation tại: {annotation_file}")
        sys.exit(1)
        
    labels_dict = load_annotations(annotation_file)
    print(f"✓ Đã load {len(labels_dict)} tên từ annotation.txt.")
    
    # Lắp nhãn vào danh sách tập ảnh
    samples = get_image_samples(image_dir, labels_dict)
    print(f"✓ Tìm thấy {len(samples)} ảnh hợp lệ cần đánh giá.")
    
    if not samples:
        print("Vòng duyệt dataset trống. Dừng đánh giá.")
        sys.exit(0)
        
    # 3. Nạp model thông qua class từ predict.py
    if not os.path.exists(config_file) or not os.path.exists(weights_file):
        print(f"Lỗi: Không tìm thấy configs/weights ở đường dẫn gốc.")
        sys.exit(1)
        
    print("\n⏳ Đang load mô hình từ predict.py ...")
    ocr_model = VietNameOCR(config_path=config_file, weights_path=weights_file)
    print("✓ Load xong mô hình!\n")

    # 4. Bắt đầu đánh giá
    results = evaluate_images(ocr_model, samples)
    if not results:
        return

    # 5. In kết quả console & Xuất ra Report 
    print(f"\n📊 KẾT QUẢ:")
    print(f"  ✓ Exact Match Accuracy:     {results['exact_match_acc']:.2f}% ({results['exact_matches']}/{results['total_samples']})")
    print(f"  ✓ Word-level Accuracy:      {results['avg_word_acc']:.2f}%")
    print(f"  ✓ Character-level Accuracy: {results['avg_char_acc']:.2f}%")
    print(f"  ✓ Thời gian chạy:           {results['elapsed_time']:.2f}s")
    print(f"  ✓ Tốc độ TB / Ảnh:          {results['avg_time_per_image']:.2f}ms")
    
    print(f"\n💾 Ghi file báo cáo chi tiết vào {report_file.name}")
    with open(report_file, 'w', encoding='utf-8') as f:

        f.write(f"Bộ test: Thư mục {image_dir.name}\n")
        f.write(f"Tổng số mẫu: {results['total_samples']}\n\n")
        
        f.write("KẾT QUẢ HIỆU NĂNG:\n")
        f.write(f" - Exact Match:    {results['exact_match_acc']:.2f}% ({results['exact_matches']}/{results['total_samples']})\n")
        f.write(f" - Word Acc:       {results['avg_word_acc']:.2f}%\n")
        f.write(f" - Char Acc:       {results['avg_char_acc']:.2f}%\n")
        f.write(f" - Time / Image:   {results['avg_time_per_image']:.2f} ms\n\n")
        
        if results['errors']:
            f.write("DANH SÁCH CÁC MẪU DỰ ĐOÁN SAI:\n")
            f.write("-" * 60 + "\n")
            # Sort lỗi theo tên để dễ tra cứu
            results['errors'].sort(key=lambda x: x['image'])
            for i, err in enumerate(results['errors'], 1):
                f.write(f"{i}. {err['image']}\n")
                f.write(f"   - Label:      {err['label']}\n")
                f.write(f"   - Prediction: {err['pred']}\n")
                f.write(f"   - Word Acc:   {err['word_acc']*100:.1f}%, Char Acc: {err['char_acc']*100:.1f}%\n")
                f.write("\n")

if __name__ == "__main__":
    main()
