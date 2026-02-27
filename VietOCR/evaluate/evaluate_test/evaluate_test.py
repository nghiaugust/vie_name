"""
Đánh giá 5 mô hình OCR trên dataset 45 thư mục
- 3 loại metrics: Exact Match Accuracy, Word-level Accuracy, Character-level Accuracy
- 5 models: 
  1. VGG Transformer (pretrained VietOCR)
  2. VGG Seq2Seq (pretrained VietOCR)
  3. VGG Transformer (fine-tuned)
  4. VGG Seq2Seq (fine-tuned)
  5. MobileNet-SVTR-CTC (fine-tuned)
- Batch size = 1 (sequential processing)
"""
import sys
import time
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
import yaml

# Add vietocr to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'vietocr'))

from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from vietocr.model.mobilenet_svtr_ctc import mobilenet_svtr_ctc
from vietocr.model.vocab_ctc import VocabCTC


class MobileNetCTCPredictor:
    """Predictor cho MobileNet-SVTR-CTC Model"""
    
    def __init__(self, config_path, weights_path, device='cuda:0'):
        # Load config
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Build vocab
        self.vocab = VocabCTC(self.config['vocab'])
        self.blank_idx = 0
        
        # Build model
        model_config = self.config['model']
        self.model = mobilenet_svtr_ctc(
            vocab_size=len(self.vocab),
            hidden=model_config.get('backbone_hidden', 256),
            svtr_depth=model_config.get('svtr_depth', 2),
            svtr_heads=model_config.get('svtr_heads', 8),
            dropout=model_config.get('dropout', 0.1),
            pretrained=False
        )
        
        # Load weights
        checkpoint = torch.load(weights_path, map_location=self.device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Image transform
        self.img_height = self.config['train']['image_height']
        self.img_width = self.config['train']['image_width']
    
    def preprocess_image(self, img):
        """Tiền xử lý ảnh"""
        w, h = img.size
        new_w = int(self.img_height * w / h)
        new_w = min(new_w, self.img_width)
        img = img.resize((new_w, self.img_height), Image.LANCZOS)
        
        img = np.array(img)
        if len(img.shape) == 2:
            img = np.stack([img, img, img], axis=2)
        img = img.transpose(2, 0, 1)
        img = img.astype(np.float32) / 255.0
        
        # Pad to full width
        c, h, w = img.shape
        if w < self.img_width:
            pad_width = self.img_width - w
            pad = np.zeros((c, h, pad_width), dtype=np.float32)
            img = np.concatenate([img, pad], axis=2)
        
        return img
    
    def decode_greedy(self, logits):
        """CTC Greedy Decoding"""
        pred = torch.argmax(logits, dim=-1)
        decoded_seq = []
        prev = None
        for token in pred.tolist():
            if token != self.blank_idx and token != prev:
                decoded_seq.append(token)
            prev = token
        return self.vocab.decode(decoded_seq)
    
    def predict(self, img):
        """Predict single image"""
        with torch.no_grad():
            img_tensor = self.preprocess_image(img)
            img_tensor = torch.FloatTensor(img_tensor).unsqueeze(0)
            img_tensor = img_tensor.to(self.device)
            
            logits = self.model(img_tensor)
            text = self.decode_greedy(logits[0])
        
        return text


def levenshtein_distance(s1, s2):
    """
    Tính số phép biến đổi tối thiểu (Thêm, Xóa, Sửa) để biến s1 thành s2
    Đây là tiêu chuẩn để tính CER (Character Error Rate) và WER (Word Error Rate) trong OCR
    """
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
    """
    Tính toán 3 metrics chuẩn mực cho OCR bằng Levenshtein Distance:
    1. Exact Match: So sánh toàn bộ chuỗi
    2. Word-level Accuracy: 1 - WER (Word Error Rate)
    3. Character-level Accuracy: 1 - CER (Character Error Rate)
    
    Returns:
        exact_match (int): 1 nếu pred == label, 0 nếu không
        word_acc (float): % từ đúng (dựa trên edit distance)
        char_acc (float): % ký tự đúng (dựa trên edit distance)
    """
    # 1. Exact Match
    exact_match = 1 if pred == label else 0
    
    # 2. Word-level Accuracy (1 - Word Error Rate)
    pred_words = pred.split()
    label_words = label.split()
    
    if len(label_words) == 0:
        word_acc = 1.0 if len(pred_words) == 0 else 0.0
    else:
        word_distance = levenshtein_distance(pred_words, label_words)
        # WER = edit_distance / len(reference)
        # Accuracy = 1 - WER (giới hạn min là 0)
        word_acc = max(0.0, 1.0 - (word_distance / len(label_words)))
    
    # 3. Character-level Accuracy (1 - Character Error Rate)
    if len(label) == 0:
        char_acc = 1.0 if len(pred) == 0 else 0.0
    else:
        char_distance = levenshtein_distance(pred, label)
        # CER = edit_distance / len(reference)
        # Accuracy = 1 - CER (giới hạn min là 0)
        char_acc = max(0.0, 1.0 - (char_distance / len(label)))
    
    return exact_match, word_acc, char_acc


def load_dataset(dataset_dir, annotation_file):
    """
    Đọc dataset từ file test_annotation.txt với format: images/filename.jpg\tlabel
    Returns:
        samples: list of (img_path, label)
    """
    samples = []
    
    with open(annotation_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split('\t')
            if len(parts) >= 2:
                # Path trong file: images/117535.jpg
                relative_path = parts[0]
                label = parts[1]
                
                # Tạo đường dẫn đầy đủ: dataset_dir/images/filename.jpg
                img_path = dataset_dir / relative_path
                
                if img_path.exists():
                    samples.append((img_path, label))
                else:
                    print(f"⚠️  Không tìm thấy: {img_path}")
    
    return samples


def evaluate_model(model_name, predictor, samples):
    """
    Đánh giá model trên dataset
    Returns:
        results: dict chứa các metrics và thời gian
    """
    print(f"\n{'='*70}")
    print(f"🔄 ĐÁNH GIÁ: {model_name}")
    print(f"{'='*70}")
    
    total_samples = len(samples)
    exact_matches = 0
    total_word_acc = 0.0
    total_char_acc = 0.0
    
    errors = []  # Lưu các sample bị lỗi để phân tích
    
    # Đo thời gian
    start_time = time.time()
    
    for img_path, label in tqdm(samples, desc=f"  {model_name}", ncols=80):
        try:
            img = Image.open(img_path).convert('RGB')
            pred = predictor.predict(img)
            
            # Tính metrics
            exact, word_acc, char_acc = calculate_metrics(pred, label)
            
            exact_matches += exact
            total_word_acc += word_acc
            total_char_acc += char_acc
            
            # Lưu lỗi nếu không exact match
            if exact == 0:
                errors.append({
                    'image': img_path.name,
                    'label': label,
                    'pred': pred,
                    'word_acc': word_acc,
                    'char_acc': char_acc
                })
        
        except Exception as e:
            print(f"\n⚠️  Lỗi predict {img_path}: {e}")
            continue
    
    # Tính thời gian
    elapsed_time = time.time() - start_time
    avg_time_per_image = (elapsed_time / total_samples) * 1000  # ms/ảnh
    
    # Tính trung bình
    exact_match_acc = (exact_matches / total_samples) * 100
    avg_word_acc = (total_word_acc / total_samples) * 100
    avg_char_acc = (total_char_acc / total_samples) * 100
    
    print(f"\n📊 KẾT QUẢ:")
    print(f"  ✓ Exact Match Accuracy:    {exact_match_acc:.2f}% ({exact_matches}/{total_samples})")
    print(f"  ✓ Word-level Accuracy:     {avg_word_acc:.2f}%")
    print(f"  ✓ Character-level Accuracy: {avg_char_acc:.2f}%")
    print(f"  ✓ Thời gian xử lý:          {elapsed_time:.2f}s")
    print(f"  ✓ Tốc độ trung bình:        {avg_time_per_image:.2f}ms/ảnh")
    
    return {
        'model': model_name,
        'total_samples': total_samples,
        'exact_match': exact_matches,
        'exact_match_acc': exact_match_acc,
        'word_acc': avg_word_acc,
        'char_acc': avg_char_acc,
        'elapsed_time': elapsed_time,
        'avg_time_per_image': avg_time_per_image,
        'errors': errors
    }


def main():
    # Cấu hình dataset
    dataset_dir = Path(__file__).parent.parent.parent / "dataset"
    annotation_file = dataset_dir / "test_annotation.txt"
    output_file = Path(__file__).parent / "ket_qua_danh_gia_test.txt"
    
    print("=" * 70)
    print("ĐÁNH GIÁ 5 MÔ HÌNH OCR")
    print("=" * 70)
    
    # Đọc dataset
    print("\n📖 Đang đọc bộ test từ test_annotation.txt...")
    samples = load_dataset(dataset_dir, annotation_file)
    print(f"✓ Sử dụng toàn bộ: {len(samples)} mẫu từ test_annotation.txt")
    
    # Cấu hình 5 models
    models_config = [
        {
            'name': 'VGG Transformer (Pretrained VietOCR)',
            'type': 'vietocr',
            'config': '../../custom_vietocr/config_base_vgg_transformer.yml',
            'weights': 'https://vocr.vn/data/vietocr/vgg_transformer.pth'
        },
        {
            'name': 'VGG Seq2Seq (Pretrained VietOCR)',
            'type': 'vietocr',
            'config': '../../custom_vietocr/config_vgg_seq2seq.yml',
            'weights': 'https://vocr.vn/data/vietocr/vgg_seq2seq.pth'
        },
        {
            'name': 'VGG Transformer (Fine-tuned)',
            'type': 'vietocr',
            'config': '../../custom_vietocr/config_base_vgg_transformer.yml',
            'weights': '../../weights/best/vgg_transformer.pth'
        },
        {
            'name': 'VGG Seq2Seq (Fine-tuned)',
            'type': 'vietocr',
            'config': '../../custom_vietocr/config_vgg_seq2seq.yml',
            'weights': '../../weights/best/seq2seq.pth'
        },
        {
            'name': 'MobileNet-SVTR-CTC (Fine-tuned)',
            'type': 'mobilenet_ctc',
            'config': '../../custom_vietocr/config_mobilenet_svtr_ctc.yml',
            'weights': '../../weights/best/mobilenet_svtr_ctc.pth'
        }
    ]
    
    all_results = []
    
    # Đánh giá từng model
    for model_cfg in models_config:
        model_name = model_cfg['name']
        
        try:
            # Load model
            print(f"\n⏳ Đang load {model_name}...")
            if model_cfg['type'] == 'vietocr':
                config = Cfg.load_config_from_file(model_cfg['config'])
                config['weights'] = model_cfg['weights']
                config['device'] = 'cuda:0'
                if 'predictor' not in config:
                    config['predictor'] = {}
                config['predictor']['beamsearch'] = False
                predictor = Predictor(config)
            else:  # mobilenet_ctc
                predictor = MobileNetCTCPredictor(
                    model_cfg['config'],
                    model_cfg['weights'],
                    device='cuda:0'
                )
            
            print(f"✓ Đã load {model_name}")
            
            # Evaluate
            result = evaluate_model(model_name, predictor, samples)
            all_results.append(result)
            
        except Exception as e:
            print(f"❌ Lỗi khi load/đánh giá {model_name}: {e}")
            continue
    
    # Ghi kết quả ra file
    print(f"\n💾 Ghi kết quả vào {output_file.name}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("KẾT QUẢ ĐÁNH GIÁ 5 MÔ HÌNH OCR\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Dataset: {annotation_file}\n")
        f.write(f"Tổng số mẫu: {len(samples)}\n")
        f.write(f"Batch size: 1 (sequential processing)\n")
        f.write(f"Device: cuda:0\n\n")
        
        # Bảng tổng quan
        f.write("=" * 70 + "\n")
        f.write("BẢNG TỔNG QUAN\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"{'Model':<40} {'Exact':<10} {'Word':<10} {'Char':<10} {'Time(ms)'}\n")
        f.write("-" * 70 + "\n")
        
        for result in all_results:
            f.write(f"{result['model']:<40} ")
            f.write(f"{result['exact_match_acc']:>5.2f}%    ")
            f.write(f"{result['word_acc']:>5.2f}%    ")
            f.write(f"{result['char_acc']:>5.2f}%    ")
            f.write(f"{result['avg_time_per_image']:>6.2f}\n")
        
        f.write("\n")
        
        # Chi tiết từng model
        f.write("\n" + "=" * 70 + "\n")
        f.write("CHI TIẾT TỪNG MÔ HÌNH\n")
        f.write("=" * 70 + "\n")
        
        for result in all_results:
            f.write("\n" + "-" * 70 + "\n")
            f.write(f"{result['model']}\n")
            f.write("-" * 70 + "\n")
            f.write(f"Tổng số mẫu: {result['total_samples']}\n\n")
            
            f.write("Kết quả:\n")
            f.write(f"  - Exact Match Accuracy:    {result['exact_match_acc']:.2f}% ")
            f.write(f"({result['exact_match']}/{result['total_samples']})\n")
            f.write(f"  - Word-level Accuracy:     {result['word_acc']:.2f}%\n")
            f.write(f"  - Character-level Accuracy: {result['char_acc']:.2f}%\n")
            f.write(f"  - Thời gian xử lý:          {result['elapsed_time']:.2f}s\n")
            f.write(f"  - Tốc độ trung bình:        {result['avg_time_per_image']:.2f}ms/ảnh\n\n")
            
            # Hiển thị một vài lỗi mẫu (tối đa 10)
            if result['errors']:
                f.write(f"Số lỗi: {len(result['errors'])}\n\n")
                f.write("Ví dụ các lỗi (tối đa 10):\n")
                for i, error in enumerate(result['errors'][:10], 1):
                    f.write(f"\n  {i}. {error['image']}\n")
                    f.write(f"     Label: {error['label']}\n")
                    f.write(f"     Pred:  {error['pred']}\n")
                    f.write(f"     Word Acc: {error['word_acc']*100:.1f}% | ")
                    f.write(f"Char Acc: {error['char_acc']*100:.1f}%\n")
        
        # So sánh
        f.write("\n\n" + "=" * 70 + "\n")
        f.write("SO SÁNH CÁC MÔ HÌNH\n")
        f.write("=" * 70 + "\n\n")
        
        # Sắp xếp theo exact match accuracy
        sorted_results = sorted(all_results, key=lambda x: x['exact_match_acc'], reverse=True)
        
        f.write("Xếp hạng theo Exact Match Accuracy:\n")
        for i, result in enumerate(sorted_results, 1):
            f.write(f"  {i}. {result['model']:<40} {result['exact_match_acc']:>6.2f}%\n")
        
        f.write("\nXếp hạng theo Word-level Accuracy:\n")
        sorted_word = sorted(all_results, key=lambda x: x['word_acc'], reverse=True)
        for i, result in enumerate(sorted_word, 1):
            f.write(f"  {i}. {result['model']:<40} {result['word_acc']:>6.2f}%\n")
        
        f.write("\nXếp hạng theo Character-level Accuracy:\n")
        sorted_char = sorted(all_results, key=lambda x: x['char_acc'], reverse=True)
        for i, result in enumerate(sorted_char, 1):
            f.write(f"  {i}. {result['model']:<40} {result['char_acc']:>6.2f}%\n")
        
        f.write("\nXếp hạng theo Tốc độ (nhanh nhất):\n")
        sorted_speed = sorted(all_results, key=lambda x: x['avg_time_per_image'])
        for i, result in enumerate(sorted_speed, 1):
            f.write(f"  {i}. {result['model']:<40} {result['avg_time_per_image']:>6.2f}ms/ảnh\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("KẾT LUẬN\n")
        f.write("=" * 70 + "\n\n")
        
        if all_results:
            best_model = sorted_results[0]
            fastest_model = sorted(all_results, key=lambda x: x['avg_time_per_image'])[0]
            
            f.write(f"Mô hình chính xác nhất (Exact Match): {best_model['model']}\n")
            f.write(f"  - Exact Match Accuracy: {best_model['exact_match_acc']:.2f}%\n")
            f.write(f"  - Word-level Accuracy: {best_model['word_acc']:.2f}%\n")
            f.write(f"  - Character-level Accuracy: {best_model['char_acc']:.2f}%\n")
            f.write(f"  - Tốc độ: {best_model['avg_time_per_image']:.2f}ms/ảnh\n\n")
            
            f.write(f"Mô hình nhanh nhất: {fastest_model['model']}\n")
            f.write(f"  - Tốc độ: {fastest_model['avg_time_per_image']:.2f}ms/ảnh\n")
            f.write(f"  - Exact Match Accuracy: {fastest_model['exact_match_acc']:.2f}%\n")
            f.write(f"  - Word-level Accuracy: {fastest_model['word_acc']:.2f}%\n")
            f.write(f"  - Character-level Accuracy: {fastest_model['char_acc']:.2f}%\n")
        
        f.write("\n" + "=" * 70 + "\n")
    
    print(f"✓ Hoàn thành! Kết quả đã lưu vào {output_file.name}")
    
    # In tóm tắt
    print("\n" + "=" * 70)
    print("TÓM TẮT KẾT QUẢ")
    print("=" * 70)
    print(f"{'Model':<40} {'Exact':<10} {'Word':<10} {'Char':<10} {'Time(ms)'}")
    print("-" * 70)
    for result in all_results:
        print(f"{result['model']:<40} {result['exact_match_acc']:>5.2f}%   {result['word_acc']:>5.2f}%   {result['char_acc']:>5.2f}%   {result['avg_time_per_image']:>6.2f}")


if __name__ == "__main__":
    main()
