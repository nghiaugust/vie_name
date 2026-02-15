"""
Script test model sau khi training
Predict text từ ảnh sử dụng model đã train
"""

import os
import sys
import argparse
from pathlib import Path
from PIL import Image

# Thêm vietocr vào Python path
BASE_DIR = Path(__file__).parent.parent  # Lên thư mục cha (VietOCR)
sys.path.insert(0, str(BASE_DIR / "vietocr"))

from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg


def predict_single_image(predictor, image_path, show_prob=False):
    """
    Predict text từ 1 ảnh
    
    Args:
        predictor: Predictor object
        image_path: Đường dẫn ảnh
        show_prob: Hiển thị confidence score
    """
    if not os.path.exists(image_path):
        print(f"❌ Không tìm thấy ảnh: {image_path}")
        return None
    
    try:
        img = Image.open(image_path)
        
        if show_prob:
            text, prob = predictor.predict(img, return_prob=True)
            print(f"📄 {image_path}")
            print(f"   Text: {text}")
            print(f"   Confidence: {prob:.4f}")
            return text, prob
        else:
            text = predictor.predict(img)
            print(f"📄 {image_path} → {text}")
            return text
            
    except Exception as e:
        print(f"❌ Lỗi khi predict {image_path}: {e}")
        return None


def predict_batch(predictor, image_folder, limit=None, show_prob=False):
    """
    Predict nhiều ảnh trong thư mục
    
    Args:
        predictor: Predictor object
        image_folder: Thư mục chứa ảnh
        limit: Giới hạn số ảnh (None = tất cả)
        show_prob: Hiển thị confidence score
    """
    if not os.path.exists(image_folder):
        print(f"❌ Không tìm thấy thư mục: {image_folder}")
        return
    
    # Tìm tất cả file ảnh
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        image_files.extend(Path(image_folder).glob(ext))
    
    if not image_files:
        print(f"❌ Không tìm thấy ảnh nào trong: {image_folder}")
        return
    
    if limit:
        image_files = image_files[:limit]
    
    print(f"\n🔍 Tìm thấy {len(image_files)} ảnh")
    print("=" * 70)
    
    results = []
    for img_path in image_files:
        result = predict_single_image(predictor, str(img_path), show_prob)
        if result:
            results.append((str(img_path), result))
    
    print("=" * 70)
    print(f"✓ Đã predict {len(results)}/{len(image_files)} ảnh")


def test_with_annotation(predictor, annotation_file, data_root, limit=100):
    """
    Test model với file annotation để tính accuracy
    
    Args:
        predictor: Predictor object
        annotation_file: File annotation
        data_root: Thư mục gốc chứa ảnh
        limit: Số lượng mẫu test (None = tất cả)
    """
    if not os.path.exists(annotation_file):
        print(f"❌ Không tìm thấy file annotation: {annotation_file}")
        return
    
    print(f"\n📊 Testing với annotation file: {annotation_file}")
    
    with open(annotation_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    if limit:
        lines = lines[:limit]
    
    print(f"Testing trên {len(lines)} mẫu...")
    print("=" * 70)
    
    correct_full = 0
    correct_chars = 0
    total_chars = 0
    errors = []
    
    for i, line in enumerate(lines):
        parts = line.strip().split('\t')
        if len(parts) < 2:
            continue
        
        img_path = parts[0]
        if img_path.startswith('./'):
            img_path = img_path[2:]
        
        full_path = os.path.join(data_root, img_path)
        ground_truth = parts[1]
        
        if not os.path.exists(full_path):
            continue
        
        try:
            img = Image.open(full_path)
            prediction = predictor.predict(img)
            
            # Tính accuracy full sequence
            if prediction == ground_truth:
                correct_full += 1
            else:
                errors.append({
                    'image': img_path,
                    'ground_truth': ground_truth,
                    'prediction': prediction
                })
            
            # Tính accuracy per character
            for c1, c2 in zip(prediction, ground_truth):
                if c1 == c2:
                    correct_chars += 1
                total_chars += 1
            
            # Xử lý trường hợp độ dài khác nhau
            total_chars += abs(len(prediction) - len(ground_truth))
            
            if (i + 1) % 10 == 0:
                print(f"Progress: {i+1}/{len(lines)}", end='\r')
                
        except Exception as e:
            print(f"Lỗi tại {img_path}: {e}")
    
    print()
    print("=" * 70)
    print("\n📈 KẾT QUẢ TEST:")
    print(f"  - Tổng số mẫu: {len(lines)}")
    print(f"  - Accuracy (full sequence): {correct_full/len(lines)*100:.2f}%")
    print(f"  - Accuracy (per character): {correct_chars/total_chars*100:.2f}%")
    
    if errors:
        print(f"\n❌ {len(errors)} lỗi. Ví dụ 5 lỗi đầu tiên:")
        for err in errors[:5]:
            print(f"  Image: {err['image']}")
            print(f"    GT:   {err['ground_truth']}")
            print(f"    Pred: {err['prediction']}")
            print()


def main():
    parser = argparse.ArgumentParser(
        description='Test VietOCR model sau khi training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Predict 1 ảnh
  python test_model.py --image dataset/images/1.jpg
  
  # Predict nhiều ảnh trong thư mục
  python test_model.py --folder dataset/images --limit 10
  
  # Test accuracy với annotation file
  python test_model.py --test --annotation dataset/val_annotation.txt
  
  # Sử dụng config khác
  python test_model.py --config my_config.yml --image test.jpg
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config_vietnamese_names.yml',
        help='File config'
    )
    
    parser.add_argument(
        '--weights',
        type=str,
        default='../weights/vietnamese_names_best.pth',
        help='File weights của model đã train'
    )
    
    parser.add_argument(
        '--image',
        type=str,
        help='Predict 1 ảnh'
    )
    
    parser.add_argument(
        '--folder',
        type=str,
        help='Predict nhiều ảnh trong thư mục'
    )
    
    parser.add_argument(
        '--test',
        action='store_true',
        help='Test accuracy với annotation file'
    )
    
    parser.add_argument(
        '--annotation',
        type=str,
        default='../dataset/val_annotation.txt',
        help='File annotation để test'
    )
    
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Giới hạn số ảnh test'
    )
    
    parser.add_argument(
        '--prob',
        action='store_true',
        help='Hiển thị confidence score'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device (cuda:0 hoặc cpu). Mặc định dùng theo config'
    )
    
    parser.add_argument(
        '--beamsearch',
        action='store_true',
        help='Sử dụng beam search (chậm hơn nhưng chính xác hơn)'
    )
    
    args = parser.parse_args()
    
    # Check config file
    if not os.path.exists(args.config):
        print(f"❌ Không tìm thấy config: {args.config}")
        return
    
    # Check weights file
    if not os.path.exists(args.weights):
        print(f"❌ Không tìm thấy weights: {args.weights}")
        print("💡 Gợi ý: Bạn cần train model trước khi test")
        return
    
    print("=" * 70)
    print("VIETOCR MODEL TESTING")
    print("=" * 70)
    
    # Load config
    print(f"\n📄 Loading config: {args.config}")
    config = Cfg.load_config_from_file(args.config)
    
    # Override weights
    config['weights'] = args.weights
    print(f"📦 Loading weights: {args.weights}")
    
    # Override device if specified
    if args.device:
        config['device'] = args.device
        print(f"🖥️  Device: {args.device}")
    else:
        print(f"🖥️  Device: {config['device']}")
    
    # Override beamsearch if specified
    if args.beamsearch:
        config['predictor']['beamsearch'] = True
        print("🔍 Beam search: ON (chính xác hơn, chậm hơn)")
    
    # Create predictor
    print("\n🚀 Initializing predictor...")
    try:
        predictor = Predictor(config)
        print("✓ Predictor ready!")
    except Exception as e:
        print(f"❌ Lỗi khi tạo predictor: {e}")
        return
    
    print()
    
    # Execute based on mode
    if args.test:
        # Test mode
        test_with_annotation(
            predictor,
            args.annotation,
            config['dataset']['data_root'],
            args.limit or 100
        )
    elif args.image:
        # Single image prediction
        predict_single_image(predictor, args.image, args.prob)
    elif args.folder:
        # Batch prediction
        predict_batch(predictor, args.folder, args.limit, args.prob)
    else:
        print("❌ Vui lòng chọn mode: --image, --folder, hoặc --test")
        print("Ví dụ: python test_model.py --image test.jpg")


if __name__ == "__main__":
    main()
