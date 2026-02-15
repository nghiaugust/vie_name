"""
Script visualize kết quả training
Hiển thị ảnh kèm prediction và ground truth để đánh giá model
"""

import os
import sys
import argparse
import random
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# Thêm vietocr vào Python path
BASE_DIR = Path(__file__).parent.parent  # Lên thư mục cha (VietOCR)
sys.path.insert(0, str(BASE_DIR / "vietocr"))

from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg


def visualize_predictions(
    predictor,
    annotation_file,
    data_root,
    num_samples=20,
    output_dir="visualization",
    show_errors_only=False
):
    """
    Visualize predictions: tạo ảnh có text prediction và ground truth
    
    Args:
        predictor: Predictor object
        annotation_file: File annotation
        data_root: Thư mục gốc chứa ảnh
        num_samples: Số lượng mẫu visualize
        output_dir: Thư mục lưu kết quả
        show_errors_only: Chỉ hiển thị các trường hợp sai
    """
    if not os.path.exists(annotation_file):
        print(f"❌ Không tìm thấy file annotation: {annotation_file}")
        return
    
    # Đọc annotation
    with open(annotation_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Shuffle để lấy random samples
    random.shuffle(lines)
    
    # Tạo thư mục output
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n📊 Visualizing predictions...")
    print(f"📁 Output directory: {output_dir}")
    print("=" * 70)
    
    correct_count = 0
    error_count = 0
    visualized = 0
    
    for line in lines:
        if visualized >= num_samples:
            break
        
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
            # Predict
            img = Image.open(full_path)
            prediction, prob = predictor.predict(img, return_prob=True)
            
            # Check if correct
            is_correct = (prediction == ground_truth)
            
            if show_errors_only and is_correct:
                continue
            
            if is_correct:
                correct_count += 1
                status = "✓ CORRECT"
                color = "green"
            else:
                error_count += 1
                status = "✗ ERROR"
                color = "red"
            
            # Tạo ảnh visualization
            img_rgb = img.convert('RGB')
            width, height = img_rgb.size
            
            # Tạo canvas mới cao hơn để chứa text
            new_height = height + 80
            canvas = Image.new('RGB', (width, new_height), color='white')
            canvas.paste(img_rgb, (0, 0))
            
            # Vẽ text
            draw = ImageDraw.Draw(canvas)
            
            try:
                # Try to use Arial font
                font = ImageFont.truetype("arial.ttf", 14)
            except:
                # Fall back to default font
                font = ImageFont.load_default()
            
            # Vẽ ground truth
            draw.text((5, height + 5), f"GT:   {ground_truth}", fill='black', font=font)
            
            # Vẽ prediction với màu tùy theo đúng/sai
            pred_color = 'green' if is_correct else 'red'
            draw.text((5, height + 25), f"Pred: {prediction}", fill=pred_color, font=font)
            
            # Vẽ confidence
            draw.text((5, height + 45), f"Conf: {prob:.4f}", fill='blue', font=font)
            
            # Vẽ status
            draw.text((5, height + 65), status, fill=pred_color, font=font)
            
            # Lưu ảnh
            output_filename = f"{visualized:04d}_{status.replace(' ', '_').replace('✓', 'correct').replace('✗', 'error')}.jpg"
            output_path = os.path.join(output_dir, output_filename)
            canvas.save(output_path)
            
            print(f"[{visualized+1:3d}] {status} | GT: {ground_truth} | Pred: {prediction} | Conf: {prob:.4f}")
            
            visualized += 1
            
        except Exception as e:
            print(f"❌ Lỗi tại {img_path}: {e}")
    
    print("=" * 70)
    print(f"\n✓ Đã visualize {visualized} mẫu")
    print(f"  - Correct: {correct_count} ({correct_count/visualized*100:.1f}%)")
    print(f"  - Errors: {error_count} ({error_count/visualized*100:.1f}%)")
    print(f"\n📁 Kết quả đã lưu tại: {output_dir}")


def create_comparison_grid(
    predictor,
    annotation_file,
    data_root,
    num_samples=16,
    output_file="comparison_grid.jpg"
):
    """
    Tạo grid so sánh nhiều predictions trên 1 ảnh
    
    Args:
        predictor: Predictor object
        annotation_file: File annotation
        data_root: Thư mục gốc
        num_samples: Số mẫu (phải là số chính phương: 4, 9, 16, 25...)
        output_file: File output
    """
    import math
    
    if not os.path.exists(annotation_file):
        print(f"❌ Không tìm thấy file annotation: {annotation_file}")
        return
    
    # Tính grid size
    grid_size = int(math.sqrt(num_samples))
    if grid_size * grid_size != num_samples:
        print(f"⚠ num_samples phải là số chính phương. Điều chỉnh thành {grid_size * grid_size}")
        num_samples = grid_size * grid_size
    
    # Đọc annotation
    with open(annotation_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    random.shuffle(lines)
    
    print(f"\n🎨 Creating comparison grid ({grid_size}x{grid_size})...")
    
    samples = []
    for line in lines:
        if len(samples) >= num_samples:
            break
        
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
            prediction, prob = predictor.predict(img, return_prob=True)
            
            samples.append({
                'image': img,
                'ground_truth': ground_truth,
                'prediction': prediction,
                'prob': prob
            })
        except:
            continue
    
    if len(samples) < num_samples:
        print(f"⚠ Chỉ tìm thấy {len(samples)} mẫu hợp lệ")
    
    # Tính kích thước grid
    cell_width = 300
    cell_height = 120
    grid_width = cell_width * grid_size
    grid_height = cell_height * grid_size
    
    # Tạo canvas
    canvas = Image.new('RGB', (grid_width, grid_height), color='white')
    draw = ImageDraw.Draw(canvas)
    
    try:
        font = ImageFont.truetype("arial.ttf", 12)
    except:
        font = ImageFont.load_default()
    
    # Vẽ từng cell
    for idx, sample in enumerate(samples):
        row = idx // grid_size
        col = idx % grid_size
        
        x = col * cell_width
        y = row * cell_height
        
        # Resize ảnh
        img = sample['image'].convert('RGB')
        img.thumbnail((cell_width - 10, 60), Image.Resampling.LANCZOS)
        
        # Paste ảnh
        canvas.paste(img, (x + 5, y + 5))
        
        # Vẽ text
        gt_text = f"GT: {sample['ground_truth'][:30]}"
        pred_text = f"Pred: {sample['prediction'][:30]}"
        
        is_correct = (sample['prediction'] == sample['ground_truth'])
        pred_color = 'green' if is_correct else 'red'
        
        draw.text((x + 5, y + 70), gt_text, fill='black', font=font)
        draw.text((x + 5, y + 85), pred_text, fill=pred_color, font=font)
        draw.text((x + 5, y + 100), f"Conf: {sample['prob']:.3f}", fill='blue', font=font)
        
        # Vẽ border
        draw.rectangle([x, y, x + cell_width - 1, y + cell_height - 1], outline='gray')
    
    # Lưu file
    canvas.save(output_file)
    print(f"✓ Grid đã được lưu tại: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize predictions của VietOCR model',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--config', type=str, default='config_vietnamese_names.yml')
    parser.add_argument('--weights', type=str, default='../weights/vietnamese_names_best.pth')
    parser.add_argument('--annotation', type=str, default='../dataset/val_annotation.txt')
    parser.add_argument('--data-root', type=str, default='../dataset/')
    parser.add_argument('--num-samples', type=int, default=20)
    parser.add_argument('--output-dir', type=str, default='../visualization')
    parser.add_argument('--errors-only', action='store_true', help='Chỉ hiển thị errors')
    parser.add_argument('--grid', action='store_true', help='Tạo comparison grid')
    parser.add_argument('--grid-output', type=str, default='comparison_grid.jpg')
    
    args = parser.parse_args()
    
    # Load config
    print("=" * 70)
    print("VIETOCR VISUALIZATION")
    print("=" * 70)
    print(f"\n📄 Loading config: {args.config}")
    
    config = Cfg.load_config_from_file(args.config)
    config['weights'] = args.weights
    
    # Create predictor
    print(f"📦 Loading weights: {args.weights}")
    print("🚀 Initializing predictor...")
    
    predictor = Predictor(config)
    print("✓ Ready!")
    
    if args.grid:
        # Create grid
        create_comparison_grid(
            predictor,
            args.annotation,
            args.data_root,
            args.num_samples,
            args.grid_output
        )
    else:
        # Visualize individual predictions
        visualize_predictions(
            predictor,
            args.annotation,
            args.data_root,
            args.num_samples,
            args.output_dir,
            args.errors_only
        )


if __name__ == "__main__":
    main()
