"""
Script training wrapper cho VietOCR
Hỗ trợ train, resume, và monitor training progress
"""

import os
import sys
import argparse
from pathlib import Path

# Thêm vietocr vào Python path
BASE_DIR = Path(__file__).parent.parent  # Lên thư mục cha (VietOCR)
VIETOCR_DIR = BASE_DIR / "vietocr" / "vietocr"
sys.path.insert(0, str(BASE_DIR / "vietocr"))

from vietocr.model.trainer import Trainer
from vietocr.tool.config import Cfg


def check_gpu():
    """Kiểm tra GPU có available không"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"✓ GPU detected: {gpu_name}")
            print(f"  Memory: {gpu_memory:.1f} GB")
            return True
        else:
            print("⚠ No GPU detected. Training will use CPU (very slow!)")
            return False
    except ImportError:
        print("⚠ PyTorch not installed properly")
        return False


def check_dataset(config):
    """Kiểm tra dataset có sẵn sàng chưa"""
    data_root = config['dataset']['data_root']
    train_ann = os.path.join(data_root, config['dataset']['train_annotation'])
    valid_ann = os.path.join(data_root, config['dataset']['valid_annotation'])
    
    issues = []
    
    if not os.path.exists(train_ann):
        issues.append(f"❌ Không tìm thấy file training: {train_ann}")
    else:
        with open(train_ann, 'r', encoding='utf-8') as f:
            num_train = len(f.readlines())
        print(f"✓ File training: {train_ann} ({num_train:,} samples)")
    
    if not os.path.exists(valid_ann):
        issues.append(f"❌ Không tìm thấy file validation: {valid_ann}")
    else:
        with open(valid_ann, 'r', encoding='utf-8') as f:
            num_val = len(f.readlines())
        print(f"✓ File validation: {valid_ann} ({num_val:,} samples)")
    
    if issues:
        for issue in issues:
            print(issue)
        print("\n💡 Gợi ý: Chạy 'python prepare_dataset.py' để tạo file annotation")
        return False
    
    return True


def create_directories(config):
    """Tạo các thư mục cần thiết"""
    export_path = config['trainer']['export']
    checkpoint_path = config['trainer']['checkpoint']
    log_path = config['trainer']['log']
    
    for path in [export_path, checkpoint_path, log_path]:
        dir_path = os.path.dirname(path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
            print(f"✓ Created directory: {dir_path}")


def train_model(config_file, checkpoint=None, from_scratch=False):
    """
    Train model
    
    Args:
        config_file: Đường dẫn file config
        checkpoint: Đường dẫn checkpoint để resume training
        from_scratch: Train từ đầu (không dùng pretrained weights)
    """
    print("=" * 70)
    print("VIETOCR TRAINING - VIETNAMESE NAMES DATASET")
    print("=" * 70)
    
    # Load config
    print(f"\n📄 Loading config: {config_file}")
    config = Cfg.load_config_from_file(config_file)
    
    # Check GPU
    print("\n🖥️  Checking hardware...")
    has_gpu = check_gpu()
    if not has_gpu and config['device'].startswith('cuda'):
        print("⚠ Config yêu cầu GPU nhưng không tìm thấy. Switching to CPU...")
        config['device'] = 'cpu'
    
    # Check dataset
    print("\n📊 Checking dataset...")
    if not check_dataset(config):
        return
    
    # Create directories
    print("\n📁 Creating directories...")
    create_directories(config)
    
    # Print training info
    print("\n⚙️  Training configuration:")
    print(f"  - Model: {config['seq_modeling']} + {config['backbone']}")
    print(f"  - Batch size: {config['trainer']['batch_size']}")
    print(f"  - Total iterations: {config['trainer']['iters']:,}")
    print(f"  - Learning rate: {config['optimizer']['max_lr']}")
    print(f"  - Image size: {config['dataset']['image_height']}px height")
    print(f"  - Augmentation: {'ON' if config['aug']['image_aug'] else 'OFF'}")
    print(f"  - Device: {config['device']}")
    
    if from_scratch:
        print(f"  - Mode: Train from scratch (no pretrained)")
    elif 'pretrain' in config:
        print(f"  - Pretrained: {config['pretrain']}")
    
    # Initialize trainer
    print("\n🚀 Initializing trainer...")
    trainer = Trainer(config, pretrained=(not from_scratch))
    
    # Load checkpoint if provided
    if checkpoint:
        if os.path.exists(checkpoint):
            print(f"📥 Loading checkpoint: {checkpoint}")
            trainer.load_checkpoint(checkpoint)
            print(f"   Resuming from iteration {trainer.iter}")
        else:
            print(f"⚠ Checkpoint not found: {checkpoint}")
            print("   Starting from iteration 0")
    
    # Start training
    print("\n" + "=" * 70)
    print("🎯 STARTING TRAINING")
    print("=" * 70)
    print("\nPress Ctrl+C to stop training and save checkpoint\n")
    
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n\n⏸️  Training interrupted by user")
        print("💾 Saving checkpoint...")
        trainer.save_checkpoint(config['trainer']['checkpoint'])
        print(f"✓ Checkpoint saved: {config['trainer']['checkpoint']}")
        print(f"✓ Best weights: {config['trainer']['export']}")
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("✓ TRAINING COMPLETED")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description='Train VietOCR model on Vietnamese Names dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train mới từ pretrained weights
  python train_vietnamese_names.py
  
  # Train từ đầu (không dùng pretrained)
  python train_vietnamese_names.py --from-scratch
  
  # Resume training từ checkpoint
  python train_vietnamese_names.py --checkpoint checkpoint/vietnamese_names_checkpoint.pth
  
  # Sử dụng config khác
  python train_vietnamese_names.py --config my_config.yml
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config_vietnamese_names.yml',
        help='Đường dẫn file config (default: config_vietnamese_names.yml)'
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Đường dẫn checkpoint để resume training'
    )
    
    parser.add_argument(
        '--from-scratch',
        action='store_true',
        help='Train từ đầu, không sử dụng pretrained weights'
    )
    
    args = parser.parse_args()
    
    # Check config file exists
    if not os.path.exists(args.config):
        print(f"❌ Config file not found: {args.config}")
        print("\n💡 Available config files:")
        for f in Path('.').glob('*.yml'):
            print(f"   - {f}")
        sys.exit(1)
    
    # Start training
    train_model(
        config_file=args.config,
        checkpoint=args.checkpoint,
        from_scratch=args.from_scratch
    )


if __name__ == "__main__":
    main()
