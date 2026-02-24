"""
Trainer cho MobileNet-SVTR-CTC Model
Training script tối ưu cho CTC loss
Tích hợp với VietOCR framework
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import yaml
import os
import sys
import time
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm

# Thêm vietocr vào Python path
BASE_DIR = Path(__file__).parent.parent  # Lên thư mục cha (VietOCR)
VIETOCR_DIR = BASE_DIR / "vietocr" / "vietocr"
sys.path.insert(0, str(BASE_DIR / "vietocr"))

# Import components
from vietocr.model.mobilenet_svtr_ctc import mobilenet_svtr_ctc, mobilenet_svtr_ctc_light
from vietocr.model.vocab_ctc import VocabCTC  # CTC-compatible vocab
from vietocr.loader.dataloader import OCRDataset
from vietocr.loader.collator_ctc import CollatorCTC  # CTC-specific collator
from vietocr.tool.logger import Logger
from vietocr.tool.utils import compute_accuracy


def check_gpu():
    """Kiểm tra GPU có available không"""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✓ GPU detected: {gpu_name}")
        print(f"  Memory: {gpu_memory:.1f} GB")
        return True
    else:
        print("⚠ No GPU detected. Training will use CPU (very slow!)")
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
        print("\n💡 Gợi ý: Chuẩn bị dataset với format:")
        print("   <image_filename><TAB><label>")
        return False
    
    return True


def create_directories(config):
    """Tạo các thư mục cần thiết"""
    # Directories are now created automatically in CTCTrainer.__init__()
    # This function is kept for backward compatibility but does nothing
    pass


class CTCTrainer:
    """Trainer cho CTC-based models"""
    
    def __init__(self, config_path):
        """
        Args:
            config_path: đường dẫn đến file config yaml
        """
        # Load config
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() 
                                  and self.config['device'] == 'cuda' else 'cpu')
        print(f"Using device: {self.device}")
        
        # Build vocab (CTC-compatible)
        self.vocab = VocabCTC(self.config['vocab'])
        self.config['vocab_size'] = len(self.vocab)  # Includes blank at index 0
        self.blank_idx = 0  # CTC blank is always at index 0
        
        print(f"Vocabulary size: {self.config['vocab_size']} (includes blank token at index 0)")
        
        # Build model
        self.model = self._build_model()
        self.model = self.model.to(self.device)
        
        # Count parameters
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Model parameters: {num_params:,}")
        
        # Setup optimizer
        self.optimizer = self._build_optimizer()
        
        # Setup scheduler
        self.scheduler = self._build_scheduler()
        
        # Setup loss
        self.criterion = nn.CTCLoss(blank=self.blank_idx, reduction='mean', 
                                   zero_infinity=True)
        
        # Setup AMP (Automatic Mixed Precision)
        self.use_amp = self.config['train'].get('use_amp', True) and self.device.type == 'cuda'
        if self.use_amp:
            self.scaler = GradScaler(device='cuda')
            print(f"✓ AMP enabled (Mixed Precision Training)")
        else:
            self.scaler = None
            print(f"✓ AMP disabled (Full Precision Training)")
        
        # Setup data loaders
        self.train_loader, self.valid_loader = self._build_dataloaders()
        
        # Setup logger
        self.logger = self._build_logger()
        
        # Setup save directory - lưu vào VietOCR/weights/mobilenet_svtr_ctc
        self.save_dir = BASE_DIR / 'weights' / 'mobilenet_svtr_ctc'
        self.save_dir.mkdir(parents=True, exist_ok=True)
        print(f"✓ Weights directory: {self.save_dir}")
        
        # Training state
        self.epoch = 0
        self.best_valid_loss = float('inf')
        self.best_full_seq_acc = 0.0
        self.best_per_char_acc = 0.0
        self.patience_counter = 0
        
    def _build_model(self):
        """Build model từ config"""
        model_name = self.config['model']['name']
        model_config = self.config['model']
        
        if model_name == 'mobilenet_svtr_ctc':
            model = mobilenet_svtr_ctc(
                vocab_size=self.config['vocab_size'],
                hidden=model_config.get('backbone_hidden', 256),
                svtr_depth=model_config.get('svtr_depth', 2),
                svtr_heads=model_config.get('svtr_heads', 8),
                dropout=model_config.get('dropout', 0.1),
                pretrained=model_config.get('pretrained', False)
            )
        elif model_name == 'mobilenet_svtr_ctc_light':
            model = mobilenet_svtr_ctc_light(
                vocab_size=self.config['vocab_size'],
                hidden=model_config.get('backbone_hidden', 128),
                svtr_depth=model_config.get('svtr_depth', 1),
                svtr_heads=model_config.get('svtr_heads', 4),
                dropout=model_config.get('dropout', 0.1)
            )
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        return model
    
    def _build_optimizer(self):
        """Build optimizer"""
        optim_name = self.config['train']['optimizer'].lower()
        lr = self.config['train']['learning_rate']
        weight_decay = self.config['train'].get('weight_decay', 0.0)
        
        if optim_name == 'adam':
            optimizer = optim.Adam(self.model.parameters(), lr=lr, 
                                 weight_decay=weight_decay)
        elif optim_name == 'adamw':
            optimizer = optim.AdamW(self.model.parameters(), lr=lr,
                                   weight_decay=weight_decay)
        elif optim_name == 'sgd':
            optimizer = optim.SGD(self.model.parameters(), lr=lr,
                                momentum=0.9, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {optim_name}")
        
        return optimizer
    
    def _build_scheduler(self):
        """Build learning rate scheduler"""
        scheduler_name = self.config['train'].get('scheduler', 'cosine')
        epochs = self.config['train']['epochs']
        min_lr = self.config['train'].get('min_lr', 1e-6)
        
        if scheduler_name == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=epochs, eta_min=min_lr
            )
        elif scheduler_name == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=30, gamma=0.1
            )
        elif scheduler_name == 'reduce':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=5
            )
        else:
            scheduler = None
        
        return scheduler
    
    def _build_dataloaders(self):
        """Build data loaders"""
        dataset_config = self.config['dataset']
        train_config = self.config['train']
        
        # LMDB paths
        train_lmdb = os.path.join(dataset_config['data_root'], 'train_lmdb')
        valid_lmdb = os.path.join(dataset_config['data_root'], 'valid_lmdb')
        
        # Create train dataset
        train_dataset = OCRDataset(
            lmdb_path=train_lmdb,
            root_dir=dataset_config['data_root'],
            annotation_path=dataset_config['train_annotation'],
            vocab=self.vocab,
            image_height=train_config['image_height'],
            image_min_width=32,
            image_max_width=train_config['image_width'],
            transform=None  # TODO: Add augmentation if needed
        )
        
        # Create validation dataset
        valid_dataset = OCRDataset(
            lmdb_path=valid_lmdb,
            root_dir=dataset_config['data_root'],
            annotation_path=dataset_config['valid_annotation'],
            vocab=self.vocab,
            image_height=train_config['image_height'],
            image_min_width=32,
            image_max_width=train_config['image_width'],
            transform=None
        )
        
        print(f"✓ Train dataset: {len(train_dataset):,} samples")
        print(f"✓ Valid dataset: {len(valid_dataset):,} samples")
        
        # CTC Collator for batching (simpler than VietOCR Collator)
        collate_fn = CollatorCTC()
        
        # DataLoaders
        # Note: num_workers=0 on Windows to avoid LMDB pickle errors
        num_workers = 0 if sys.platform == 'win32' else self.config.get('num_workers', 4)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=train_config['batch_size'],
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=train_config.get('valid_batch_size', train_config['batch_size']),
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        return train_loader, valid_loader
    
    def _build_logger(self):
        """Build logger"""
        # Lưu log vào VietOCR/logs/
        log_dir = BASE_DIR / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Tạo tên file log với timestamp để dễ phân biệt các lần training
        import datetime
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        model_name = self.config['model']['name']
        log_file = log_dir / f'{model_name}_{timestamp}.log'
        
        logger = Logger(str(log_file))
        print(f"✓ Log file: {log_file}")
        return logger
    
    def train_epoch(self):
        """Train one epoch"""
        self.model.train()
        total_loss = 0
        total_samples = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch+1} [Train]", ncols=100)
        
        for batch_idx, batch in enumerate(pbar):
            # Get batch data - CollatorCTC returns dict
            images = batch['img'].to(self.device)  # (N, C, H, W)
            tgt_output = batch['tgt_output']  # (N, max_len) padded with 0
            # Note: batch['tgt_lengths'] available but not needed (we compute from tgt_output)
            
            # Forward pass with AMP
            with autocast(device_type="cuda", enabled=self.use_amp):
                logits = self.model(images)  # (N, T, vocab_size)
                
                # Prepare for CTC loss
                # CTC expects: log_probs (T, N, C), targets (N, S), input_lengths (N), target_lengths (N)
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)  # (N, T, C)
                log_probs = log_probs.permute(1, 0, 2)  # (T, N, C)
                
                # Input lengths (all same after backbone)
                batch_size = images.size(0)
                input_lengths = torch.full((batch_size,), logits.size(1), dtype=torch.long)
                
                # Target lengths and targets (CTC format)
                target_lengths = []
                targets_list = []
                valid_batch_indices = []  # Track valid samples
                
                for idx, tgt in enumerate(tgt_output):
                    # CTC vocab: index 0 is blank, actual chars start from 1
                    # Remove blank tokens (0) - these are padding in VietOCR format
                    valid_tokens = [t for t in tgt.tolist() if t > 0]
                    
                    # CRITICAL: Validate target is not empty
                    if len(valid_tokens) == 0:
                        print(f"⚠ Warning: Empty target at batch index {idx}, skipping...")
                        continue
                    
                    # CRITICAL: Validate target_length <= input_length (CTC requirement)
                    if len(valid_tokens) > logits.size(1):
                        print(f"⚠ Warning: Target longer than input ({len(valid_tokens)} > {logits.size(1)}), truncating...")
                        valid_tokens = valid_tokens[:logits.size(1)]
                    
                    target_lengths.append(len(valid_tokens))
                    targets_list.extend(valid_tokens)
                    valid_batch_indices.append(idx)
                
                # Skip batch if no valid targets
                if len(target_lengths) == 0:
                    print(f"⚠ Warning: No valid targets in batch, skipping...")
                    continue
                
                # Use only valid samples
                if len(valid_batch_indices) < batch_size:
                    print(f"⚠ Warning: Only {len(valid_batch_indices)}/{batch_size} valid samples")
                    log_probs = log_probs[:, valid_batch_indices, :]
                    input_lengths = torch.full((len(valid_batch_indices),), logits.size(1), dtype=torch.long)
                else:
                    input_lengths = torch.full((batch_size,), logits.size(1), dtype=torch.long)
                
                target_lengths = torch.tensor(target_lengths, dtype=torch.long)
                targets = torch.tensor(targets_list, dtype=torch.long)
                
                # Compute CTC loss (CTC doesn't support Half precision, cast to float32)
                log_probs_fp32 = log_probs.float()
                loss = self.criterion(log_probs_fp32, targets, input_lengths, target_lengths)
            
            # Backward with AMP
            self.optimizer.zero_grad()
            
            if self.use_amp:
                # AMP: Scale loss, backward, unscale, clip gradients, step
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)  # Unscale before clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Normal training
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / total_samples if total_samples > 0 else 0
        
        return avg_loss
    
    def validate(self):
        """Validate model"""
        self.model.eval()
        total_loss = 0
        total_samples = 0
        all_predictions = []
        all_ground_truths = []
        
        pbar = tqdm(self.valid_loader, desc=f"Epoch {self.epoch+1} [Valid]", ncols=100)
        
        with torch.no_grad():
            for batch in pbar:
                # Get batch data - CollatorCTC returns dict
                images = batch['img'].to(self.device)
                tgt_output = batch['tgt_output']
                
                # Forward with AMP
                with autocast(device_type="cuda", enabled=self.use_amp):
                    logits = self.model(images)
                
                # Prepare for CTC loss
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                log_probs = log_probs.permute(1, 0, 2)  # (T, N, C)
                
                batch_size = images.size(0)
                input_lengths = torch.full((batch_size,), logits.size(1), dtype=torch.long)
                
                # Target lengths and targets (CTC format)
                target_lengths = []
                targets_list = []
                ground_truths = []
                
                for tgt in tgt_output:
                    # Remove blank tokens (0)
                    valid_tokens = [t for t in tgt.tolist() if t > 0]
                    
                    # Skip empty targets
                    if len(valid_tokens) == 0:
                        target_lengths.append(1)  # Dummy length to avoid crash
                        targets_list.append(1)  # Dummy token
                        ground_truths.append("")  # Empty string
                        continue
                    
                    target_lengths.append(len(valid_tokens))
                    targets_list.extend(valid_tokens)
                    # Decode using CTC vocab (automatically skips blank)
                    ground_truths.append(self.vocab.decode(valid_tokens))
                
                target_lengths = torch.tensor(target_lengths, dtype=torch.long)
                targets = torch.tensor(targets_list, dtype=torch.long)
                
                # Loss (CTC doesn't support Half precision, cast to float32)
                log_probs_fp32 = log_probs.float()
                loss = self.criterion(log_probs_fp32, targets, input_lengths, target_lengths)
                
                # Decode predictions (greedy for validation speed)
                preds = torch.argmax(logits, dim=-1)  # (N, T)
                
                for pred in preds:
                    # CTC greedy decode: remove blanks and consecutive duplicates
                    decoded_seq = []
                    prev = None
                    for token in pred.tolist():
                        # Skip blank (0) and consecutive duplicates
                        if token != self.blank_idx and token != prev:
                            decoded_seq.append(token)
                        prev = token
                    # Decode using CTC vocab (blank already removed)
                    pred_text = self.vocab.decode(decoded_seq)
                    all_predictions.append(pred_text)
                
                all_ground_truths.extend(ground_truths)
                
                total_loss += loss.item() * batch_size
                total_samples += batch_size
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / total_samples if total_samples > 0 else 0
        
        # Compute accuracy (2 loại)
        # 1. Full Sequence Accuracy: Cả câu phải đúng 100% → khắt khe
        full_seq_acc = compute_accuracy(all_ground_truths, all_predictions, mode='full_sequence')
        
        # 2. Per Character Accuracy: Tính theo từng ký tự → dễ dàng hơn
        per_char_acc = compute_accuracy(all_ground_truths, all_predictions, mode='per_char')
        
        # Show some examples
        if len(all_predictions) > 0:
            print("\n📝 Sample predictions:")
            for i in range(min(3, len(all_predictions))):
                print(f"   GT:   '{all_ground_truths[i]}'")
                print(f"   Pred: '{all_predictions[i]}'")
        
        return avg_loss, full_seq_acc, per_char_acc
    
    def save_checkpoint(self, name='checkpoint.pth'):
        """Save checkpoint"""
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_valid_loss': self.best_valid_loss,
            'best_full_seq_acc': self.best_full_seq_acc,
            'best_per_char_acc': self.best_per_char_acc,
            'config': self.config
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save AMP scaler state
        if self.use_amp and self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        save_path = self.save_dir / name
        torch.save(checkpoint, save_path)
        print(f"Saved checkpoint to {save_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint"""
        # PyTorch 2.6+ requires weights_only=False for full checkpoints
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.best_valid_loss = checkpoint['best_valid_loss']
        self.best_full_seq_acc = checkpoint.get('best_full_seq_acc', 0.0)
        self.best_per_char_acc = checkpoint.get('best_per_char_acc', 0.0)
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load AMP scaler state
        if self.use_amp and self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            print("   ✓ AMP scaler state loaded")
        
        print(f"Loaded checkpoint from {checkpoint_path}")
        print(f"   Resuming from epoch {self.epoch + 1}")
        print(f"   Best Full Seq Acc so far: {self.best_full_seq_acc*100:.2f}%")
        print(f"   Best Per Char Acc so far: {self.best_per_char_acc*100:.2f}%")
    
    def train(self):
        """Main training loop"""
        num_epochs = self.config['train']['epochs']
        patience = self.config['train'].get('patience', 15)
        
        print("\n" + "=" * 70)
        print("🎯 STARTING TRAINING")
        print("=" * 70)
        print(f"Total epochs: {num_epochs}")
        print(f"Patience: {patience}")
        print(f"Device: {self.device}")
        print("Press Ctrl+C to stop and save checkpoint")
        print("=" * 70 + "\n")
        
        # Log training configuration
        self.logger.log("=" * 70)
        self.logger.log("MOBILENET-SVTR-CTC TRAINING LOG")
        self.logger.log("=" * 70)
        self.logger.log(f"Model: {self.config['model']['name']}")
        self.logger.log(f"Vocabulary size: {self.config['vocab_size']}")
        self.logger.log(f"Total epochs: {num_epochs}")
        self.logger.log(f"Batch size: {self.config['train']['batch_size']}")
        self.logger.log(f"Learning rate: {self.config['train']['learning_rate']}")
        self.logger.log(f"Optimizer: {self.config['train']['optimizer']}")
        self.logger.log(f"Device: {self.device}")
        self.logger.log(f"AMP enabled: {self.use_amp}")
        self.logger.log(f"Train samples: {len(self.train_loader.dataset):,}")
        self.logger.log(f"Valid samples: {len(self.valid_loader.dataset):,}")
        self.logger.log("=" * 70 + "\n")
        
        try:
            for epoch in range(self.epoch, num_epochs):
                self.epoch = epoch
                start_time = time.time()
                
                # Train
                train_loss = self.train_epoch()
                
                # Validate
                valid_loss, full_seq_acc, per_char_acc = self.validate()
                
                # Update scheduler
                if self.scheduler is not None:
                    if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(valid_loss)
                    else:
                        self.scheduler.step()
                
                # Time
                epoch_time = time.time() - start_time
                
                # Log
                lr = self.optimizer.param_groups[0]['lr']
                print(f"\n{'='*70}")
                print(f"Epoch {epoch+1}/{num_epochs} completed in {epoch_time:.1f}s")
                print(f"Learning Rate: {lr:.6f}")
                print(f"Train Loss: {train_loss:.4f}")
                print(f"Valid Loss: {valid_loss:.4f}")
                print(f"Accuracy - Full Seq: {full_seq_acc*100:.2f}% | Per Char: {per_char_acc*100:.2f}%")
                
                # Log to file
                self.logger.log(f"\nEpoch {epoch+1}/{num_epochs}:")
                self.logger.log(f"  Time: {epoch_time:.1f}s")
                self.logger.log(f"  Learning Rate: {lr:.6f}")
                self.logger.log(f"  Train Loss: {train_loss:.4f}")
                self.logger.log(f"  Valid Loss: {valid_loss:.4f}")
                self.logger.log(f"  Full Sequence Acc: {full_seq_acc*100:.2f}%")
                self.logger.log(f"  Per Character Acc: {per_char_acc*100:.2f}%")
                
                # Save best model - based on Full Sequence Accuracy
                if full_seq_acc > self.best_full_seq_acc:
                    self.best_valid_loss = valid_loss
                    self.best_full_seq_acc = full_seq_acc
                    self.best_per_char_acc = per_char_acc
                    self.save_checkpoint('best_model.pth')
                    self.patience_counter = 0
                    print(f"✓ New best model saved! (Full Seq Acc: {full_seq_acc*100:.2f}%)")
                    self.logger.log(f"  ✓ New best model saved! (Full Seq Acc: {full_seq_acc*100:.2f}%)")
                else:
                    self.patience_counter += 1
                    print(f"  Patience: {self.patience_counter}/{patience} (Best: {self.best_full_seq_acc*100:.2f}%)")
                    self.logger.log(f"  Patience: {self.patience_counter}/{patience} (Best: {self.best_full_seq_acc*100:.2f}%)")
                
                # Save periodic checkpoint
                if (epoch + 1) % self.config['train']['save_interval'] == 0:
                    self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pth')
                
                print("=" * 70)
                
                # Early stopping
                if self.patience_counter >= patience:
                    print(f"\n⏸️  Early stopping at epoch {epoch+1}")
                    self.logger.log(f"\n⏸️  Early stopping at epoch {epoch+1}")
                    break
            
            print("\n" + "=" * 70)
            print("✓ TRAINING COMPLETED")
            print("=" * 70)
            print(f"Best validation loss: {self.best_valid_loss:.4f}")
            print(f"Best Full Sequence Accuracy: {self.best_full_seq_acc*100:.2f}%")
            print(f"Best Per Character Accuracy: {self.best_per_char_acc*100:.2f}%")
            print(f"Best model saved at: {self.save_dir / 'best_model.pth'}")
            
            # Log final results
            self.logger.log("\n" + "=" * 70)
            self.logger.log("✓ TRAINING COMPLETED")
            self.logger.log("=" * 70)
            self.logger.log(f"Best validation loss: {self.best_valid_loss:.4f}")
            self.logger.log(f"Best Full Sequence Accuracy: {self.best_full_seq_acc*100:.2f}%")
            self.logger.log(f"Best Per Character Accuracy: {self.best_per_char_acc*100:.2f}%")
            self.logger.log(f"Best model saved at: {self.save_dir / 'best_model.pth'}")
            self.logger.close()
            
        except KeyboardInterrupt:
            print("\n\n⏸️  Training interrupted by user")
            print("💾 Saving checkpoint...")
            self.save_checkpoint('interrupted_checkpoint.pth')
            print("✓ Checkpoint saved")
            print(f"✓ Best model: {self.save_dir / 'best_model.pth'}")
            
            # Log interruption
            self.logger.log("\n⏸️  Training interrupted by user")
            self.logger.log(f"Last epoch: {self.epoch+1}")
            self.logger.log(f"Best valid loss so far: {self.best_valid_loss:.4f}")
            self.logger.log(f"Best Full Seq Acc: {self.best_full_seq_acc*100:.2f}%")
            self.logger.log(f"Best Per Char Acc: {self.best_per_char_acc*100:.2f}%")
            self.logger.close()
            
        except Exception as e:
            print(f"\n❌ Error during training: {e}")
            import traceback
            traceback.print_exc()
            print("\n💾 Saving emergency checkpoint...")
            self.save_checkpoint('error_checkpoint.pth')
            
            # Log error
            self.logger.log(f"\n❌ Error during training: {e}")
            self.logger.log(traceback.format_exc())
            self.logger.close()
            raise


def train_model(config_file, checkpoint=None):
    """
    Train model
    
    Args:
        config_file: Đường dẫn file config
        checkpoint: Đường dẫn checkpoint để resume training
    """
    print("=" * 70)
    print("MOBILENET-SVTR-CTC TRAINING")
    print("Optimized for Vietnamese Name Recognition")
    print("=" * 70)
    
    # Load config
    print(f"\n📄 Loading config: {config_file}")
    
    # Check GPU
    print("\n🖥️  Checking hardware...")
    has_gpu = check_gpu()
    
    # Load config from file
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    if not has_gpu and config.get('device', 'cuda') == 'cuda':
        print("⚠ Config requires GPU but none found. Switching to CPU...")
        config['device'] = 'cpu'
    
    # Check dataset
    print("\n📊 Checking dataset...")
    if not check_dataset(config):
        print("\n❌ Dataset check failed. Please prepare dataset first.")
        return
    
    # Create directories
    print("\n📁 Creating directories...")
    create_directories(config)
    
    # Print training info
    print("\n⚙️  Training configuration:")
    model_name = config['model']['name']
    print(f"  - Model: {model_name}")
    print(f"  - Backbone hidden: {config['model'].get('backbone_hidden', 256)}")
    print(f"  - SVTR depth: {config['model'].get('svtr_depth', 2)}")
    print(f"  - SVTR heads: {config['model'].get('svtr_heads', 8)}")
    print(f"  - Batch size: {config['train']['batch_size']}")
    print(f"  - Epochs: {config['train']['epochs']}")
    print(f"  - Learning rate: {config['train']['learning_rate']}")
    print(f"  - Image size: {config['train']['image_height']}x{config['train']['image_width']}")
    print(f"  - AMP (Mixed Precision): {'ON' if config['train'].get('use_amp', True) else 'OFF'}")
    print(f"  - Device: {config.get('device', 'cuda')}")
    
    # Initialize trainer
    print("\n🚀 Initializing trainer...")
    trainer = CTCTrainer(config_file)
    
    # Load checkpoint if provided
    if checkpoint:
        if os.path.exists(checkpoint):
            print(f"📥 Loading checkpoint: {checkpoint}")
            trainer.load_checkpoint(checkpoint)
            print(f"   Resuming from epoch {trainer.epoch+1}")
        else:
            print(f"⚠ Checkpoint not found: {checkpoint}")
            print("   Starting from epoch 1")
    
    # Start training
    trainer.train()


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Train MobileNet-SVTR-CTC model for Vietnamese Name Recognition',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train với config mặc định
  python train_mobilenet_svtr_ctc.py --config config_mobilenet_svtr_ctc.yml
  
  # Resume từ checkpoint
  python train_mobilenet_svtr_ctc.py --config config.yml --resume weights/checkpoint_epoch_50.pth
  
  # Train với GPU cụ thể
  CUDA_VISIBLE_DEVICES=0 python train_mobilenet_svtr_ctc.py --config config.yml

Configs:
  - config_mobilenet_svtr_ctc.yml: Standard version (256 hidden, 2 depth)
  - Sửa trong config để dùng Light version (128 hidden, 1 depth)
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Đường dẫn file config YAML'
    )
    
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Đường dẫn checkpoint để resume training'
    )
    
    args = parser.parse_args()
    
    # Check config file exists
    if not os.path.exists(args.config):
        print(f"❌ Config file not found: {args.config}")
        print("\n💡 Available config files:")
        config_dir = Path(__file__).parent
        for f in config_dir.glob('*.yml'):
            print(f"   - {f.name}")
        sys.exit(1)
    
    # Start training
    train_model(
        config_file=args.config,
        checkpoint=args.resume
    )


if __name__ == '__main__':
    main()
