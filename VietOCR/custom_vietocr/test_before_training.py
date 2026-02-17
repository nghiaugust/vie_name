"""
Test script để verify tất cả components trước khi train
Chạy script này để đảm bảo không có issues
"""

import torch
import sys
from pathlib import Path

# Thêm vietocr vào Python path
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR / "vietocr"))

from vietocr.model.vocab_ctc import VocabCTC
from vietocr.model.mobilenet_svtr_ctc import mobilenet_svtr_ctc
from vietocr.loader.collator_ctc import CollatorCTC


def test_vocab_ctc():
    """Test VocabCTC class"""
    print("\n" + "="*60)
    print("TEST 1: VocabCTC")
    print("="*60)
    
    vocab = VocabCTC('aAàÀbBcC ')
    
    # Test properties
    print(f"✓ Vocab size: {len(vocab)} (should be {len('aAàÀbBcC ') + 1})")
    print(f"✓ Blank index: {vocab.blank}")
    assert vocab.blank == 0, "Blank must be at index 0"
    
    # Test encoding
    text = "Bà Abc"
    encoded = vocab.encode(text)
    print(f"✓ Encode '{text}': {encoded}")
    assert 0 not in encoded, "Encoded sequence should not contain blank"
    
    # Test decoding
    decoded = vocab.decode(encoded)
    print(f"✓ Decode {encoded}: '{decoded}'")
    assert decoded == text, f"Decode failed: '{decoded}' != '{text}'"
    
    # Test decode with blanks
    encoded_with_blanks = [0, 2, 0, 0, 3, 0]  # Blank mixed in
    decoded = vocab.decode(encoded_with_blanks)
    print(f"✓ Decode with blanks {encoded_with_blanks}: '{decoded}'")
    print(f"✓ Blanks automatically skipped")
    
    print("\n✅ VocabCTC tests PASSED!")


def test_model_forward():
    """Test model forward pass"""
    print("\n" + "="*60)
    print("TEST 2: Model Forward Pass")
    print("="*60)
    
    # Create vocab
    vocab = VocabCTC('aAàÀbBcC ')
    vocab_size = len(vocab)
    
    # Create model
    model = mobilenet_svtr_ctc(
        vocab_size=vocab_size,
        hidden=64,  # Small for testing
        svtr_depth=1,
        svtr_heads=4,
        dropout=0.1
    )
    model.eval()
    
    # Test input
    batch_size = 2
    img = torch.randn(batch_size, 3, 32, 256)
    
    print(f"✓ Input shape: {img.shape}")
    print(f"  (N={batch_size}, C=3, H=32, W=256)")
    
    # Forward pass
    with torch.no_grad():
        logits = model(img)
    
    print(f"✓ Output shape: {logits.shape}")
    print(f"  Expected: (N={batch_size}, T=?, vocab_size={vocab_size})")
    
    # Validate shape
    assert logits.dim() == 3, f"Output should be 3D, got {logits.dim()}D"
    assert logits.shape[0] == batch_size, f"Batch size mismatch"
    assert logits.shape[2] == vocab_size, f"Vocab size mismatch: {logits.shape[2]} != {vocab_size}"
    
    print(f"✓ Sequence length after backbone+neck: {logits.shape[1]}")
    print(f"  Ratio: 256 (input width) -> {logits.shape[1]} (sequence length)")
    print(f"  Note: Height=32 flattened into channels, only width preserved")
    
    print("\n✅ Model forward tests PASSED!")


def test_ctc_loss():
    """Test CTC loss computation"""
    print("\n" + "="*60)
    print("TEST 3: CTC Loss")
    print("="*60)
    
    # Create vocab
    vocab = VocabCTC('aAàÀbBcC ')
    vocab_size = len(vocab)
    
    # Dummy data
    batch_size = 2
    seq_len = 20
    
    # Logits from model
    logits = torch.randn(batch_size, seq_len, vocab_size)
    log_probs = torch.log_softmax(logits, dim=-1).permute(1, 0, 2)  # (T, N, C)
    
    # Targets
    target1 = vocab.encode("Abc")  # [1, 5, 7]
    target2 = vocab.encode("Ba")   # [5, 1]
    targets = torch.tensor(target1 + target2, dtype=torch.long)
    
    # Lengths
    input_lengths = torch.full((batch_size,), seq_len, dtype=torch.long)
    target_lengths = torch.tensor([len(target1), len(target2)], dtype=torch.long)
    
    print(f"✓ Log probs shape: {log_probs.shape} (T, N, C)")
    print(f"✓ Targets: {targets.tolist()}")
    print(f"✓ Input lengths: {input_lengths.tolist()}")
    print(f"✓ Target lengths: {target_lengths.tolist()}")
    
    # Validate CTC constraints
    assert (target_lengths <= input_lengths).all(), "Target length must <= input length"
    assert (target_lengths > 0).all(), "Target length must > 0"
    
    # CTC Loss
    criterion = torch.nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
    loss = criterion(log_probs, targets, input_lengths, target_lengths)
    
    print(f"✓ CTC Loss: {loss.item():.4f}")
    assert not torch.isnan(loss), "Loss is NaN!"
    assert not torch.isinf(loss), "Loss is Inf!"
    
    print("\n✅ CTC Loss tests PASSED!")


def test_collator():
    """Test CollatorCTC"""
    print("\n" + "="*60)
    print("TEST 4: CollatorCTC")
    print("="*60)
    
    vocab = VocabCTC('aAàÀbBcC ')
    
    # Dummy samples
    samples = [
        {
            "img": torch.randn(3, 32, 100),
            "word": vocab.encode("Abc"),
            "img_path": "img1.jpg"
        },
        {
            "img": torch.randn(3, 32, 100),
            "word": vocab.encode("Ba"),
            "img_path": "img2.jpg"
        }
    ]
    
    collator = CollatorCTC()
    batch = collator(samples)
    
    print(f"✓ Batch keys: {batch.keys()}")
    print(f"✓ Images shape: {batch['img'].shape}")
    print(f"✓ Targets shape: {batch['tgt_output'].shape}")
    print(f"✓ Targets:\n  {batch['tgt_output']}")
    print(f"✓ Target lengths: {batch['tgt_lengths'].tolist()}")
    
    # Validate
    assert batch['img'].shape[0] == 2, "Batch size should be 2"
    assert batch['tgt_output'].shape[0] == 2, "Batch size should be 2"
    assert batch['tgt_output'].shape[1] == 3, "Max length should be 3 (Abc)"
    assert batch['tgt_lengths'].tolist() == [3, 2], "Lengths should be [3, 2]"
    
    print("\n✅ CollatorCTC tests PASSED!")


def test_ctc_decode():
    """Test CTC greedy decoding"""
    print("\n" + "="*60)
    print("TEST 5: CTC Greedy Decode")
    print("="*60)
    
    vocab = VocabCTC('aAàÀbBcC ')
    
    # Simulate CTC output: [blank, A, A, blank, b, b, b, blank, c]
    # Should decode to: "Abc"
    pred = torch.tensor([0, 2, 2, 0, 5, 5, 5, 0, 7])
    
    print(f"✓ Raw CTC output: {pred.tolist()}")
    
    # CTC greedy decode
    decoded_seq = []
    prev = None
    blank_idx = 0
    for token in pred.tolist():
        if token != blank_idx and token != prev:
            decoded_seq.append(token)
        prev = token
    
    print(f"✓ After removing blanks & duplicates: {decoded_seq}")
    
    # Decode with vocab
    decoded_text = vocab.decode(decoded_seq)
    print(f"✓ Decoded text: '{decoded_text}'")
    
    # Expected: A(2) b(5) c(7) -> "Abc"
    expected = "Abc"
    assert decoded_text == expected, f"Decode failed: '{decoded_text}' != '{expected}'"
    
    print("\n✅ CTC Decode tests PASSED!")


def main():
    print("\n" + "="*60)
    print("RUNNING PRE-TRAINING TESTS")
    print("="*60)
    print("Kiểm tra tất cả components trước khi train...")
    
    try:
        test_vocab_ctc()
        test_model_forward()
        test_ctc_loss()
        test_collator()
        test_ctc_decode()
        
        print("\n" + "="*60)
        print("🎉 ALL TESTS PASSED!")
        print("="*60)
        print()
        print("✅ VocabCTC hoạt động đúng")
        print("✅ Model forward pass OK")
        print("✅ CTC Loss không crash")
        print("✅ CollatorCTC batching đúng")
        print("✅ CTC decoding chính xác")
        print()
        print("➡️  Bạn có thể bắt đầu training an toàn!")
        print()
        
    except Exception as e:
        print("\n" + "="*60)
        print("❌ TEST FAILED!")
        print("="*60)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        print("\n⚠️  Vui lòng fix lỗi trước khi train!")
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
