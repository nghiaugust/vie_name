"""
Script đơn giản để kiểm tra GPU
"""

import torch

print("=" * 60)
print("KIỂM TRA GPU")
print("=" * 60)

# Check PyTorch version
print(f"\n📦 PyTorch version: {torch.__version__}")

# Check CUDA availability
print(f"\n🔍 CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    # Get CUDA version
    print(f"✓ CUDA version: {torch.version.cuda}")
    
    # Get number of GPUs
    num_gpus = torch.cuda.device_count()
    print(f"✓ Number of GPUs: {num_gpus}")
    
    # Get info for each GPU
    for i in range(num_gpus):
        print(f"\n🎮 GPU {i}:")
        print(f"  Name: {torch.cuda.get_device_name(i)}")
        
        # Memory info
        total_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(f"  Total memory: {total_memory:.2f} GB")
        
        # Current memory usage
        allocated = torch.cuda.memory_allocated(i) / 1e9
        cached = torch.cuda.memory_reserved(i) / 1e9
        print(f"  Allocated: {allocated:.2f} GB")
        print(f"  Cached: {cached:.2f} GB")
        print(f"  Free: {total_memory - allocated:.2f} GB")
        
        # Compute capability
        capability = torch.cuda.get_device_capability(i)
        print(f"  Compute capability: {capability[0]}.{capability[1]}")
    
    # Test GPU with a simple operation
    print("\n⚡ Testing GPU computation...")
    try:
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        z = torch.matmul(x, y)
        print("✓ GPU computation test: PASSED")
        
        # Check which device is being used
        print(f"✓ Current device: cuda:{torch.cuda.current_device()}")
        
    except Exception as e:
        print(f"✗ GPU computation test: FAILED")
        print(f"  Error: {e}")
        
else:
    print("\n❌ No GPU detected!")
    print("⚠ Training will use CPU (very slow)")
    print("\n💡 Possible reasons:")
    print("  - NVIDIA GPU drivers not installed")
    print("  - CUDA toolkit not installed")
    print("  - PyTorch CPU version installed (not GPU version)")
    print("\n📝 To install PyTorch with CUDA:")
    print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")

print("\n" + "=" * 60)
