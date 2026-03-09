"""
Quick GPU Check Script
"""
import torch

print("="*70)
print("GPU CHECK")
print("="*70)

print(f"\nPyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"GPU Count: {torch.cuda.device_count()}")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print("\n✅ GPU IS READY!")
else:
    print("\n❌ GPU NOT AVAILABLE")
    print("\nPossible reasons:")
    print("1. PyTorch CPU-only version is installed")
    print("2. Running in wrong virtual environment")
    print("3. CUDA drivers not properly configured")

    print("\n📋 SOLUTION:")
    print("Run these commands:")
    print("  pip uninstall torch torchvision")
    print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")

print("="*70)
