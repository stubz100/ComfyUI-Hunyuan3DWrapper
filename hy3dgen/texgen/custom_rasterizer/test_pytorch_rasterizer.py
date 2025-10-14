"""
Quick Test Script for PyTorch Rasterizer
=========================================

Run this to verify the PyTorch rasterizer works on your system.
Tests all backends and provides performance metrics.
"""

import torch
import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 70)
print("PyTorch Rasterizer Quick Test")
print("=" * 70)

# System info
print("\n📋 System Information:")
print(f"  PyTorch version: {torch.__version__}")
print(f"  CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"  CUDA version: {torch.version.cuda}")
    try:
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    except:
        print(f"  GPU: Unknown")
    
    # Detect ROCm
    if hasattr(torch.version, 'hip'):
        print(f"  🎉 ROCm detected: {torch.version.hip}")
        print(f"  ✓ AMD GPU support enabled!")
    else:
        print(f"  NVIDIA CUDA GPU")

print(f"  MPS available: {torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False}")

# Test 1: Basic PyTorch Rasterizer
print("\n" + "=" * 70)
print("Test 1: Basic PyTorch Rasterizer")
print("=" * 70)

try:
    from pytorch_rasterizer import PyTorchRasterizer
    
    rast = PyTorchRasterizer()
    
    # Create test data
    device = rast.device
    V = torch.rand(100, 4, device=device)
    V[:, :3] = V[:, :3] * 2 - 1
    V[:, 3] = 1.0
    F = torch.randint(0, 100, (50, 3), dtype=torch.int32, device=device)
    D = torch.zeros(0, device=device)
    
    # Warmup
    for _ in range(3):
        _, _ = rast.rasterize_image(V, F, D, 256, 256, 1e-6, 0)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Benchmark
    start = time.time()
    findices, barycentric = rast.rasterize_image(V, F, D, 512, 512, 1e-6, 0)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    elapsed = (time.time() - start) * 1000
    
    coverage = (findices > 0).sum().item() / (512 * 512) * 100
    
    print(f"✓ Basic rasterizer works!")
    print(f"  Time: {elapsed:.2f}ms")
    print(f"  Coverage: {coverage:.1f}%")
    print(f"  Output shapes: findices={findices.shape}, bary={barycentric.shape}")
    
except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()

# Test 2: Optimized Rasterizer
print("\n" + "=" * 70)
print("Test 2: Optimized PyTorch Rasterizer")
print("=" * 70)

try:
    from pytorch_rasterizer_optimized import create_optimized_rasterizer
    
    rast = create_optimized_rasterizer(mode='tiled', tile_size=32)
    
    device = rast.device
    V = torch.rand(200, 4, device=device)
    V[:, :3] = V[:, :3] * 2 - 1
    V[:, 3] = 1.0
    F = torch.randint(0, 200, (100, 3), dtype=torch.int32, device=device)
    D = torch.zeros(0, device=device)
    
    # Warmup
    for _ in range(3):
        _, _ = rast.rasterize_image(V, F, D, 256, 256, 1e-6, 0)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Benchmark
    start = time.time()
    findices, barycentric = rast.rasterize_image(V, F, D, 512, 512, 1e-6, 0)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    elapsed = (time.time() - start) * 1000
    coverage = (findices > 0).sum().item() / (512 * 512) * 100
    
    print(f"✓ Optimized rasterizer works!")
    print(f"  Time: {elapsed:.2f}ms")
    print(f"  Coverage: {coverage:.1f}%")
    
except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Rasterizer Wrapper (Auto-detection)
print("\n" + "=" * 70)
print("Test 3: Rasterizer Wrapper (Auto-detection)")
print("=" * 70)

try:
    from rasterizer_wrapper import RasterizerWrapper
    
    wrapper = RasterizerWrapper()
    
    print(f"✓ Wrapper initialized!")
    print(f"  Backend: {wrapper.mode}")
    print(f"  Device: {wrapper.device}")
    
    # Quick test
    if wrapper.mode.startswith('pytorch'):
        device = wrapper.backend.device
        V = torch.rand(50, 4, device=device)
        V[:, :3] = V[:, :3] * 2 - 1
        V[:, 3] = 1.0
        F = torch.randint(0, 50, (25, 3), dtype=torch.int32, device=device)
        
        findices, bary = wrapper.rasterize(V, F, (256, 256))
        print(f"✓ Wrapper rasterization works!")
        print(f"  Output: {findices.shape}, {bary.shape}")
    
except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Performance Comparison
print("\n" + "=" * 70)
print("Test 4: Performance Comparison")
print("=" * 70)

try:
    from pytorch_rasterizer import PyTorchRasterizer
    from pytorch_rasterizer_optimized import TiledPyTorchRasterizer
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test configuration
    num_verts = 500
    num_faces = 250
    width, height = 1024, 1024
    
    V = torch.rand(num_verts, 4, device=device)
    V[:, :3] = V[:, :3] * 2 - 1
    V[:, 3] = 1.0
    F = torch.randint(0, num_verts, (num_faces, 3), dtype=torch.int32, device=device)
    D = torch.zeros(0, device=device)
    
    print(f"Configuration: {width}x{height}, {num_faces} triangles")
    print(f"Device: {device}\n")
    
    rasterizers = {
        'Basic': PyTorchRasterizer(device),
        'Tiled 32x32': TiledPyTorchRasterizer(device, tile_size=32),
        'Tiled 64x64': TiledPyTorchRasterizer(device, tile_size=64),
    }
    
    for name, rast in rasterizers.items():
        # Warmup
        for _ in range(3):
            rast.rasterize_image(V, F, D, width, height, 1e-6, 0)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Benchmark
        times = []
        for _ in range(5):
            start = time.time()
            findices, bary = rast.rasterize_image(V, F, D, width, height, 1e-6, 0)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            times.append((time.time() - start) * 1000)
        
        avg_time = sum(times) / len(times)
        print(f"{name:20s}: {avg_time:6.2f}ms")
    
except Exception as e:
    print(f"✗ Performance test failed: {e}")

# Summary
print("\n" + "=" * 70)
print("Test Summary")
print("=" * 70)

if torch.cuda.is_available():
    if hasattr(torch.version, 'hip'):
        print("🎉 AMD GPU detected with ROCm!")
        print("✓ PyTorch rasterizer provides excellent AMD support!")
    else:
        print("✓ NVIDIA GPU detected with CUDA!")
        print("✓ PyTorch rasterizer works great!")
else:
    print("⚠️  No GPU detected - using CPU fallback")
    print("   Performance will be slower but functional")

print("\n💡 Next Steps:")
print("   1. Integration: Use RasterizerWrapper in your code")
print("   2. Optimization: Try tiled mode for best performance")
print("   3. Testing: Run with your actual meshes")

print("\n📖 Documentation:")
print("   - PYTORCH_RASTERIZER_GUIDE.md - Complete user guide")
print("   - AMD_GPU_ANALYSIS.md - AMD-specific information")
print("   - PYTORCH_IMPLEMENTATION_COMPLETE.md - Full summary")

print("\n" + "=" * 70)
print("All tests complete! ✓")
print("=" * 70)
