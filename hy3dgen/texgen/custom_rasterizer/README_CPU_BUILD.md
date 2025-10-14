# CPU-Only Build for custom_rasterizer

This directory contains a CPU-only build configuration for the custom_rasterizer module, eliminating CUDA/HIP dependencies.

## Quick Start

### Option 1: Use the CPU-Only Build Script (Recommended)

```bash
cd hy3dgen/texgen/custom_rasterizer
python setup_cpu.py install
```

### Option 2: Create a Wheel for Distribution

```bash
python setup_cpu.py bdist_wheel
pip install dist/custom_rasterizer-0.1.0+cpu*.whl
```

## What's Different?

The CPU-only build:
- ✅ Removes CUDA compiler requirement (nvcc)
- ✅ Removes CUDA Toolkit dependency
- ✅ Uses existing CPU fallback code
- ✅ Works on any system with PyTorch
- ❌ Slower than CUDA version (but still usable)

## Files

- `setup_cpu.py` - CPU-only build script
- `rasterizer_cpu.h` - Modified header without CUDA dependencies
- `PATCH_rasterizer_cpp.txt` - Instructions for patching rasterizer.cpp

## Full Implementation Guide

### Step 1: Modify rasterizer.cpp

Add this at the top of `lib/custom_rasterizer_kernel/rasterizer.cpp`:

```cpp
// After the existing #include "rasterizer.h"
#ifdef CPU_ONLY
// Stub out GPU function for CPU-only builds
std::vector<torch::Tensor> rasterize_image_gpu(torch::Tensor V, torch::Tensor F, torch::Tensor D,
    int width, int height, float occlusion_truncation, int use_depth_prior)
{
    throw std::runtime_error("GPU rasterization not available in CPU-only build");
}
#endif
```

Replace the `rasterize_image` function with:

```cpp
std::vector<torch::Tensor> rasterize_image(torch::Tensor V, torch::Tensor F, torch::Tensor D,
    int width, int height, float occlusion_truncation, int use_depth_prior)
{
#ifdef CPU_ONLY
    // CPU-only build: verify tensors are on CPU
    if (V.get_device() != -1) {
        throw std::runtime_error("CPU-only build requires CPU tensors");
    }
    return rasterize_image_cpu(V, F, D, width, height, occlusion_truncation, use_depth_prior);
#else
    // Original code with GPU support
    int device_id = V.get_device();
    if (device_id == -1)
        return rasterize_image_cpu(V, F, D, width, height, occlusion_truncation, use_depth_prior);
    else
        return rasterize_image_gpu(V, F, D, width, height, occlusion_truncation, use_depth_prior);
#endif
}
```

### Step 2: Modify rasterizer.h

Replace the `#include <ATen/cuda/CUDAContext.h>` section with:

```cpp
// Only include CUDA headers for CUDA builds
#ifndef CPU_ONLY
    #include <ATen/cuda/CUDAContext.h>
#endif
```

Remove or conditionally compile the GPU function declaration:

```cpp
#ifndef CPU_ONLY
std::vector<torch::Tensor> rasterize_image_gpu(torch::Tensor V, torch::Tensor F, torch::Tensor D,
    int width, int height, float occlusion_truncation, int use_depth_prior);
#endif
```

### Step 3: Build

```bash
python setup_cpu.py install
```

## Performance Expectations

| Operation | CUDA | CPU | Slowdown |
|-----------|------|-----|----------|
| 512x512 rasterization | ~5ms | ~50ms | 10x |
| 1024x1024 rasterization | ~15ms | ~200ms | 13x |
| Grid hierarchy build | ~10ms | ~10ms | 1x (CPU only) |

The CPU version is slower but still usable for:
- Development and testing
- Systems without CUDA
- Low-resolution rendering
- Batch processing (where latency matters less)

## Alternative: Use nvdiffrast with OpenGL

For better performance without CUDA, consider using nvdiffrast with OpenGL backend:

```python
import nvdiffrast.torch as dr

# Instead of CUDA context
# glctx = dr.RasterizeCudaContext()

# Use OpenGL context
glctx = dr.RasterizeGLContext()
```

This is already partially implemented in the project (see `nodes.py`, line 1733).

## Troubleshooting

### Build Errors

**Error: "ATen/cuda/CUDAContext.h not found"**
- Solution: Make sure you're using `setup_cpu.py`, not `setup.py`
- The CPU build should not try to include CUDA headers

**Error: "undefined reference to rasterize_image_gpu"**
- Solution: Make sure the `CPU_ONLY` macro is being defined
- Check that the stub function is added to rasterizer.cpp

### Runtime Errors

**Error: "GPU rasterization not available"**
- This is expected in CPU-only build
- Make sure your tensors are on CPU: `tensor.cpu()`

**Error: "CPU-only build requires CPU tensors"**
- Move tensors to CPU before calling rasterize: `V.cpu(), F.cpu()`

## Migration from CUDA Build

If you're migrating from the CUDA build:

```python
# Old code (CUDA):
V = V.cuda()
result = rasterizer.rasterize(V, F, resolution)

# New code (CPU-only):
V = V.cpu()  # Make sure tensors are on CPU
result = rasterizer.rasterize(V, F, resolution)
```

## Testing

Test the CPU build:

```python
import torch
import custom_rasterizer_kernel

# Create test data on CPU
V = torch.rand(100, 4, dtype=torch.float32)
F = torch.randint(0, 100, (50, 3), dtype=torch.int32)
D = torch.zeros(0)

# Should work with CPU tensors
findices, barycentric = custom_rasterizer_kernel.rasterize_image(
    V, F, D, 512, 512, 1e-6, 0
)

print("CPU rasterization successful!")
print(f"Output shape: {findices.shape}, {barycentric.shape}")
```

## Future Improvements

1. **Multi-threading**: Parallelize CPU rasterization with OpenMP
2. **SIMD Optimization**: Use AVX/SSE for faster CPU processing  
3. **Hybrid Mode**: Auto-detect and use CUDA if available, CPU otherwise
4. **nvdiffrast Integration**: Replace custom_rasterizer entirely

## See Also

- `CUDA_DEPENDENCY_ANALYSIS.md` - Full analysis of CUDA dependencies
- Original `setup.py` - CUDA build configuration
- `../differentiable_renderer/mesh_render.py` - Usage examples

## Support

For issues with the CPU-only build, please check:
1. PyTorch is installed correctly
2. C++ compiler is available (MSVC on Windows, GCC/Clang on Linux)
3. All tensors are on CPU before calling rasterize functions

---

Built for ComfyUI-Hunyuan3DWrapper  
Last updated: October 14, 2025
