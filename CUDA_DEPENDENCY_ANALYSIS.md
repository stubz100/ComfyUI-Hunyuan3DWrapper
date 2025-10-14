# CUDA Dependency Analysis & Refactoring Options

## Executive Summary

The `custom_rasterizer` module in `/hy3dgen/texgen/custom_rasterizer` is a **CUDA/HIP-dependent** C++ extension that performs triangle rasterization and grid neighbor operations for texture generation. While it's possible to refactor this code, there are several alternatives that may be more practical.

## Current Architecture

### Files & Dependencies

```
hy3dgen/texgen/custom_rasterizer/
├── setup.py                           # Build script using CUDAExtension
├── lib/custom_rasterizer_kernel/
│   ├── rasterizer.cpp                 # Main rasterization logic (has CPU fallback!)
│   ├── rasterizer.h                   # Header with CUDA macros (__host__ __device__)
│   ├── rasterizer_gpu.cu              # CUDA GPU kernels
│   └── grid_neighbor.cpp              # Grid hierarchy building (CPU only)
└── custom_rasterizer/
    ├── render.py                      # Python interface
    └── [other utility files]
```

### Key Functions

1. **`rasterize_image()`** - Main rasterization function
   - Projects 3D vertices to 2D screen space
   - Performs triangle rasterization with z-buffering
   - Computes barycentric coordinates for texture mapping
   - **Already has CPU fallback implementation!**

2. **`build_hierarchy()`** - Grid-based spatial hierarchy
   - Builds multi-resolution grid structure
   - Computes neighbor relationships
   - Used for texture synthesis
   - **CPU only (no CUDA)**

## Good News: CPU Fallback Already Exists!

Looking at `rasterizer.cpp`, line 144-151:

```cpp
std::vector<torch::Tensor> rasterize_image(torch::Tensor V, torch::Tensor F, torch::Tensor D,
    int width, int height, float occlusion_truncation, int use_depth_prior)
{
    int device_id = V.get_device();
    if (device_id == -1)
        return rasterize_image_cpu(V, F, D, width, height, occlusion_truncation, use_depth_prior);
    else
        return rasterize_image_gpu(V, F, D, width, height, occlusion_truncation, use_depth_prior);
}
```

**The module automatically falls back to CPU when CUDA is not available!**

## Alternative: nvdiffrast (Already Implemented!)

The project **already has** an alternative renderer using `nvdiffrast` in `nodes.py` (line 1733+):

```python
class Hy3DNvdiffrastRenderer:
    # Uses nvdiffrast.torch for rendering
    # Supports OpenGL backend!
```

### nvdiffrast Advantages:
- ✅ **OpenGL backend available** (no CUDA required)
- ✅ Already integrated in the project
- ✅ More flexible and feature-rich
- ✅ Better maintained by NVIDIA
- ✅ Industry standard for differentiable rendering

### nvdiffrast OpenGL Support:
```python
# Instead of:
glctx = dr.RasterizeCudaContext()

# Use:
glctx = dr.RasterizeGLContext()  # Uses OpenGL!
```

## Refactoring Options

### Option 1: Use CPU-Only Build (Easiest) ⭐ RECOMMENDED

**Effort**: Low  
**Compatibility**: High  

Modify `setup.py` to build without CUDA:

```python
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension
import torch

# Remove CUDA-specific code
custom_rasterizer_module = CppExtension('custom_rasterizer_kernel', [
    'lib/custom_rasterizer_kernel/rasterizer.cpp',
    'lib/custom_rasterizer_kernel/grid_neighbor.cpp',
    # Remove: 'lib/custom_rasterizer_kernel/rasterizer_gpu.cu',
])

setup(
    packages=find_packages(),
    version="0.1.0",
    name='custom_rasterizer',
    include_package_data=True,
    package_dir={'': '.'},
    ext_modules=[custom_rasterizer_module],
    cmdclass={'build_ext': BuildExtension},   
)
```

**Required Changes**:
1. Remove `.cu` file from build
2. Remove CUDA includes from `rasterizer.h`
3. Remove `rasterize_image_gpu()` function declaration
4. Stub out GPU function or make it call CPU version

### Option 2: Extend nvdiffrast Integration (Moderate)

**Effort**: Moderate  
**Compatibility**: High  

Replace `custom_rasterizer` usage in `differentiable_renderer/mesh_render.py` with nvdiffrast:

**Pros**:
- Professional solution
- Better maintained
- OpenGL support
- More features

**Cons**:
- Need to refactor texture generation pipeline
- May need to port grid hierarchy code
- Testing required

### Option 3: Pure Python Implementation (High Effort)

**Effort**: High  
**Compatibility**: Highest  

Rewrite rasterization in pure PyTorch:

**Pros**:
- No compilation needed
- Works everywhere
- Easy to debug

**Cons**:
- Slower performance
- Significant development time
- Need to port all algorithms

### Option 4: OpenGL via PyOpenGL (High Effort)

**Effort**: High  
**Compatibility**: Moderate  

Replace CUDA with OpenGL compute shaders:

**Pros**:
- Widely available
- Good performance

**Cons**:
- Still needs GPU
- Complex implementation
- Platform-specific issues

## Detailed Implementation: Option 1 (CPU-Only)

### Step 1: Modify setup.py

Create a new `setup_cpu.py`:

```python
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension
import torch

torch_version = torch.__version__.split('+')[0].replace('.', '')
version = f"0.1.0+cpu"

custom_rasterizer_module = CppExtension(
    'custom_rasterizer_kernel',
    [
        'lib/custom_rasterizer_kernel/rasterizer.cpp',
        'lib/custom_rasterizer_kernel/grid_neighbor.cpp',
    ],
    extra_compile_args={
        'cxx': ['-O3', '-DCPU_ONLY']
    }
)

setup(
    packages=find_packages(),
    version=version,
    name='custom_rasterizer',
    include_package_data=True,
    package_dir={'': '.'},
    ext_modules=[custom_rasterizer_module],
    cmdclass={'build_ext': BuildExtension},   
)
```

### Step 2: Modify rasterizer.h

```cpp
#ifndef RASTERIZER_H_
#define RASTERIZER_H_

#include <torch/extension.h>
#include <vector>
#include <ATen/ATen.h>
#include <cstdint>

#define INT64 uint64_t
#define MAXINT 2147483647

// Change device macros to host-only for CPU build
#ifdef CPU_ONLY
    #define __host__
    #define __device__
#endif

// Remove CUDA context include for CPU-only builds
#ifndef CPU_ONLY
    #include <ATen/cuda/CUDAContext.h>
#endif

// ... rest of the header ...

// Function declarations
std::vector<torch::Tensor> rasterize_image_cpu(torch::Tensor V, torch::Tensor F, torch::Tensor D,
    int width, int height, float occlusion_truncation, int use_depth_prior);

#ifndef CPU_ONLY
std::vector<torch::Tensor> rasterize_image_gpu(torch::Tensor V, torch::Tensor F, torch::Tensor D,
    int width, int height, float occlusion_truncation, int use_depth_prior);
#endif

#endif
```

### Step 3: Modify rasterizer.cpp

```cpp
std::vector<torch::Tensor> rasterize_image(torch::Tensor V, torch::Tensor F, torch::Tensor D,
    int width, int height, float occlusion_truncation, int use_depth_prior)
{
#ifdef CPU_ONLY
    // CPU-only build
    return rasterize_image_cpu(V, F, D, width, height, occlusion_truncation, use_depth_prior);
#else
    // Original logic with GPU support
    int device_id = V.get_device();
    if (device_id == -1)
        return rasterize_image_cpu(V, F, D, width, height, occlusion_truncation, use_depth_prior);
    else
        return rasterize_image_gpu(V, F, D, width, height, occlusion_truncation, use_depth_prior);
#endif
}
```

### Step 4: Build Instructions

```bash
cd hy3dgen/texgen/custom_rasterizer
python setup_cpu.py install
```

Or create a CPU wheel:

```bash
python setup_cpu.py bdist_wheel
pip install dist/custom_rasterizer-0.1.0+cpu-*.whl
```

## Performance Comparison

| Implementation | Speed | Compatibility | Effort |
|---------------|-------|---------------|--------|
| CUDA (current) | ⚡⚡⚡⚡⚡ | ❌ CUDA only | ✅ Done |
| CPU fallback | ⚡⚡ | ✅ Universal | ⚡ Very Low |
| nvdiffrast+OpenGL | ⚡⚡⚡⚡ | ✅ OpenGL GPU | ⚡⚡ Low |
| Pure PyTorch | ⚡ | ✅ Universal | ⚡⚡⚡⚡ High |

## Recommendations

### For Immediate Use:
**Use Option 1 (CPU-Only Build)** 
- Minimal code changes
- CPU fallback already implemented
- Universal compatibility
- Just remove CUDA compilation

### For Long-term Solution:
**Expand nvdiffrast Integration (Option 2)**
- Better maintained
- Professional solution
- OpenGL backend available
- More features for future development

### Quick Fix for Users:
Add automatic fallback in `mesh_render.py`:

```python
def __init__(self, ...):
    try:
        import custom_rasterizer as cr
        self.raster = cr
        self.raster_mode = 'cr'
    except ImportError:
        log.warning("custom_rasterizer not available, falling back to nvdiffrast")
        try:
            import nvdiffrast.torch as dr
            self.raster = dr
            self.raster_mode = 'nvdiffrast'
        except ImportError:
            raise ImportError("No rasterizer available. Install custom_rasterizer or nvdiffrast")
```

## Technical Details: What the Code Does

### Rasterization Process

1. **Vertex Transformation** (Lines in `rasterizer_gpu.cu` 102-106):
   - Converts 3D vertices to 2D screen coordinates
   - Applies perspective projection
   - Computes depth values for z-buffering

2. **Triangle Rasterization** (Lines 3-38):
   - Iterates over pixels covered by each triangle
   - Uses barycentric coordinates for interpolation
   - Maintains z-buffer for occlusion

3. **Barycentric Interpolation** (Lines 40-80):
   - Computes barycentric weights for each pixel
   - Enables smooth interpolation of vertex attributes
   - Used for texture mapping and normal interpolation

### Grid Hierarchy (grid_neighbor.cpp)

- Builds spatial data structure for efficient neighbor queries
- Used in texture synthesis algorithms
- **No CUDA dependency** - pure CPU code

## Conclusion

The **best immediate solution** is to build the CPU-only version (Option 1). The code already supports it; you just need to modify the build process.

For a **long-term robust solution**, integrate nvdiffrast more deeply and use its OpenGL backend for non-CUDA systems.

## Next Steps

1. ✅ Create `setup_cpu.py` with CPU-only build
2. ✅ Test CPU fallback performance
3. ✅ Build CPU wheel for distribution
4. 🔄 Document nvdiffrast OpenGL usage
5. 🔄 Consider full nvdiffrast migration

---

**Author**: GitHub Copilot  
**Date**: October 14, 2025  
**Status**: Analysis Complete - Ready for Implementation
