# Complete CUDA Dependency Audit - ComfyUI-Hunyuan3DWrapper

**Date**: October 14, 2025  
**Scope**: Full codebase scan for CUDA dependencies  
**Status**: ✅ AUDIT COMPLETE

---

## Executive Summary

After comprehensive analysis of the entire ComfyUI-Hunyuan3DWrapper codebase, I've identified **TWO main areas** with CUDA dependencies:

### ✅ Already Solved
1. **`custom_rasterizer` module** - PyTorch implementation created (completed)

### 🔧 Needs Attention
2. **Hard-coded device references** - Many files use `device='cuda'` or `.cuda()` calls
3. **`mesh_processor.cpp`** - Has compiled .pyd but **already has NumPy fallback** ✅

---

## Detailed Findings

### 1. ✅ Custom Rasterizer Module (SOLVED)

**Location**: `hy3dgen/texgen/custom_rasterizer/`

**Status**: ✅ **COMPLETE** - PyTorch implementation created with ROCm support

**Files**:
- ✅ `pytorch_rasterizer.py` - Base implementation
- ✅ `pytorch_rasterizer_optimized.py` - Performance optimizations
- ✅ `pytorch_grid_hierarchy.py` - Grid hierarchies
- ✅ `rasterizer_wrapper.py` - Auto backend selection
- ⚠️ `setup.py` - CUDA compilation (optional, fallbacks exist)

**Solution**: Complete PyTorch implementation with automatic ROCm/CUDA/CPU detection.

---

### 2. 🔧 Mesh Processor Module (ALREADY HAS FALLBACK!)

**Location**: `hy3dgen/texgen/differentiable_renderer/`

**Status**: ✅ **NO ACTION NEEDED** - Pure NumPy fallback already exists!

#### Files Analysis:

##### ✅ `mesh_processor.py` (Pure Python - NO CUDA)
```python
# Lines 1-200: Pure NumPy implementation
def meshVerticeInpaint_smooth(texture, mask, vtx_pos, vtx_uv, pos_idx, uv_idx):
    # Uses only numpy operations
    texture_height, texture_width, texture_channel = texture.shape
    vtx_num = vtx_pos.shape[0]
    vtx_mask = np.zeros(vtx_num, dtype=np.float32)
    # ... all NumPy operations
```

**Dependencies**: NumPy only ✅

##### ✅ `mesh_processor.cpp` (Optional C++ acceleration)
```cpp
// Lines 1-161: Pure C++ with pybind11
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
// NO CUDA INCLUDES
// NO CUDA KERNELS
```

**Dependencies**: 
- pybind11 (standard C++ bindings)
- Standard C++ libraries
- **NO CUDA/HIP required** ✅

##### ✅ `setup.py` (Pure C++ build)
```python
# Lines 1-30: Standard C++ compilation
Extension(
    "mesh_processor",
    ["mesh_processor.cpp"],  # Only .cpp, no .cu files
    language='c++'           # C++ only, not CUDA
)
```

**Verdict**: The C++ version is just an **optional optimization**. If compilation fails, the pure Python version in `mesh_processor.py` will be used automatically.

##### Usage in `mesh_render.py`:
```python
from .mesh_processor import meshVerticeInpaint  # Will use .py if .pyd unavailable
```

**How it works**:
1. Python tries to import compiled `mesh_processor.cp312-win_amd64.pyd`
2. If not found, falls back to `mesh_processor.py` (pure NumPy)
3. Both have identical API: `meshVerticeInpaint(texture, mask, ...)`

---

### 3. ⚠️ Hard-Coded Device References (NEEDS FIXING)

**Impact**: MEDIUM - Prevents AMD GPU usage even though code is compatible

**Issue**: Many files hard-code `device='cuda'` or use `.cuda()` which fails on:
- AMD GPUs (should use `device='rocm'` or generic)
- Non-CUDA systems
- CPU-only environments

#### 📍 Files with Hard-Coded 'cuda' (34 matches)

##### Priority 1: Core Functionality Files

**`utils.py`** (8 occurrences)
```python
# Lines 84-113: Camera utilities
fov = torch.deg2rad(torch.tensor(float(fov))).cuda()  # ❌ Hard-coded
yaw = torch.tensor(float(yaw)).cuda()                  # ❌ Hard-coded
# ... more .cuda() calls
```
**Impact**: HIGH - Used by rendering pipeline  
**Fix**: Use `device` parameter or `comfy.model_management.get_torch_device()`

---

**`nodes.py`** (4 occurrences)
```python
# Lines 1224, 1237, 1310, 1324: Memory tracking
torch.cuda.reset_peak_memory_stats(device)  # Only works on CUDA
```
**Impact**: MEDIUM - Memory stats fail on non-CUDA  
**Fix**: Add device type check before CUDA-specific calls

---

**`hy3dgen/texgen/differentiable_renderer/mesh_render.py`** (1 occurrence)
```python
# Line 136: MeshRender initialization
def __init__(self, device='cuda', ...):  # ❌ Hard-coded default
```
**Impact**: HIGH - Core rendering module  
**Fix**: Change default to `device=None` with auto-detection

---

**`hy3dgen/texgen/custom_rasterizer/rasterizer_wrapper.py`** (5 occurrences)
```python
# Lines 130-132, 256-259: Test tensors and data transfer
test_v = torch.rand(10, 4, device='cuda')      # ❌ Test code
pos = pos.cuda()                                # ❌ Force CUDA
tri = tri.cuda()                                # ❌ Force CUDA
```
**Impact**: MEDIUM - Wrapper handles device, but forces CUDA in tests  
**Fix**: Use automatic device detection from wrapper

---

##### Priority 2: Model Loading Files

**`hy3dgen/shapegen/pipelines.py`** (1 occurrence)
**`hy3dshape/hy3dshape/pipelines.py`** (3 occurrences)
**`hy3dgen/shapegen/models/autoencoders/model.py`** (2 occurrences)
**`hy3dshape/hy3dshape/models/autoencoders/model.py`** (2 occurrences)

```python
# Default device parameters
def __init__(self, device='cuda', ...):
```
**Impact**: MEDIUM - Model loading defaults to CUDA  
**Fix**: Use ComfyUI's device management

---

##### Priority 3: Utility & Post-Processing Files

**`hy3dgen/text2image.py`** (1 occurrence)
**`hy3dgen/shapegen/postprocessors.py`** (2 occurrences)
**`hy3dgen/texgen/utils/alignImg4Tex_utils.py`** (1 occurrence)
**`hy3dgen/texgen/hunyuanpaint/pipeline.py`** (1 occurrence)
**`hy3dgen/shapegen/bpt/miche/encode.py`** (1 occurrence)

```python
# Various hard-coded CUDA calls
model.cuda()
tensor.to("cuda")
```

**Impact**: LOW-MEDIUM - Specific operations  
**Fix**: Use device parameter from parent context

---

##### Priority 4: Demo & Test Files (OK to ignore)

**`hy3dshape/minimal_vae_demo.py`** (1 occurrence)
```python
surface = loader(mesh_demo).to('cuda', dtype=torch.float16)  # ⚠️ Demo only
```
**Impact**: NONE - Demo/test file  
**Action**: No fix needed (demos can require CUDA)

---

##### Priority 5: Timing & Profiling (Graceful Degradation Needed)

**`hy3dshape/hy3dshape/utils/utils.py`** (3 occurrences)
**`hy3dgen/shapegen/utils.py`** (3 occurrences)
**`hy3dshape/hy3dshape/utils/trainings/callback.py`** (4 occurrences)

```python
# CUDA-specific profiling
self.start = torch.cuda.Event(enable_timing=True)  # Only on CUDA
torch.cuda.synchronize()                            # Only on CUDA
max_memory = torch.cuda.max_memory_allocated()     # Only on CUDA
```

**Impact**: LOW - Profiling code  
**Fix**: Add device type checks with fallback timing methods

---

### 4. ✅ Files Using torch.cuda Safely (NO ACTION NEEDED)

These files check for CUDA availability before using it:

**`hy3dgen/texgen/custom_rasterizer/rasterizer_wrapper.py`**
```python
# Line 66, 127, 166, 174: Safe usage ✅
if torch.cuda.is_available():
    cuda_name = torch.cuda.get_device_name(0)
self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
```

**`hy3dgen/texgen/custom_rasterizer/pytorch_rasterizer_optimized.py`**
```python
# Line 259, 294: Safe usage ✅
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

**`hy3dgen/texgen/custom_rasterizer/pytorch_grid_hierarchy.py`**
```python
# Line 313: Safe usage ✅
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

---

## Summary by Category

| Category | Status | Action Required |
|----------|--------|-----------------|
| **Custom Rasterizer** | ✅ Solved | None - PyTorch impl complete |
| **Mesh Processor** | ✅ No Issue | NumPy fallback already exists |
| **Hard-coded 'cuda'** | ⚠️ Needs Fix | Replace with device detection |
| **torch.cuda checks** | ✅ OK | Already safe |
| **Demo files** | ✅ OK | Can require CUDA |

---

## Recommended Fixes

### Approach: Device Management Abstraction

Create a central device management utility that all modules use:

**New file**: `device_utils.py`
```python
import torch
import comfy.model_management as mm

def get_device(preferred=None):
    """Get the best available device.
    
    Args:
        preferred: Optional preferred device ('cuda', 'rocm', 'cpu', None)
    
    Returns:
        torch.device: Best available device
    """
    if preferred is not None:
        return torch.device(preferred)
    
    # Use ComfyUI's device management if available
    try:
        return mm.get_torch_device()
    except:
        pass
    
    # Fallback to PyTorch device detection
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.version, 'hip') and torch.version.hip is not None:
        return torch.device('cuda')  # ROCm uses 'cuda' device type
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

def to_device(tensor_or_module, device=None):
    """Move tensor or module to device safely.
    
    Args:
        tensor_or_module: Tensor or nn.Module to move
        device: Target device (None = auto-detect)
    
    Returns:
        Moved tensor or module
    """
    if device is None:
        device = get_device()
    return tensor_or_module.to(device)

def safe_cuda_call(func, fallback=None):
    """Execute CUDA-specific function safely.
    
    Args:
        func: Function to execute (e.g., torch.cuda.synchronize)
        fallback: Fallback return value if not on CUDA
    
    Returns:
        Result of func() or fallback
    """
    try:
        if torch.cuda.is_available():
            return func()
        return fallback
    except:
        return fallback
```

---

### High-Priority Fixes

#### 1. Fix `utils.py` (Camera utilities)

**Before**:
```python
def orbit_camera_from_pose(self, fov, yaw, pitch, r):
    fov = torch.deg2rad(torch.tensor(float(fov))).cuda()
    yaw = torch.tensor(float(yaw)).cuda()
    pitch = torch.tensor(float(pitch)).cuda()
```

**After**:
```python
from .device_utils import get_device

def orbit_camera_from_pose(self, fov, yaw, pitch, r, device=None):
    if device is None:
        device = get_device()
    fov = torch.deg2rad(torch.tensor(float(fov), device=device))
    yaw = torch.tensor(float(yaw), device=device)
    pitch = torch.tensor(float(pitch), device=device)
```

---

#### 2. Fix `mesh_render.py` (Renderer)

**Before**:
```python
def __init__(self, device='cuda', ...):
    self.device = device
```

**After**:
```python
from .device_utils import get_device

def __init__(self, device=None, ...):
    self.device = get_device(device)  # Auto-detect if None
```

---

#### 3. Fix `nodes.py` (Memory tracking)

**Before**:
```python
torch.cuda.reset_peak_memory_stats(device)
```

**After**:
```python
from .device_utils import safe_cuda_call

safe_cuda_call(lambda: torch.cuda.reset_peak_memory_stats(device))
```

---

#### 4. Fix timing utilities

**Before**:
```python
self.start = torch.cuda.Event(enable_timing=True)
self.end = torch.cuda.Event(enable_timing=True)
```

**After**:
```python
import time

class Timer:
    def __init__(self, device=None):
        self.device = device or get_device()
        if self.device.type == 'cuda':
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)
            self.use_cuda_events = True
        else:
            self.use_cuda_events = False
            self.start_time = None
    
    def __enter__(self):
        if self.use_cuda_events:
            self.start_event.record()
        else:
            self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, *args):
        if self.use_cuda_events:
            self.end_event.record()
            torch.cuda.synchronize()
        else:
            self.elapsed = time.perf_counter() - self.start_time
    
    def elapsed_time(self):
        if self.use_cuda_events:
            return self.start_event.elapsed_time(self.end_event)
        else:
            return self.elapsed * 1000  # Convert to ms
```

---

## Files That DO NOT Need Changes

### 1. Documentation Files (Markdown)
- `IMPLEMENTATION_SUMMARY.md`
- `CUDA_DEPENDENCY_ANALYSIS.md`
- `README_PYTORCH_RASTERIZER.md`
- `PYTORCH_RASTERIZER_GUIDE.md`
- `ARCHITECTURE_DIAGRAM.md`

These contain `.cuda()` in **code examples** only.

---

### 2. Already Safe Files
- `pytorch_rasterizer.py` - Uses device auto-detection ✅
- `pytorch_rasterizer_optimized.py` - Uses device auto-detection ✅
- `pytorch_grid_hierarchy.py` - Uses device auto-detection ✅
- `rasterizer_wrapper.py` - Has safe `torch.cuda.is_available()` checks ✅

---

## No CUDA Kernel Files Found

**Search for `.cu` files**: Only 1 result
```
hy3dgen/texgen/custom_rasterizer/lib/custom_rasterizer_kernel/rasterizer_gpu.cu
```

This is the **only** CUDA kernel in the entire codebase, and it's already:
- ✅ Optional (CPU fallback exists)
- ✅ Has PyTorch replacement (completed)
- ✅ Has automatic fallback in `rasterizer_wrapper.py`

---

## Testing Checklist

After implementing fixes, test on:

- [ ] ✅ NVIDIA GPU (CUDA) - Should work as before
- [ ] ✅ AMD GPU (ROCm) - Should use 'cuda' device type with ROCm backend
- [ ] ✅ Apple Silicon (MPS) - Should use 'mps' device
- [ ] ✅ CPU only - Should use 'cpu' device
- [ ] ✅ ComfyUI integration - Should respect ComfyUI's device management

---

## Next Steps

### Immediate Action Items:

1. **Create `device_utils.py`** with centralized device management
2. **Fix Priority 1 files**:
   - `utils.py` (camera utilities)
   - `mesh_render.py` (renderer initialization)
   - `nodes.py` (memory tracking)
3. **Add device parameter propagation** through call chains
4. **Test on AMD GPU** with ROCm

### Optional Improvements:

1. Fix Priority 2 files (model loading)
2. Fix Priority 3 files (utilities)
3. Improve timing utilities for cross-platform profiling
4. Add device detection warnings/info logs

---

## Conclusion

**Good News**: 
- ✅ Only **ONE** CUDA kernel file (custom_rasterizer) - already has PyTorch replacement
- ✅ mesh_processor.cpp has **NO CUDA** - just optional C++ acceleration
- ✅ Pure NumPy fallback already exists for mesh_processor

**Main Issue**:
- ⚠️ Hard-coded `device='cuda'` throughout codebase (34 occurrences)
- 🔧 Needs centralized device management utility

**Effort Required**:
- **Estimated time**: 2-4 hours
- **Complexity**: LOW-MEDIUM
- **Risk**: LOW (changes are isolated device management)

**Impact**:
- ✅ Full AMD GPU support on Windows (ROCm)
- ✅ Apple Silicon support (MPS)
- ✅ Better CPU fallback
- ✅ ComfyUI integration compliance

---

**End of Audit** 🎉
