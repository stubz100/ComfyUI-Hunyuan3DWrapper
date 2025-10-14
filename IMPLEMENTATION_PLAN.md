# CUDA Dependency Elimination - Implementation Plan

**Date**: October 14, 2025  
**Status**: 🔧 IN PROGRESS  
**Estimated Completion**: 2-4 hours

---

## Quick Summary

After comprehensive audit, found:
- ✅ **Custom rasterizer**: PyTorch replacement already complete
- ✅ **Mesh processor**: Pure NumPy fallback already exists (NO CUDA!)
- ⚠️ **Hard-coded devices**: 34 occurrences need fixing
- ✅ **CUDA kernels**: Only 1 file (custom_rasterizer), already has replacement

---

## What We've Created

### 1. ✅ Device Management Utility (`device_utils.py`)

**Location**: `f:\comfyui\ComfyUI\custom_nodes\ComfyUI-Hunyuan3DWrapper\device_utils.py`

**Features**:
- ✅ Automatic device detection (CUDA/ROCm/MPS/CPU)
- ✅ ComfyUI integration
- ✅ Safe CUDA call wrapper
- ✅ Cross-platform Timer class
- ✅ Device info utilities
- ✅ Optimal dtype detection

**API**:
```python
from device_utils import get_device, to_device, safe_cuda_call, Timer

# Auto-detect best device
device = get_device()

# Move tensors/models safely
tensor = to_device(torch.rand(3, 3))
model = to_device(MyModel())

# Safe CUDA operations
safe_cuda_call(lambda: torch.cuda.synchronize())

# Cross-platform timing
with Timer() as timer:
    result = model(input)
print(f"Elapsed: {timer.elapsed_ms():.2f} ms")
```

---

## Files That Need Fixing

### Priority 1: Core Functionality (HIGH IMPACT)

#### 1. `utils.py` - Camera Utilities
**Lines**: 84-113 (8 occurrences of `.cuda()`)

**Current**:
```python
def orbit_camera_from_pose(self, fov, yaw, pitch, r):
    fov = torch.deg2rad(torch.tensor(float(fov))).cuda()
    yaw = torch.tensor(float(yaw)).cuda()
    pitch = torch.tensor(float(pitch)).cuda()
```

**Fix Needed**:
```python
from device_utils import get_device

def orbit_camera_from_pose(self, fov, yaw, pitch, r, device=None):
    if device is None:
        device = get_device()
    fov = torch.deg2rad(torch.tensor(float(fov), device=device))
    yaw = torch.tensor(float(yaw), device=device)
    pitch = torch.tensor(float(pitch), device=device)
```

---

#### 2. `hy3dgen/texgen/differentiable_renderer/mesh_render.py` - Renderer
**Line**: 136 (hard-coded `device='cuda'` default)

**Current**:
```python
def __init__(self, device='cuda', ...):
    self.device = device
```

**Fix Needed**:
```python
from device_utils import get_device

def __init__(self, device=None, ...):
    if device is None:
        device = get_device()
    self.device = device
```

---

#### 3. `nodes.py` - Memory Tracking
**Lines**: 1224, 1237, 1310, 1324 (4 occurrences)

**Current**:
```python
torch.cuda.reset_peak_memory_stats(device)
```

**Fix Needed**:
```python
from device_utils import safe_cuda_call

safe_cuda_call(lambda: torch.cuda.reset_peak_memory_stats(device))
```

---

#### 4. `hy3dgen/texgen/custom_rasterizer/rasterizer_wrapper.py` - Tests
**Lines**: 130-132, 256-259 (5 occurrences)

**Current**:
```python
test_v = torch.rand(10, 4, device='cuda')
pos = pos.cuda()
tri = tri.cuda()
```

**Fix Needed**:
```python
# For tests (lines 130-132)
device = self.device  # Use wrapper's detected device
test_v = torch.rand(10, 4, device=device)
test_f = torch.randint(0, 10, (5, 3), dtype=torch.int32, device=device)
test_d = torch.zeros(0, device=device)

# For data transfer (lines 256-259)
pos = pos.to(self.device)
tri = tri.to(self.device)
if clamp_depth is not None:
    clamp_depth = clamp_depth.to(self.device)
```

---

### Priority 2: Model Loading (MEDIUM IMPACT)

#### Files to Fix (13 occurrences total):
- `hy3dgen/shapegen/pipelines.py` (1)
- `hy3dshape/hy3dshape/pipelines.py` (3)
- `hy3dgen/shapegen/models/autoencoders/model.py` (2)
- `hy3dshape/hy3dshape/models/autoencoders/model.py` (2)
- `hy3dgen/text2image.py` (1)
- `hy3dgen/shapegen/postprocessors.py` (2)
- `hy3dgen/texgen/utils/alignImg4Tex_utils.py` (1)
- `hy3dgen/texgen/hunyuanpaint/pipeline.py` (1)

**Pattern**:
```python
# Current
def __init__(self, device='cuda', ...):

# Fix to
from device_utils import get_device

def __init__(self, device=None, ...):
    if device is None:
        device = get_device()
```

---

### Priority 3: Timing & Profiling (LOW IMPACT)

#### Files to Fix (10 occurrences total):
- `hy3dshape/hy3dshape/utils/utils.py` (3)
- `hy3dgen/shapegen/utils.py` (3)
- `hy3dshape/hy3dshape/utils/trainings/callback.py` (4)

**Current**:
```python
self.start = torch.cuda.Event(enable_timing=True)
self.end = torch.cuda.Event(enable_timing=True)
torch.cuda.synchronize()
```

**Fix to**:
```python
from device_utils import Timer

# Replace CUDA events with Timer class
self.timer = Timer()
```

---

## Files That DON'T Need Changes

### ✅ Already Safe (using torch.cuda.is_available())
- `rasterizer_wrapper.py` - Lines 66, 127, 166, 174
- `pytorch_rasterizer_optimized.py` - Lines 259, 294, 324, 333
- `pytorch_grid_hierarchy.py` - Line 313

### ✅ Documentation Only
- All `.md` files (examples in documentation)

### ✅ Demo/Test Files
- `hy3dshape/minimal_vae_demo.py` - Demo can require CUDA

---

## Implementation Steps

### Phase 1: Device Management ✅ COMPLETE
- [x] Create `device_utils.py`
- [x] Implement get_device()
- [x] Implement to_device()
- [x] Implement safe_cuda_call()
- [x] Implement Timer class
- [x] Add device info utilities

### Phase 2: High Priority Fixes (Recommended Now)
- [ ] Fix `utils.py` camera utilities
- [ ] Fix `mesh_render.py` renderer
- [ ] Fix `nodes.py` memory tracking
- [ ] Fix `rasterizer_wrapper.py` tests

### Phase 3: Medium Priority Fixes (Optional)
- [ ] Fix model loading files (13 occurrences)
- [ ] Update default parameters

### Phase 4: Low Priority Fixes (Optional)
- [ ] Replace CUDA timing with Timer class
- [ ] Add graceful degradation for profiling

### Phase 5: Testing
- [ ] Test on NVIDIA GPU (CUDA)
- [ ] Test on AMD GPU (ROCm)
- [ ] Test on Apple Silicon (MPS)
- [ ] Test on CPU only
- [ ] Test ComfyUI integration

---

## Automated Fix Script

Want me to create a script to automatically apply all fixes?

**Option 1**: Create `fix_cuda_dependencies.py` that:
1. Backs up original files
2. Applies all regex replacements
3. Adds device_utils imports
4. Validates syntax

**Option 2**: Manual fixes (more careful but slower)
- Use replace_string_in_file for each occurrence
- Review each change
- Test incrementally

---

## Testing After Fixes

### Test 1: Device Detection
```python
from device_utils import get_device, get_device_info

device = get_device()
info = get_device_info()
print(f"Device: {device}")
print(f"Info: {info}")
```

**Expected**:
- NVIDIA: "Device: cuda (NVIDIA GeForce RTX...)"
- AMD: "Device: cuda (AMD Radeon RX 7900 XTX) [ROCm]"
- Apple: "Device: mps (Apple Silicon GPU)"
- CPU: "Device: cpu (CPU)"

---

### Test 2: Timing
```python
from device_utils import Timer
import torch

device = get_device()
x = torch.rand(1000, 1000, device=device)

with Timer() as timer:
    y = x @ x.T

print(f"Matrix multiply: {timer.elapsed_ms():.2f} ms")
```

**Expected**: Works on all devices

---

### Test 3: Safe CUDA Calls
```python
from device_utils import safe_cuda_call

# Should not crash on non-CUDA devices
safe_cuda_call(lambda: torch.cuda.synchronize())
mem = safe_cuda_call(
    lambda: torch.cuda.memory_allocated() / 1024**3,
    fallback=0.0
)
print(f"Memory: {mem:.2f} GB")
```

**Expected**: No crashes, returns 0.0 on non-CUDA

---

### Test 4: Full Pipeline
```python
# Run actual Hunyuan3D pipeline
# Should work on any device
```

---

## Next Actions

### What Would You Like Me To Do?

1. **Start Fixing Files Manually** 🔧
   - I'll fix Priority 1 files one by one
   - You can review each change
   - Safest approach
   
2. **Create Automated Fix Script** 🤖
   - Creates backup
   - Applies all fixes at once
   - Faster but less control
   
3. **Fix Specific Files Only** 🎯
   - Tell me which files are most critical
   - I'll focus on those first
   
4. **Just Test device_utils.py** 🧪
   - Verify the utility works first
   - Then decide on fixes

---

## Risk Assessment

### Low Risk Changes ✅
- Adding `device_utils.py` (new file, no breaking changes)
- Fixing function signatures with device parameter
- Adding safe_cuda_call wrappers

### Medium Risk Changes ⚠️
- Changing default device='cuda' to device=None
- Requires testing to ensure backwards compatibility

### No Risk Changes 👍
- Documentation updates
- Adding device checks before CUDA calls

---

## Backwards Compatibility

All fixes maintain backwards compatibility:

```python
# Old code still works
renderer = MeshRender(device='cuda')

# New code also works
renderer = MeshRender()  # Auto-detects
renderer = MeshRender(device='rocm')  # Explicit ROCm
renderer = MeshRender(device='cpu')  # Force CPU
```

---

## Summary

**Created**:
- ✅ `device_utils.py` - Complete device management solution
- ✅ `COMPLETE_CUDA_AUDIT.md` - Full dependency audit
- ✅ This implementation plan

**Found**:
- 34 hard-coded device references that need fixing
- 1 CUDA kernel file (already has PyTorch replacement)
- 0 additional CUDA dependencies in mesh_processor (pure NumPy fallback exists!)

**Required Effort**:
- 2-4 hours for complete fix
- Can be done incrementally
- Low risk

**Recommended Next Step**: 
Fix Priority 1 files (utils.py, mesh_render.py, nodes.py, rasterizer_wrapper.py) - these are the most impactful and will unlock AMD GPU support immediately.

---

**Ready to proceed?** Let me know which approach you prefer! 🚀
