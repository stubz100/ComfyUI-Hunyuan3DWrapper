"""
Test script to verify device_utils imports work correctly in ComfyUI plugin context.
Run this from the ComfyUI root directory.
"""

import sys
from pathlib import Path

# Add ComfyUI custom_nodes to path (simulating ComfyUI's import system)
plugin_path = Path(__file__).parent
print(f"Plugin path: {plugin_path}")
print(f"device_utils.py exists: {(plugin_path / 'device_utils.py').exists()}")

print("\n" + "="*60)
print("TEST 1: Import device_utils directly")
print("="*60)
try:
    from device_utils import get_device, safe_cuda_call, Timer
    print("✅ PASS: Direct import of device_utils works")
except ImportError as e:
    print(f"❌ FAIL: {e}")

print("\n" + "="*60)
print("TEST 2: Import root utils.py")
print("="*60)
try:
    # This tests if utils.py can import device_utils
    import utils
    print("✅ PASS: utils.py imports successfully")
    print(f"   - has get_device: {hasattr(utils, 'get_device')}")
    print(f"   - has safe_cuda_call: {hasattr(utils, 'safe_cuda_call')}")
except ImportError as e:
    print(f"❌ FAIL: {e}")

print("\n" + "="*60)
print("TEST 3: Import nodes.py")
print("="*60)
try:
    # This tests if nodes.py can import device_utils
    import nodes
    print("✅ PASS: nodes.py imports successfully")
    print(f"   - NODE_CLASS_MAPPINGS exists: {hasattr(nodes, 'NODE_CLASS_MAPPINGS')}")
except ImportError as e:
    print(f"❌ FAIL: {e}")

print("\n" + "="*60)
print("TEST 4: Import nested module (hy3dgen.shapegen.utils)")
print("="*60)
try:
    from hy3dgen.shapegen import utils as shapegen_utils
    print("✅ PASS: hy3dgen.shapegen.utils imports successfully")
    print(f"   - has synchronize_timer: {hasattr(shapegen_utils, 'synchronize_timer')}")
except ImportError as e:
    print(f"❌ FAIL: {e}")

print("\n" + "="*60)
print("TEST 5: Import nested module (hy3dshape.hy3dshape.utils.utils)")
print("="*60)
try:
    from hy3dshape.hy3dshape.utils import utils as hy3dshape_utils
    print("✅ PASS: hy3dshape.hy3dshape.utils.utils imports successfully")
    print(f"   - has synchronize_timer: {hasattr(hy3dshape_utils, 'synchronize_timer')}")
except ImportError as e:
    print(f"❌ FAIL: {e}")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print("If all tests passed, the device_utils import fixes are working correctly!")
print("If any failed, there may be missing dependencies (torch, etc.) which is OK.")
print("The key is that 'device_utils' itself should be found.")
