"""
CPU-Only Build Script for custom_rasterizer
This version removes CUDA dependencies and uses only CPU fallback implementation.
"""

from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension
import torch
import sys

torch_version = torch.__version__.split('+')[0].replace('.', '')
version = f"0.1.0+cpu.torch{torch_version}"

print(f"Building custom_rasterizer CPU-only version {version}")
print("=" * 60)
print("This build removes CUDA dependencies and uses CPU fallback.")
print("Performance will be lower but compatibility is universal.")
print("=" * 60)

# CPU-only extension - no CUDA files
custom_rasterizer_module = CppExtension(
    'custom_rasterizer_kernel',
    sources=[
        'lib/custom_rasterizer_kernel/rasterizer.cpp',
        'lib/custom_rasterizer_kernel/grid_neighbor.cpp',
        # Note: rasterizer_gpu.cu is NOT included
    ],
    extra_compile_args={
        'cxx': ['-O3', '-DCPU_ONLY'] if sys.platform != 'win32' else ['/O2', '/DCPU_ONLY']
    },
    define_macros=[
        ('CPU_ONLY', None),
    ]
)

setup(
    packages=find_packages(),
    version=version,
    name='custom_rasterizer',
    description='CPU-only custom rasterizer for Hunyuan3D texture generation',
    include_package_data=True,
    package_dir={'': '.'},
    ext_modules=[custom_rasterizer_module],
    cmdclass={
        'build_ext': BuildExtension.with_options(no_python_abi_suffix=True)
    },   
)

print("\n" + "=" * 60)
print("Build complete! The CPU-only version has been installed.")
print("Note: This version will be slower than CUDA but works everywhere.")
print("=" * 60)
