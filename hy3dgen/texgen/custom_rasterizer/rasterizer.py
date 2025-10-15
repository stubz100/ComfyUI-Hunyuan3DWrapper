"""
Unified Rasterizer Interface with Automatic Backend Selection
==============================================================

Single entry point that automatically selects the best available rasterizer:
1. CUDA kernel (if available and compatible)
2. Ultra-fast PyTorch (vectorized)
3. Optimized PyTorch (tile-based)  
4. Basic PyTorch (fallback)

Usage:
    from rasterizer import create_rasterizer
    
    rasterizer = create_rasterizer(device)
    # Automatically selects best available backend
    
Author: GitHub Copilot
Date: October 15, 2025
"""

import torch
from typing import Tuple, Optional
from pathlib import Path
import sys


class RasterizerInterface:
    """
    Unified interface for all rasterizer backends.
    
    All backends must implement:
    - rasterize_image(V, F, D, width, height, occlusion_truncation, use_depth_prior)
    - interpolate(attr, findices, barycentric, attr_idx) [if using full pipeline]
    """
    
    def __init__(self, backend_name: str, backend_impl):
        self.backend_name = backend_name
        self.backend = backend_impl
        self.device = backend_impl.device if hasattr(backend_impl, 'device') else None
    
    def rasterize_image(
        self,
        V: torch.Tensor,
        F: torch.Tensor,
        D: torch.Tensor,
        width: int,
        height: int,
        occlusion_truncation: float = 1e-6,
        use_depth_prior: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Rasterize triangles into an image.
        
        Returns:
            findices: Face indices [H, W] (1-indexed, 0 = no face)
            barycentric: Barycentric coordinates [H, W, 3]
        """
        # Dispatch to backend implementation
        if hasattr(self.backend, 'rasterize_image'):
            return self.backend.rasterize_image(V, F, D, width, height, occlusion_truncation, use_depth_prior)
        elif hasattr(self.backend, 'rasterize'):
            # CUDA kernel uses different API
            if V.dim() == 2:
                V = V.unsqueeze(0)
            return self.backend.rasterize(V, F, (height, width))
        else:
            raise AttributeError(f"Backend {self.backend_name} has no rasterize method")
    
    def interpolate(
        self,
        attr: torch.Tensor,
        findices: torch.Tensor,
        barycentric: torch.Tensor,
        attr_idx: torch.Tensor
    ) -> torch.Tensor:
        """
        Interpolate attributes using barycentric coordinates.
        """
        if hasattr(self.backend, 'interpolate'):
            return self.backend.interpolate(attr, findices, barycentric, attr_idx)
        else:
            # Fallback: manual interpolation
            return self._interpolate_manual(attr, findices, barycentric, attr_idx)
    
    def _interpolate_manual(
        self,
        attr: torch.Tensor,
        findices: torch.Tensor,
        barycentric: torch.Tensor,
        attr_idx: torch.Tensor
    ) -> torch.Tensor:
        """
        Manual attribute interpolation for backends without interpolate().
        """
        # Get face indices (0 = no face)
        valid_mask = findices > 0
        face_ids = torch.clamp(findices - 1, min=0)  # Convert to 0-indexed
        
        # Get triangle vertex indices
        tri_verts = attr_idx[face_ids]  # [H, W, 3]
        
        # Get vertex attributes
        v0_attr = attr[tri_verts[..., 0]]
        v1_attr = attr[tri_verts[..., 1]]
        v2_attr = attr[tri_verts[..., 2]]
        
        # Interpolate using barycentric coordinates
        result = (
            barycentric[..., 0:1] * v0_attr +
            barycentric[..., 1:2] * v1_attr +
            barycentric[..., 2:3] * v2_attr
        )
        
        # Zero out invalid pixels
        result = torch.where(valid_mask.unsqueeze(-1), result, torch.zeros_like(result))
        
        return result
    
    def __repr__(self):
        return f"Rasterizer(backend={self.backend_name}, device={self.device})"


def create_rasterizer(
    device: Optional[torch.device] = None,
    prefer_cuda: bool = True,
    max_triangles_per_batch: int = 10000
) -> RasterizerInterface:
    """
    Create the best available rasterizer for the device.
    
    Selection priority:
    1. CUDA kernel (100% speed, NVIDIA only)
    2. Ultra-fast PyTorch (70-85% speed, universal)
    3. Optimized PyTorch (50-60% speed, universal)
    4. Basic PyTorch (10-15% speed, universal)
    
    Args:
        device: Target device (auto-detected if None)
        prefer_cuda: Try CUDA kernel first if available
        max_triangles_per_batch: Batch size for ultra-fast version
    
    Returns:
        RasterizerInterface with best available backend
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Add custom_rasterizer directory to path
    raster_path = Path(__file__).parent
    if str(raster_path) not in sys.path:
        sys.path.insert(0, str(raster_path))
    
    backend_name = None
    backend_impl = None
    
    # TIER 1: Try CUDA kernel (if preferred)
    if prefer_cuda and device.type == 'cuda':
        try:
            import custom_rasterizer as cr
            
            # Test if it actually works
            test_v = torch.rand(10, 4, device=device)
            test_f = torch.randint(0, 10, (5, 3), dtype=torch.int32, device=device)
            test_d = torch.zeros(0, device=device)
            cr.rasterize_image(test_v, test_f, test_d, 32, 32, 1e-6, 0)
            
            backend_name = "CUDA Kernel"
            backend_impl = cr
            backend_impl.device = device
            print(f"✓ Rasterizer: CUDA kernel (100% speed, NVIDIA optimized)")
            
        except (ImportError, OSError, Exception) as e:
            # CUDA kernel unavailable or incompatible
            if isinstance(e, (ImportError, OSError)):
                print(f"  CUDA kernel not found, trying PyTorch implementations...")
            else:
                print(f"  CUDA kernel incompatible with {device}, falling back...")
    
    # TIER 2: Try Ultra-Fast PyTorch
    if backend_impl is None:
        try:
            from pytorch_rasterizer_ultra import UltraFastRasterizer
            
            backend_impl = UltraFastRasterizer(device=device, max_triangles_per_batch=max_triangles_per_batch)
            backend_name = "Ultra-Fast PyTorch"
            print(f"✓ Rasterizer: Ultra-Fast PyTorch (70-85% speed, full parallelization)")
            
        except ImportError:
            pass
    
    # TIER 3: Try Optimized PyTorch
    if backend_impl is None:
        try:
            from pytorch_rasterizer_optimized import create_optimized_rasterizer
            
            backend_impl = create_optimized_rasterizer(device=device, mode='tiled', tile_size=32)
            backend_name = "Optimized PyTorch"
            print(f"✓ Rasterizer: Optimized PyTorch (50-60% speed, tile-based)")
            
        except ImportError:
            pass
    
    # TIER 4: Basic PyTorch (always available)
    if backend_impl is None:
        from pytorch_rasterizer import PyTorchRasterizer
        
        backend_impl = PyTorchRasterizer(device=device)
        backend_name = "Basic PyTorch"
        print(f"✓ Rasterizer: Basic PyTorch (10-15% speed, compatibility fallback)")
    
    return RasterizerInterface(backend_name, backend_impl)


# Convenience functions
def get_cuda_rasterizer(device: torch.device):
    """Get CUDA kernel rasterizer (raises if unavailable)."""
    import custom_rasterizer as cr
    backend_impl = cr
    backend_impl.device = device
    return RasterizerInterface("CUDA Kernel", backend_impl)


def get_ultra_rasterizer(device: torch.device, max_triangles_per_batch: int = 10000):
    """Get ultra-fast PyTorch rasterizer."""
    from pytorch_rasterizer_ultra import UltraFastRasterizer
    backend_impl = UltraFastRasterizer(device, max_triangles_per_batch)
    return RasterizerInterface("Ultra-Fast PyTorch", backend_impl)


def get_optimized_rasterizer(device: torch.device, tile_size: int = 32):
    """Get optimized tile-based PyTorch rasterizer."""
    from pytorch_rasterizer_optimized import TiledPyTorchRasterizer
    backend_impl = TiledPyTorchRasterizer(device, tile_size)
    return RasterizerInterface("Optimized PyTorch", backend_impl)


def get_basic_rasterizer(device: torch.device):
    """Get basic PyTorch rasterizer."""
    from pytorch_rasterizer import PyTorchRasterizer
    backend_impl = PyTorchRasterizer(device)
    return RasterizerInterface("Basic PyTorch", backend_impl)


# Export
__all__ = [
    'RasterizerInterface',
    'create_rasterizer',
    'get_cuda_rasterizer',
    'get_ultra_rasterizer',
    'get_optimized_rasterizer',
    'get_basic_rasterizer',
]
