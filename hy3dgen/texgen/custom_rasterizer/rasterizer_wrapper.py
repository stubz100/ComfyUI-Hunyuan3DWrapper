"""
Automatic Fallback Wrapper for custom_rasterizer
This module automatically handles CPU/GPU rasterization and falls back to alternatives.

Now includes PyTorch implementation for universal GPU support (CUDA/ROCm/MPS)!
"""

import torch
import warnings
from typing import Tuple, Optional, List

class RasterizerWrapper:
    """
    Wrapper that automatically handles CPU/GPU rasterization with fallbacks.
    
    Priority order:
    1. custom_rasterizer with CUDA (fastest)
    2. custom_rasterizer CPU-only (moderate)
    3. nvdiffrast with OpenGL (fast, needs GPU)
    4. nvdiffrast with CUDA (fastest, needs CUDA)
    
    Usage:
        rasterizer = RasterizerWrapper()
        findices, barycentric = rasterizer.rasterize(pos, tri, resolution)
    """
    
    def __init__(self, prefer_mode: Optional[str] = None):
        """
        Initialize rasterizer with automatic backend detection.
        
        Args:
            prefer_mode: Override automatic detection. 
                        Options: 'custom_rasterizer', 'nvdiffrast', None (auto)
        """
        self.backend = None
        self.mode = None
        self.device = 'cpu'
        
        if prefer_mode:
            self._init_preferred(prefer_mode)
        else:
            self._init_auto()
    
    def _init_preferred(self, mode: str):
        """Initialize with user-preferred backend."""
        if mode == 'custom_rasterizer':
            self._try_custom_rasterizer()
        elif mode == 'nvdiffrast':
            self._try_nvdiffrast()
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        if self.backend is None:
            raise RuntimeError(f"Failed to initialize preferred backend: {mode}")
    
    def _try_pytorch_rasterizer(self) -> bool:
        """Try to load PyTorch rasterizer (always available!)."""
        try:
            from .pytorch_rasterizer_optimized import create_optimized_rasterizer
            
            # Create optimized rasterizer (auto-detects best mode)
            self.backend = create_optimized_rasterizer(mode='auto')
            self.device = str(self.backend.device)
            
            # Detect backend type
            if torch.cuda.is_available():
                try:
                    cuda_name = torch.cuda.get_device_name(0) if hasattr(torch.cuda, 'get_device_name') else 'GPU'
                    backend_type = torch.version.hip if hasattr(torch.version, 'hip') else torch.version.cuda
                    
                    if backend_type and 'hip' in str(backend_type).lower():
                        self.mode = 'pytorch_rocm'
                        print(f"✓ Using PyTorch rasterizer with ROCm on {cuda_name}")
                    else:
                        self.mode = 'pytorch_cuda'
                        print(f"✓ Using PyTorch rasterizer with CUDA on {cuda_name}")
                except Exception:
                    self.mode = 'pytorch_cuda'
                    print("✓ Using PyTorch rasterizer with CUDA")
            elif torch.backends.mps.is_available():
                self.mode = 'pytorch_mps'
                print("✓ Using PyTorch rasterizer with MPS (Apple Silicon)")
            else:
                self.mode = 'pytorch_cpu'
                self.device = 'cpu'
                print("✓ Using PyTorch rasterizer on CPU")
                warnings.warn(
                    "PyTorch rasterizer running on CPU. Performance will be slower. "
                    "For better performance, use a CUDA/ROCm GPU."
                )
            
            return True
            
        except ImportError as e:
            warnings.warn(f"Could not load PyTorch rasterizer: {e}")
            return False
    
    def _init_auto(self):
        """Automatically detect and initialize best available backend."""
        # Priority 1: Try PyTorch implementation (universal support!)
        if self._try_pytorch_rasterizer():
            return
        
        # Priority 2: Try custom_rasterizer
        if self._try_custom_rasterizer():
            return
        
        # Priority 3: Fall back to nvdiffrast
        if self._try_nvdiffrast():
            return
        
        raise RuntimeError(
            "No rasterizer backend available!\n"
            "Please install one of:\n"
            "  1. PyTorch (should already be installed)\n"
            "  2. custom_rasterizer (CPU or CUDA version)\n"
            "  3. nvdiffrast (pip install nvdiffrast)\n"
        )
    
    def _try_custom_rasterizer(self) -> bool:
        """Try to load custom_rasterizer."""
        try:
            import custom_rasterizer_kernel as cr
            self.backend = cr
            
            # Test if CUDA is available
            if torch.cuda.is_available():
                try:
                    # Try GPU rasterization - use detected device
                    device = torch.device('cuda')
                    test_v = torch.rand(10, 4, device=device)
                    test_f = torch.randint(0, 10, (5, 3), dtype=torch.int32, device=device)
                    test_d = torch.zeros(0, device=device)
                    cr.rasterize_image(test_v, test_f, test_d, 64, 64, 1e-6, 0)
                    self.mode = 'custom_rasterizer_cuda'
                    self.device = 'cuda'
                    print("✓ Using custom_rasterizer with CUDA (fastest)")
                except Exception:
                    # CUDA failed, must be CPU-only build
                    self.mode = 'custom_rasterizer_cpu'
                    self.device = 'cpu'
                    print("✓ Using custom_rasterizer with CPU (CUDA not available)")
                    warnings.warn(
                        "custom_rasterizer running on CPU. Performance will be slower. "
                        "For better performance, install CUDA version or use nvdiffrast."
                    )
            else:
                self.mode = 'custom_rasterizer_cpu'
                self.device = 'cpu'
                print("✓ Using custom_rasterizer with CPU (CUDA not available)")
            
            return True
            
        except ImportError:
            return False
    
    def _try_nvdiffrast(self) -> bool:
        """Try to load nvdiffrast."""
        try:
            import nvdiffrast.torch as dr
            self.backend = dr
            
            # Try to create GL context (OpenGL)
            try:
                ctx = dr.RasterizeGLContext()
                self.mode = 'nvdiffrast_opengl'
                self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
                print("✓ Using nvdiffrast with OpenGL (good performance, no CUDA needed)")
                self._gl_context = ctx
                return True
            except Exception:
                pass
            
            # Try CUDA context
            if torch.cuda.is_available():
                try:
                    ctx = dr.RasterizeCudaContext()
                    self.mode = 'nvdiffrast_cuda'
                    self.device = 'cuda'
                    print("✓ Using nvdiffrast with CUDA (fastest)")
                    self._cuda_context = ctx
                    return True
                except Exception:
                    pass
            
            return False
            
        except ImportError:
            return False
    
    def rasterize(
        self, 
        pos: torch.Tensor, 
        tri: torch.Tensor, 
        resolution: Tuple[int, int],
        clamp_depth: Optional[torch.Tensor] = None,
        use_depth_prior: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Rasterize triangles.
        
        Args:
            pos: Vertex positions [N, 4] (homogeneous coordinates)
            tri: Triangle indices [M, 3]
            resolution: (height, width)
            clamp_depth: Optional depth clamping
            use_depth_prior: Whether to use depth priority
        
        Returns:
            findices: Face indices [H, W]
            barycentric: Barycentric coordinates [H, W, 3]
        """
        if self.mode.startswith('pytorch'):
            return self._rasterize_pytorch(pos, tri, resolution, clamp_depth, use_depth_prior)
        elif self.mode.startswith('custom_rasterizer'):
            return self._rasterize_custom(pos, tri, resolution, clamp_depth, use_depth_prior)
        elif self.mode.startswith('nvdiffrast'):
            return self._rasterize_nvdiffrast(pos, tri, resolution)
        else:
            raise RuntimeError("No rasterizer backend initialized")
    
    def _rasterize_pytorch(
        self,
        pos: torch.Tensor,
        tri: torch.Tensor,
        resolution: Tuple[int, int],
        clamp_depth: Optional[torch.Tensor],
        use_depth_prior: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Rasterize using PyTorch implementation."""
        if clamp_depth is None:
            clamp_depth = torch.zeros(0, device=pos.device)
        
        # PyTorch rasterizer handles device placement automatically
        findices, barycentric = self.backend.rasterize_image(
            pos, tri, clamp_depth, resolution[1], resolution[0], 1e-6, use_depth_prior
        )
        
        return findices, barycentric
    
    def _rasterize_custom(
        self, 
        pos: torch.Tensor, 
        tri: torch.Tensor, 
        resolution: Tuple[int, int],
        clamp_depth: Optional[torch.Tensor],
        use_depth_prior: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Rasterize using custom_rasterizer."""
        # Ensure tensors are on correct device
        if self.device == 'cpu':
            pos = pos.cpu()
            tri = tri.cpu()
            if clamp_depth is not None:
                clamp_depth = clamp_depth.cpu()
        else:
            pos = pos.to(self.device)
            tri = tri.to(self.device)
            if clamp_depth is not None:
                clamp_depth = clamp_depth.to(self.device)
        
        if clamp_depth is None:
            clamp_depth = torch.zeros(0, device=pos.device)
        
        # Call custom rasterizer
        if pos.dim() == 2:
            pos = pos.unsqueeze(0)
        
        findices, barycentric = self.backend.rasterize_image(
            pos[0], tri, clamp_depth, resolution[1], resolution[0], 1e-6, use_depth_prior
        )
        
        return findices, barycentric
    
    def _rasterize_nvdiffrast(
        self,
        pos: torch.Tensor,
        tri: torch.Tensor, 
        resolution: Tuple[int, int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Rasterize using nvdiffrast."""
        # nvdiffrast expects different format
        # This is a simplified version - full implementation would need more work
        warnings.warn(
            "nvdiffrast fallback not fully implemented. "
            "Please use custom_rasterizer or implement full nvdiffrast support."
        )
        raise NotImplementedError("nvdiffrast fallback needs full implementation")
    
    def interpolate(
        self,
        col: torch.Tensor,
        findices: torch.Tensor, 
        barycentric: torch.Tensor,
        tri: torch.Tensor
    ) -> torch.Tensor:
        """
        Interpolate attributes using barycentric coordinates.
        
        Args:
            col: Vertex attributes [N, C] or [1, N, C]
            findices: Face indices from rasterize [H, W]
            barycentric: Barycentric coords from rasterize [H, W, 3]
            tri: Triangle indices [M, 3]
        
        Returns:
            result: Interpolated attributes [1, H, W, C]
        """
        if self.mode.startswith('pytorch'):
            # Use PyTorch rasterizer's interpolation
            return self.backend.interpolate(col, findices, barycentric, tri)
        
        elif self.mode.startswith('custom_rasterizer'):
            # Move to correct device
            device = 'cuda' if self.device == 'cuda' else 'cpu'
            col = col.to(device)
            
            f = findices - 1 + (findices == 0)
            vcol = col[0, tri.long()[f.long()]]
            result = barycentric.view(*barycentric.shape, 1) * vcol
            result = torch.sum(result, axis=-2)
            return result.view(1, *result.shape)
        else:
            raise NotImplementedError(f"interpolate not implemented for {self.mode}")
    
    def __str__(self) -> str:
        return f"RasterizerWrapper(mode={self.mode}, device={self.device})"


# Convenience function for backward compatibility
def create_rasterizer(**kwargs):
    """Create a rasterizer instance with automatic backend selection."""
    return RasterizerWrapper(**kwargs)


# Example usage
if __name__ == "__main__":
    print("Testing RasterizerWrapper...")
    
    try:
        rast = RasterizerWrapper()
        print(f"Initialized: {rast}")
        
        # Create test data
        device = rast.device
        pos = torch.rand(100, 4, device=device)
        tri = torch.randint(0, 100, (50, 3), dtype=torch.int32, device=device)
        
        # Test rasterization
        findices, barycentric = rast.rasterize(pos, tri, (512, 512))
        
        print(f"✓ Rasterization successful!")
        print(f"  Output shapes: findices={findices.shape}, barycentric={barycentric.shape}")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
