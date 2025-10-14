"""
Device Management Utilities for ComfyUI-Hunyuan3DWrapper

Provides centralized device detection and management that works across:
- NVIDIA GPUs (CUDA)
- AMD GPUs (ROCm) 
- Apple Silicon (MPS)
- CPU fallback

Integrates with ComfyUI's model management when available.
"""

import torch
import logging
from typing import Optional, Union, Callable, Any

logger = logging.getLogger(__name__)


def get_device(preferred: Optional[str] = None) -> torch.device:
    """Get the best available device for computation.
    
    Priority order:
    1. User-specified preferred device (if provided)
    2. ComfyUI's device management (if available)
    3. CUDA/ROCm (if available)
    4. MPS (if available on Apple Silicon)
    5. CPU (fallback)
    
    Args:
        preferred: Optional preferred device string ('cuda', 'rocm', 'mps', 'cpu', etc.)
                  If provided, returns this device without validation.
    
    Returns:
        torch.device: The selected device
    
    Examples:
        >>> device = get_device()  # Auto-detect best device
        >>> device = get_device('cuda')  # Force CUDA
        >>> device = get_device('cpu')  # Force CPU
    """
    if preferred is not None:
        try:
            return torch.device(preferred)
        except:
            logger.warning(f"Invalid preferred device '{preferred}', falling back to auto-detection")
    
    # Try ComfyUI's device management first
    try:
        import comfy.model_management as mm
        device = mm.get_torch_device()
        logger.debug(f"Using ComfyUI device management: {device}")
        return device
    except ImportError:
        logger.debug("ComfyUI model management not available, using PyTorch device detection")
    except Exception as e:
        logger.debug(f"Error accessing ComfyUI device management: {e}")
    
    # Fallback to PyTorch device detection
    if torch.cuda.is_available():
        device = torch.device('cuda')
        try:
            # Try to get device name for logging
            device_name = torch.cuda.get_device_name(0)
            
            # Detect if we're using ROCm (AMD GPU)
            if hasattr(torch.version, 'hip') and torch.version.hip is not None:
                logger.info(f"Using AMD GPU via ROCm: {device_name}")
            else:
                logger.info(f"Using NVIDIA GPU via CUDA: {device_name}")
        except:
            logger.info("Using CUDA device")
        return device
    
    # Check for Apple Silicon MPS
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        logger.info("Using Apple Silicon MPS device")
        return device
    
    # Fallback to CPU
    device = torch.device('cpu')
    logger.info("Using CPU device (no GPU detected)")
    return device


def to_device(
    tensor_or_module: Union[torch.Tensor, torch.nn.Module], 
    device: Optional[Union[str, torch.device]] = None,
    **kwargs
) -> Union[torch.Tensor, torch.nn.Module]:
    """Move tensor or module to specified device safely.
    
    Args:
        tensor_or_module: PyTorch tensor or nn.Module to move
        device: Target device (None = auto-detect best device)
        **kwargs: Additional arguments passed to .to() method (e.g., dtype, non_blocking)
    
    Returns:
        Moved tensor or module
    
    Examples:
        >>> tensor = torch.rand(3, 3)
        >>> tensor_gpu = to_device(tensor)  # Auto-detect device
        >>> tensor_cpu = to_device(tensor, 'cpu')  # Force CPU
        >>> model_gpu = to_device(model, dtype=torch.float16)  # Move with dtype
    """
    if device is None:
        device = get_device()
    elif isinstance(device, str):
        device = torch.device(device)
    
    try:
        return tensor_or_module.to(device, **kwargs)
    except Exception as e:
        logger.warning(f"Failed to move to {device}: {e}. Trying CPU fallback.")
        return tensor_or_module.to('cpu', **kwargs)


def safe_cuda_call(func: Callable, fallback: Any = None) -> Any:
    """Execute CUDA-specific function safely with automatic fallback.
    
    Only executes the function if CUDA is available. Returns fallback otherwise.
    Useful for CUDA-specific operations like synchronize() or memory stats.
    
    Args:
        func: Function to execute (should be CUDA-specific)
        fallback: Value to return if CUDA not available or function fails
    
    Returns:
        Result of func() if successful, otherwise fallback
    
    Examples:
        >>> # Synchronize only if on CUDA
        >>> safe_cuda_call(lambda: torch.cuda.synchronize())
        
        >>> # Get memory stats with fallback
        >>> mem = safe_cuda_call(
        ...     lambda: torch.cuda.memory_allocated() / 1024**3,
        ...     fallback=0.0
        ... )
    """
    try:
        if torch.cuda.is_available():
            return func()
        return fallback
    except Exception as e:
        logger.debug(f"CUDA call failed: {e}")
        return fallback


def is_cuda_available() -> bool:
    """Check if CUDA (including ROCm) is available.
    
    Returns:
        bool: True if CUDA/ROCm available, False otherwise
    """
    return torch.cuda.is_available()


def is_rocm() -> bool:
    """Check if running on AMD GPU via ROCm.
    
    Returns:
        bool: True if ROCm detected, False otherwise
    """
    return hasattr(torch.version, 'hip') and torch.version.hip is not None


def is_mps_available() -> bool:
    """Check if Apple Silicon MPS is available.
    
    Returns:
        bool: True if MPS available, False otherwise
    """
    return hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()


def get_device_info() -> dict:
    """Get detailed information about available devices.
    
    Returns:
        dict: Device information including type, name, memory, etc.
    
    Examples:
        >>> info = get_device_info()
        >>> print(f"Device: {info['type']}, Name: {info.get('name', 'N/A')}")
    """
    info = {
        'type': None,
        'name': None,
        'memory_total': None,
        'memory_allocated': None,
        'backend': None,
    }
    
    if torch.cuda.is_available():
        info['type'] = 'cuda'
        try:
            info['name'] = torch.cuda.get_device_name(0)
            info['memory_total'] = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            info['memory_allocated'] = torch.cuda.memory_allocated(0) / 1024**3  # GB
            
            if is_rocm():
                info['backend'] = 'ROCm'
            else:
                info['backend'] = 'CUDA'
        except:
            pass
    elif is_mps_available():
        info['type'] = 'mps'
        info['name'] = 'Apple Silicon GPU'
        info['backend'] = 'MPS'
    else:
        info['type'] = 'cpu'
        info['name'] = 'CPU'
        info['backend'] = 'CPU'
    
    return info


class Timer:
    """Cross-platform timer that works on CUDA, MPS, and CPU.
    
    Automatically uses CUDA events on CUDA devices for accurate GPU timing,
    and falls back to CPU time.perf_counter() on other devices.
    
    Examples:
        >>> timer = Timer()
        >>> with timer:
        ...     # Your code here
        ...     result = model(input)
        >>> print(f"Elapsed: {timer.elapsed_ms():.2f} ms")
        
        >>> # Or manual control
        >>> timer = Timer()
        >>> timer.start()
        >>> result = model(input)
        >>> timer.stop()
        >>> print(f"Elapsed: {timer.elapsed_ms():.2f} ms")
    """
    
    def __init__(self, device: Optional[Union[str, torch.device]] = None):
        """Initialize timer for specified device.
        
        Args:
            device: Device to time operations on (None = auto-detect)
        """
        if device is None:
            device = get_device()
        elif isinstance(device, str):
            device = torch.device(device)
        
        self.device = device
        self.use_cuda_events = (device.type == 'cuda' and torch.cuda.is_available())
        
        if self.use_cuda_events:
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)
        else:
            self.start_time = None
            self.end_time = None
    
    def start(self):
        """Start the timer."""
        if self.use_cuda_events:
            self.start_event.record()
        else:
            import time
            self.start_time = time.perf_counter()
    
    def stop(self):
        """Stop the timer."""
        if self.use_cuda_events:
            self.end_event.record()
            torch.cuda.synchronize()
        else:
            import time
            self.end_time = time.perf_counter()
    
    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds.
        
        Returns:
            float: Elapsed time in milliseconds
        """
        if self.use_cuda_events:
            return self.start_event.elapsed_time(self.end_event)
        else:
            if self.start_time is None or self.end_time is None:
                return 0.0
            return (self.end_time - self.start_time) * 1000.0
    
    def elapsed_s(self) -> float:
        """Get elapsed time in seconds.
        
        Returns:
            float: Elapsed time in seconds
        """
        return self.elapsed_ms() / 1000.0
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, *args):
        """Context manager exit."""
        self.stop()


def get_optimal_dtype(device: Optional[Union[str, torch.device]] = None) -> torch.dtype:
    """Get optimal dtype for specified device.
    
    Returns fp16 for CUDA/ROCm, fp32 for MPS/CPU.
    
    Args:
        device: Device to check (None = auto-detect)
    
    Returns:
        torch.dtype: Optimal dtype for the device
    """
    if device is None:
        device = get_device()
    elif isinstance(device, str):
        device = torch.device(device)
    
    # CUDA/ROCm can efficiently use fp16
    if device.type == 'cuda':
        return torch.float16
    
    # MPS and CPU work better with fp32
    return torch.float32


# Convenience function for creating tensors on the right device
def tensor(*args, device: Optional[Union[str, torch.device]] = None, **kwargs) -> torch.Tensor:
    """Create a tensor on the specified device.
    
    Wrapper around torch.tensor() that automatically handles device selection.
    
    Args:
        *args: Arguments passed to torch.tensor()
        device: Target device (None = auto-detect)
        **kwargs: Additional keyword arguments passed to torch.tensor()
    
    Returns:
        torch.Tensor: Created tensor on specified device
    
    Examples:
        >>> t = tensor([1, 2, 3])  # Auto-detect device
        >>> t = tensor([1, 2, 3], device='cpu')  # Force CPU
        >>> t = tensor([[1, 2], [3, 4]], dtype=torch.float32)  # With dtype
    """
    if device is None:
        device = get_device()
    return torch.tensor(*args, device=device, **kwargs)


# Module-level device cache to avoid repeated detection
_cached_device = None

def get_cached_device() -> torch.device:
    """Get cached device (computed once, reused).
    
    Faster than get_device() but doesn't adapt to runtime changes.
    Use this when you know the device won't change during runtime.
    
    Returns:
        torch.device: Cached device
    """
    global _cached_device
    if _cached_device is None:
        _cached_device = get_device()
    return _cached_device


def clear_device_cache():
    """Clear cached device, forcing re-detection on next call."""
    global _cached_device
    _cached_device = None


# Print device info on module import
if __name__ != '__main__':
    try:
        device = get_device()
        info = get_device_info()
        logger.info(f"Device initialized: {info['type']} ({info.get('name', 'Unknown')})")
    except Exception as e:
        logger.debug(f"Error during device initialization: {e}")
