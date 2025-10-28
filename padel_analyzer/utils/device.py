"""
Device detection utilities for cross-platform ML support.

Automatically detects and selects the best available device:
- CUDA on Windows/Linux with NVIDIA GPUs
- MPS on macOS with Apple Silicon
- CPU as fallback
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def get_device(preferred_device: Optional[str] = None) -> str:
    """
    Get the best available device for ML inference.
    
    Args:
        preferred_device: Optional device preference ("cuda", "mps", "cpu", "auto", or None for auto-detect)
        
    Returns:
        Device string compatible with PyTorch: "cuda", "mps", or "cpu"
    """
    try:
        import torch
        
        # If user specified a device, try to use it
        if preferred_device and preferred_device != "auto":
            if preferred_device == "cuda" and torch.cuda.is_available():
                logger.info("Using CUDA (GPU) device")
                return "cuda"
            elif preferred_device == "mps" and torch.backends.mps.is_available():
                logger.info("Using MPS (Apple Silicon GPU) device")
                return "mps"
            elif preferred_device == "cpu":
                logger.info("Using CPU device (as requested)")
                return "cpu"
            else:
                logger.warning(
                    f"Requested device '{preferred_device}' not available, auto-detecting..."
                )
        
        # Auto-detect best available device
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            logger.info(f"Auto-detected CUDA device: {device_name}")
            return "cuda"
        elif torch.backends.mps.is_available():
            logger.info("Auto-detected MPS (Apple Silicon GPU) device")
            return "mps"
        else:
            logger.info("No GPU available, using CPU device")
            return "cpu"
            
    except ImportError:
        logger.warning("PyTorch not installed, defaulting to CPU")
        return "cpu"


def get_device_info() -> dict:
    """
    Get detailed information about available devices.
    
    Returns:
        Dictionary with device availability and information
    """
    info = {
        "cuda_available": False,
        "mps_available": False,
        "cpu_available": True,  # CPU always available
        "recommended_device": "cpu"
    }
    
    try:
        import torch
        
        info["cuda_available"] = torch.cuda.is_available()
        if info["cuda_available"]:
            info["cuda_device_count"] = torch.cuda.device_count()
            info["cuda_device_name"] = torch.cuda.get_device_name(0)
            info["recommended_device"] = "cuda"
        
        info["mps_available"] = torch.backends.mps.is_available()
        if info["mps_available"] and not info["cuda_available"]:
            info["recommended_device"] = "mps"
            
    except ImportError:
        logger.warning("PyTorch not installed, device info limited")
    
    return info
