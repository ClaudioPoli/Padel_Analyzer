"""
Configuration management for the Padel Analyzer.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple
import json
from pathlib import Path


@dataclass
class VideoConfig:
    """Video processing configuration."""
    supported_formats: List[str] = field(default_factory=lambda: ['.mp4', '.mov', '.avi', '.mkv'])
    target_fps: Optional[int] = None  # If None, use original FPS
    target_resolution: Optional[Tuple[int, int]] = None  # (width, height), if None, use original


@dataclass
class TrackingConfig:
    """Tracking configuration for players and ball."""
    player_detection_confidence: float = 0.5
    ball_detection_confidence: float = 0.3
    max_tracking_distance: int = 100  # pixels
    interpolate_missing: bool = True
    max_interpolation_gap: int = 5  # frames


@dataclass
class FieldDetectionConfig:
    """Field detection configuration."""
    line_detection_threshold: int = 100
    corner_detection_tolerance: int = 10  # pixels
    use_homography: bool = True
    auto_calibrate: bool = True


@dataclass
class ModelConfig:
    """Model configuration."""
    player_model: str = "yolov8n.pt"  # Model name or path for detection
    ball_model: str = "yolov8n.pt"  # Model name or path
    device: str = "auto"  # "auto", "cpu", "cuda", or "mps"
    batch_size: int = 1


@dataclass
class Config:
    """
    Main configuration class for Padel Analyzer.
    
    Contains all configuration parameters for video processing,
    tracking, detection, and model settings.
    """
    video: VideoConfig = field(default_factory=VideoConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    field_detection: FieldDetectionConfig = field(default_factory=FieldDetectionConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    
    def __post_init__(self):
        """Initialize nested configs if they're dictionaries."""
        if isinstance(self.video, dict):
            self.video = VideoConfig(**self.video)
        if isinstance(self.tracking, dict):
            self.tracking = TrackingConfig(**self.tracking)
        if isinstance(self.field_detection, dict):
            self.field_detection = FieldDetectionConfig(**self.field_detection)
        if isinstance(self.model, dict):
            self.model = ModelConfig(**self.model)
    
    @classmethod
    def from_file(cls, config_path: str) -> 'Config':
        """
        Load configuration from a JSON file.
        
        Args:
            config_path: Path to JSON configuration file
            
        Returns:
            Config object
        """
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)
    
    def to_file(self, config_path: str):
        """
        Save configuration to a JSON file.
        
        Args:
            config_path: Path to save JSON configuration
        """
        config_dict = {
            'video': self.video.__dict__,
            'tracking': self.tracking.__dict__,
            'field_detection': self.field_detection.__dict__,
            'model': self.model.__dict__
        }
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def update(self, **kwargs):
        """
        Update configuration parameters.
        
        Args:
            **kwargs: Configuration parameters to update
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
