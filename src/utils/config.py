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
    use_keypoints_for_team: bool = False  # Use keypoints for team assignment
    # --- Identity stabilization parameters ---
    identity_history_len: int = 30  # frames of position/bbox history per player
    identity_init_frames: int = 5  # frames to collect before assigning stable IDs
    occlusion_iou_threshold: float = 0.10  # bbox IoU above which players are "occluding"
    lost_player_max_frames: int = 60  # keep predicting lost player for N frames
    swap_penalty: float = 150.0  # extra cost for swapping two overlapping players
    appearance_cost_weight: float = 250.0  # scale appearance dissimilarity
    kalman_process_noise: float = 5.0  # Kalman filter process noise
    kalman_measurement_noise: float = 10.0  # Kalman filter measurement noise


@dataclass
class FieldDetectionConfig:
    """Field detection configuration."""
    line_detection_threshold: int = 100
    corner_detection_tolerance: int = 10  # pixels
    use_homography: bool = True
    auto_calibrate: bool = True


@dataclass
class FieldKeypointsConfig:
    """Configuration for YOLO-pose based field keypoints detection."""
    enabled: bool = True  # Use keypoints-based detection instead of legacy
    model_path: str = "models/field_skeleton_yolo11m_pose5/weights/best.pt"
    min_keypoint_confidence: float = 0.5  # Min confidence to trust a keypoint
    min_detection_confidence: float = 0.25  # Min box detection confidence
    interpolate_missing: bool = True  # Use geometric interpolation for low-confidence kpts
    temporal_smoothing: bool = True  # Smooth keypoints across frames in video
    smoothing_window: int = 5  # Number of frames for temporal smoothing
    num_sample_frames: int = 10  # Number of frames to sample for static detection


@dataclass
class ModelConfig:
    """Model configuration."""
    player_model: str = "yolov8n.pt"  # Model name or path for detection
    ball_model: str = "yolov8n.pt"  # Model name or path
    device: str = "auto"  # "auto", "cpu", "cuda", or "mps"
    batch_size: int = 1


@dataclass
class PoseConfig:
    """
    Pose estimation configuration.
    
    Uses YOLOv8-Pose for zero-shot pose estimation (recommended).
    For fine-tuning, set use_custom_model=True and provide model_path.
    """
    enabled: bool = True  # Enable pose estimation
    pose_model: str = "yolov8n-pose.pt"  # YOLO pose model (n/s/m/l/x variants)
    min_confidence: float = 0.25  # Minimum keypoint confidence
    estimate_for_all_frames: bool = False  # If False, sample frames for efficiency
    frame_sample_rate: int = 3  # Process every Nth frame when sampling


@dataclass
class ActionRecognitionConfig:
    """
    Action/shot recognition configuration.
    
    Current approach: Rule-based classification using pose geometry.
    For production: Set use_ml_model=True and provide a fine-tuned model.
    
    Recommended fine-tuning approach:
    - Collect ~500-1000 examples per action class
    - Use pose sequences (16 frames) as input
    - Train LSTM or Transformer-based classifier
    """
    enabled: bool = True  # Enable action recognition
    use_ml_model: bool = False  # Use ML model instead of rules
    model_path: Optional[str] = None  # Path to fine-tuned action recognition model
    buffer_size: int = 16  # Number of frames for temporal analysis
    min_action_confidence: float = 0.5  # Minimum confidence to report action


@dataclass
class HeatmapConfig:
    """
    Heatmap generation configuration.
    
    Controls how player position heatmaps are generated and rendered.
    Uses field keypoints + homography to map player positions to real court coordinates.
    """
    enabled: bool = True  # Enable heatmap generation
    resolution: int = 100  # Grid cells per meter (higher = finer detail)
    sigma: float = 0.5  # Gaussian smoothing sigma in meters
    use_feet_position: bool = True  # Use bottom-center of bbox (feet) instead of center
    per_player: bool = True  # Generate per-player heatmaps
    per_team: bool = True  # Generate per-team heatmaps
    colormap: str = "jet"  # Matplotlib/OpenCV colormap name
    court_padding: float = 1.0  # Extra meters around court edges
    alpha: float = 0.6  # Overlay transparency (0=transparent, 1=opaque)
    output_width: int = 600  # Width of rendered court diagram in pixels
    normalize: bool = True  # Normalize heatmap values to [0, 1]


@dataclass
class Config:
    """
    Main configuration class for Padel Analyzer.
    
    Contains all configuration parameters for video processing,
    tracking, detection, pose estimation, action recognition, and model settings.
    """
    video: VideoConfig = field(default_factory=VideoConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    field_detection: FieldDetectionConfig = field(default_factory=FieldDetectionConfig)
    field_keypoints: FieldKeypointsConfig = field(default_factory=FieldKeypointsConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    pose: PoseConfig = field(default_factory=PoseConfig)
    action_recognition: ActionRecognitionConfig = field(default_factory=ActionRecognitionConfig)
    heatmap: HeatmapConfig = field(default_factory=HeatmapConfig)
    
    def __post_init__(self):
        """Initialize nested configs if they're dictionaries."""
        if isinstance(self.video, dict):
            self.video = VideoConfig(**self.video)
        if isinstance(self.tracking, dict):
            self.tracking = TrackingConfig(**self.tracking)
        if isinstance(self.field_detection, dict):
            self.field_detection = FieldDetectionConfig(**self.field_detection)
        if isinstance(self.field_keypoints, dict):
            self.field_keypoints = FieldKeypointsConfig(**self.field_keypoints)
        if isinstance(self.model, dict):
            self.model = ModelConfig(**self.model)
        if isinstance(self.pose, dict):
            self.pose = PoseConfig(**self.pose)
        if isinstance(self.action_recognition, dict):
            self.action_recognition = ActionRecognitionConfig(**self.action_recognition)
        if isinstance(self.heatmap, dict):
            self.heatmap = HeatmapConfig(**self.heatmap)
    
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
            'field_keypoints': self.field_keypoints.__dict__,
            'model': self.model.__dict__,
            'pose': self.pose.__dict__,
            'action_recognition': self.action_recognition.__dict__,
            'heatmap': self.heatmap.__dict__
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
