"""
Video loader module for loading and preprocessing video files.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Iterator
import logging
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class VideoLoader:
    """
    Handles loading and preprocessing of video files.
    
    Supports various video formats: mp4, mov, avi, etc.
    """
    
    SUPPORTED_FORMATS = {'.mp4', '.mov', '.avi', '.mkv', '.flv', '.wmv'}
    
    def __init__(self, config: Any):
        """
        Initialize the VideoLoader.
        
        Args:
            config: Configuration object containing video processing settings
        """
        self.config = config
        
    def load(self, video_path: Path) -> Dict[str, Any]:
        """
        Load a video file and extract basic information.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Dictionary containing:
            - path: Original video path
            - format: Video format
            - metadata: Video metadata (fps, resolution, duration, etc.)
            - capture: OpenCV VideoCapture object for frame iteration
            
        Raises:
            ValueError: If video format is not supported or video cannot be opened
            FileNotFoundError: If video file doesn't exist
        """
        video_path = Path(video_path)
        
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        if video_path.suffix.lower() not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported video format: {video_path.suffix}. "
                f"Supported formats: {', '.join(self.SUPPORTED_FORMATS)}"
            )
        
        logger.info(f"Loading video: {video_path}")
        
        # Open video with OpenCV
        capture = cv2.VideoCapture(str(video_path))
        
        if not capture.isOpened():
            raise ValueError(f"Failed to open video file: {video_path}")
        
        # Extract metadata
        metadata = self._extract_metadata(capture)
        
        logger.info(
            f"Video loaded: {metadata['width']}x{metadata['height']} @ {metadata['fps']:.2f} FPS, "
            f"{metadata['frame_count']} frames, {metadata['duration']:.2f}s"
        )
        
        video_data = {
            "path": str(video_path),
            "format": video_path.suffix.lower(),
            "metadata": metadata,
            "capture": capture
        }
        
        return video_data
    
    def _extract_metadata(self, capture: cv2.VideoCapture) -> Dict[str, Any]:
        """
        Extract metadata from video file.
        
        Args:
            capture: OpenCV VideoCapture object
            
        Returns:
            Dictionary containing video metadata
        """
        fps = capture.get(cv2.CAP_PROP_FPS)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate duration
        duration = frame_count / fps if fps > 0 else 0.0
        
        return {
            "fps": fps,
            "width": width,
            "height": height,
            "duration": duration,
            "frame_count": frame_count,
        }
    
    def get_frames(self, video_data: Dict[str, Any]) -> Iterator[np.ndarray]:
        """
        Get frames from loaded video.
        
        Args:
            video_data: Video data dictionary from load()
            
        Yields:
            Video frames as numpy arrays (BGR format)
        """
        capture = video_data.get("capture")
        if capture is None:
            raise ValueError("Video not properly loaded. Call load() first.")
        
        # Reset to beginning
        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        while True:
            ret, frame = capture.read()
            if not ret:
                break
            yield frame
    
    def get_frame_at(self, video_data: Dict[str, Any], frame_number: int) -> Optional[np.ndarray]:
        """
        Get a specific frame from the video.
        
        Args:
            video_data: Video data dictionary from load()
            frame_number: Frame index to retrieve
            
        Returns:
            Frame as numpy array or None if frame doesn't exist
        """
        capture = video_data.get("capture")
        if capture is None:
            raise ValueError("Video not properly loaded. Call load() first.")
        
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = capture.read()
        
        return frame if ret else None
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess a single video frame.
        
        Args:
            frame: Raw video frame (BGR format)
            
        Returns:
            Preprocessed frame ready for analysis
        """
        # Apply target resolution if specified
        if self.config.video.target_resolution is not None:
            target_width, target_height = self.config.video.target_resolution
            frame = cv2.resize(frame, (target_width, target_height))
        
        return frame
    
    def release(self, video_data: Dict[str, Any]):
        """
        Release video resources.
        
        Args:
            video_data: Video data dictionary from load()
        """
        capture = video_data.get("capture")
        if capture is not None:
            capture.release()
