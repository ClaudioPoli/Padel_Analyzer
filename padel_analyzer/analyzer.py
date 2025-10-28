"""
Main analyzer module that orchestrates the video analysis pipeline.
"""

from pathlib import Path
from typing import Optional, Dict, Any, List

from .video.video_loader import VideoLoader
from .tracking.player_tracker import PlayerTracker
from .tracking.ball_tracker import BallTracker
from .detection.field_detector import FieldDetector
from .utils.config import Config


class PadelAnalyzer:
    """
    Main class for analyzing padel match videos.
    
    This class orchestrates the entire analysis pipeline:
    1. Video loading and preprocessing
    2. Field detection
    3. Player tracking
    4. Ball tracking
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the PadelAnalyzer.
        
        Args:
            config: Configuration object. If None, default config is used.
        """
        self.config = config or Config()
        self.video_loader = VideoLoader(self.config)
        self.field_detector = FieldDetector(self.config)
        self.player_tracker = PlayerTracker(self.config)
        self.ball_tracker = BallTracker(self.config)
        
    def analyze_video(self, video_path: str) -> Dict[str, Any]:
        """
        Analyze a padel match video.
        
        Args:
            video_path: Path to the video file (mp4, mov, etc.)
            
        Returns:
            Dictionary containing analysis results:
            - field_info: Detected field boundaries and characteristics
            - player_tracks: Tracked player positions over time
            - ball_tracks: Tracked ball positions over time
            - metadata: Video metadata and analysis statistics
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Load video
        video_data = self.video_loader.load(video_path)
        
        try:
            # Detect field
            field_info = self.field_detector.detect(video_data)
            
            # Track players
            player_tracks = self.player_tracker.track(video_data, field_info)
            
            # Track ball
            ball_tracks = self.ball_tracker.track(video_data, field_info)
            
            return {
                "field_info": field_info,
                "player_tracks": player_tracks,
                "ball_tracks": ball_tracks,
                "metadata": video_data.get("metadata", {})
            }
        finally:
            # Clean up video resources
            self.video_loader.release(video_data)
    
    def analyze_video_batch(self, video_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Analyze multiple padel match videos.
        
        Args:
            video_paths: List of paths to video files
            
        Returns:
            List of analysis results for each video
        """
        results = []
        for video_path in video_paths:
            try:
                result = self.analyze_video(video_path)
                results.append(result)
            except Exception as e:
                results.append({"error": str(e), "video_path": video_path})
        return results
