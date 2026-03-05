"""
Main analyzer module that orchestrates the video analysis pipeline.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, List

from .video.video_loader import VideoLoader
from .tracking.player_tracker import PlayerTracker
from .tracking.ball_tracker import BallTracker
from .tracking.pose_estimator import PoseEstimator
from .tracking.action_recognizer import ActionRecognizer, PadelAction
from .detection.field_detector import FieldDetector
from .detection.keypoint_field_detector import KeypointFieldDetector
from .utils.config import Config

logger = logging.getLogger(__name__)


class PadelAnalyzer:
    """
    Main class for analyzing padel match videos.
    
    This class orchestrates the entire analysis pipeline:
    1. Video loading and preprocessing
    2. Field detection
    3. Player tracking
    4. Ball tracking
    5. Pose estimation (optional, enabled by config)
    6. Action recognition (optional, enabled by config)
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the PadelAnalyzer.
        
        Args:
            config: Configuration object. If None, default config is used.
        """
        self.config = config or Config()
        self.video_loader = VideoLoader(self.config)
        
        # Field detector: prefer keypoint-based if enabled
        if self.config.field_keypoints.enabled:
            self.field_detector = KeypointFieldDetector(self.config)
            logger.info("Using keypoint-based field detection")
        else:
            self.field_detector = FieldDetector(self.config)
            logger.info("Using legacy field detection")
        
        self.player_tracker = PlayerTracker(self.config)
        self.ball_tracker = BallTracker(self.config)
        
        # Initialize pose estimator if enabled
        self.pose_estimator = None
        if self.config.pose.enabled:
            self.pose_estimator = PoseEstimator(self.config)
            logger.info("Pose estimation enabled")
        
        # Initialize action recognizer if enabled
        self.action_recognizer = None
        if self.config.action_recognition.enabled:
            self.action_recognizer = ActionRecognizer(self.config)
            logger.info("Action recognition enabled")
        
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
            - pose_data: Pose estimation data for players (if enabled)
            - actions: Detected player actions/shots (if enabled)
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
            
            # Track ball (pass player tracks for proximity hints)
            ball_tracks = self.ball_tracker.track(video_data, field_info, player_tracks)
            
            # Pose estimation and action recognition
            pose_data = {}
            actions = {}
            
            if self.pose_estimator is not None and self.config.pose.enabled:
                pose_data, actions = self._analyze_poses_and_actions(
                    video_data, player_tracks
                )
            
            return {
                "field_info": field_info,
                "player_tracks": player_tracks,
                "ball_tracks": ball_tracks,
                "pose_data": pose_data,
                "actions": actions,
                "metadata": video_data.get("metadata", {})
            }
        finally:
            # Clean up video resources
            self.video_loader.release(video_data)
    
    def _analyze_poses_and_actions(
        self, 
        video_data: Dict[str, Any], 
        player_tracks: List[Dict[str, Any]]
    ) -> tuple:
        """
        Analyze poses and recognize actions for tracked players.
        
        Args:
            video_data: Video data from VideoLoader
            player_tracks: List of player tracking data
            
        Returns:
            Tuple of (pose_data, actions) dictionaries
        """
        import cv2
        
        pose_data = {}  # player_id -> list of pose data per frame
        actions = {}    # player_id -> list of detected actions
        
        capture = video_data.get("capture")
        if capture is None:
            logger.warning("Video capture not available for pose estimation")
            return pose_data, actions
        
        metadata = video_data.get("metadata", {})
        fps = metadata.get("fps", 30.0)
        frame_count = metadata.get("frame_count", 0)
        
        # Determine frames to process
        sample_rate = self.config.pose.frame_sample_rate
        if self.config.pose.estimate_for_all_frames:
            sample_rate = 1
        
        logger.info(f"Running pose estimation (sample rate: every {sample_rate} frame(s))")
        
        # Create index of player bboxes per frame
        player_bboxes_per_frame = {}
        for track in player_tracks:
            player_id = track["player_id"]
            for i, frame_num in enumerate(track["frame_numbers"]):
                if frame_num not in player_bboxes_per_frame:
                    player_bboxes_per_frame[frame_num] = []
                player_bboxes_per_frame[frame_num].append({
                    "player_id": player_id,
                    "bbox": track["bounding_boxes"][i]
                })
        
        # Initialize pose data storage
        for track in player_tracks:
            pose_data[track["player_id"]] = []
            actions[track["player_id"]] = []
        
        # Reset video to beginning
        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        frame_idx = 0
        processed_frames = 0
        
        while True:
            ret, frame = capture.read()
            if not ret:
                break
            
            # Only process sampled frames
            if frame_idx % sample_rate == 0:
                # Get player bboxes for this frame
                frame_players = player_bboxes_per_frame.get(frame_idx, [])
                
                for player_info in frame_players:
                    player_id = player_info["player_id"]
                    bbox = player_info["bbox"]
                    
                    # Estimate pose for this player
                    pose = self.pose_estimator.estimate_pose_for_player(frame, bbox)
                    
                    if pose is not None:
                        pose_entry = {
                            "frame_number": frame_idx,
                            "keypoints": pose["keypoints"].tolist(),
                            "keypoints_conf": pose["keypoints_conf"].tolist(),
                            "confidence": pose["confidence"]
                        }
                        pose_data[player_id].append(pose_entry)
                        
                        # Recognize action if enabled
                        if self.action_recognizer is not None:
                            action_result = self.action_recognizer.classify_action(
                                pose["keypoints"],
                                pose["keypoints_conf"],
                                player_id=player_id
                            )
                            
                            # Only record significant actions
                            if (action_result["confidence"] >= 
                                self.config.action_recognition.min_action_confidence):
                                action_entry = {
                                    "frame_number": frame_idx,
                                    "action": action_result["action"].value,
                                    "confidence": action_result["confidence"]
                                }
                                actions[player_id].append(action_entry)
                
                processed_frames += 1
                if processed_frames % 50 == 0:
                    logger.info(f"Processed {processed_frames} frames for pose estimation")
            
            frame_idx += 1
        
        logger.info(f"Pose estimation complete. Processed {processed_frames} frames.")
        
        # Summarize actions for each player
        for player_id in actions:
            if actions[player_id]:
                action_summary = self._summarize_actions(actions[player_id])
                actions[player_id] = {
                    "timeline": actions[player_id],
                    "summary": action_summary
                }
            else:
                actions[player_id] = {
                    "timeline": [],
                    "summary": {}
                }
        
        return pose_data, actions
    
    def _summarize_actions(self, action_timeline: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Summarize detected actions for a player.
        
        Args:
            action_timeline: List of action detections over time
            
        Returns:
            Summary statistics of actions
        """
        from collections import Counter
        
        action_counts = Counter(entry["action"] for entry in action_timeline)
        
        return {
            "action_counts": dict(action_counts),
            "total_detections": len(action_timeline),
            "dominant_action": action_counts.most_common(1)[0][0] if action_counts else None
        }
    
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
