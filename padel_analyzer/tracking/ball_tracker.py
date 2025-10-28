"""
Ball tracking module for detecting and tracking the ball in video frames.
"""

from typing import Dict, Any, List, Optional, Tuple
import logging
import numpy as np
import cv2
from scipy.interpolate import interp1d

logger = logging.getLogger(__name__)


class BallTracker:
    """
    Tracks ball movement throughout a padel match video.
    
    Uses computer vision techniques for ball detection and trajectory interpolation.
    """
    
    def __init__(self, config: Any):
        """
        Initialize the BallTracker.
        
        Args:
            config: Configuration object containing tracking settings
        """
        self.config = config
        self.model = None
        self._try_load_model()
        
    def _try_load_model(self):
        """Try to load YOLO model for ball detection (if available)."""
        try:
            from ultralytics import YOLO
            from padel_analyzer.utils.device import get_device
            
            # For now, use YOLO for ball detection too
            # In production, you might want a specialized ball detector
            model_name = self.config.model.ball_model
            
            # If it's the placeholder name, use a YOLO model instead
            if model_name == "custom_ball_detector":
                model_name = "yolov8n.pt"  # Use nano model for speed
            
            logger.info(f"Loading ball detection model: {model_name}")
            self.model = YOLO(model_name)
            
            device = get_device(self.config.model.device)
            self.model.to(device)
            
            logger.info(f"Ball tracker initialized with device: {device}")
            
        except Exception as e:
            logger.warning(f"Could not load ball detection model: {e}")
            logger.info("Falling back to traditional CV methods")
            self.model = None
    
    def track(self, video_data: Dict[str, Any], field_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Track the ball throughout the video.
        
        Args:
            video_data: Video data from VideoLoader
            field_info: Field detection information (used to filter detections)
            
        Returns:
            Dictionary containing ball tracking data:
            - positions: List of (x, y, frame_number) tuples
            - velocities: List of velocity vectors
            - in_play: List of boolean values indicating if ball is in play
            - trajectory: Interpolated ball trajectory
            - confidence_scores: Detection confidence per frame
        """
        logger.info("Starting ball tracking")
        
        capture = video_data.get("capture")
        if capture is None:
            raise ValueError("Video not properly loaded")
        
        metadata = video_data.get("metadata", {})
        frame_count = metadata.get("frame_count", 0)
        fps = metadata.get("fps", 30.0)
        
        # Reset video to beginning
        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # Detect ball in each frame
        raw_detections = []
        
        logger.info(f"Processing {frame_count} frames for ball tracking...")
        
        frame_idx = 0
        while True:
            ret, frame = capture.read()
            if not ret:
                break
            
            # Detect ball in frame
            ball_detection = self.detect_ball_in_frame(
                frame, 
                field_mask=field_info.get("court_mask")
            )
            
            if ball_detection is not None:
                x, y, confidence = ball_detection
                raw_detections.append({
                    "x": x,
                    "y": y,
                    "frame": frame_idx,
                    "confidence": confidence
                })
            
            frame_idx += 1
            
            if frame_idx % 100 == 0:
                logger.info(f"Processed {frame_idx}/{frame_count} frames")
        
        logger.info(f"Detected ball in {len(raw_detections)} frames")
        
        # Process detections into tracks
        if len(raw_detections) == 0:
            logger.warning("No ball detections found")
            return self._empty_ball_tracks()
        
        # Convert to position list
        positions = [(d["x"], d["y"], d["frame"]) for d in raw_detections]
        confidence_scores = [d["confidence"] for d in raw_detections]
        
        # Interpolate missing frames
        trajectory = self.interpolate_trajectory(positions, frame_count)
        
        # Calculate velocities
        velocities = self.calculate_velocity(trajectory, fps)
        
        # Determine when ball is in play (heuristic: when it's moving)
        in_play = self._calculate_in_play(velocities)
        
        ball_tracks = {
            "positions": positions,
            "velocities": velocities,
            "in_play": in_play,
            "trajectory": trajectory,
            "confidence_scores": confidence_scores
        }
        
        logger.info(f"Ball tracking complete: {len(trajectory)} trajectory points")
        
        return ball_tracks
    
    def _empty_ball_tracks(self) -> Dict[str, Any]:
        """Return empty ball tracks structure."""
        return {
            "positions": [],
            "velocities": [],
            "in_play": [],
            "trajectory": [],
            "confidence_scores": []
        }
    
    def detect_ball_in_frame(
        self, 
        frame: np.ndarray, 
        field_mask: Optional[np.ndarray] = None
    ) -> Optional[Tuple[int, int, float]]:
        """
        Detect the ball in a single frame.
        
        Args:
            frame: Video frame to process
            field_mask: Optional field mask to limit search area
            
        Returns:
            Tuple of (x, y, confidence) if ball detected, None otherwise
        """
        # Try model-based detection first
        if self.model is not None:
            return self._detect_ball_with_model(frame, field_mask)
        
        # Fallback to traditional CV methods
        return self._detect_ball_traditional(frame, field_mask)
    
    def _detect_ball_with_model(
        self, 
        frame: np.ndarray, 
        field_mask: Optional[np.ndarray] = None
    ) -> Optional[Tuple[int, int, float]]:
        """
        Detect ball using YOLO model.
        
        Args:
            frame: Video frame
            field_mask: Optional court mask
            
        Returns:
            Ball position and confidence or None
        """
        try:
            # Run detection with sports ball class (class 32 in COCO)
            results = self.model(
                frame,
                classes=[32],  # sports ball
                conf=self.config.tracking.ball_detection_confidence,
                verbose=False
            )
            
            best_detection = None
            best_confidence = 0.0
            
            for result in results:
                boxes = result.boxes
                
                if boxes is None or len(boxes) == 0:
                    continue
                
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0].cpu().numpy())
                    
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    
                    # Filter by court mask
                    if field_mask is not None:
                        h, w = field_mask.shape
                        if 0 <= center_y < h and 0 <= center_x < w:
                            if field_mask[center_y, center_x] == 0:
                                continue
                    
                    # Keep best detection
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_detection = (center_x, center_y, confidence)
            
            return best_detection
            
        except Exception as e:
            logger.warning(f"Error in model-based ball detection: {e}")
            return None
    
    def _detect_ball_traditional(
        self, 
        frame: np.ndarray, 
        field_mask: Optional[np.ndarray] = None
    ) -> Optional[Tuple[int, int, float]]:
        """
        Detect ball using traditional CV methods (circle detection).
        
        Args:
            frame: Video frame
            field_mask: Optional court mask
            
        Returns:
            Ball position and confidence or None
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        
        # Detect circles using Hough Circle Transform
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=20,
            param1=50,
            param2=30,
            minRadius=5,
            maxRadius=30
        )
        
        if circles is None:
            return None
        
        circles = np.uint16(np.around(circles))
        
        # Find best circle (consider brightness and position)
        best_circle = None
        best_score = 0.0
        
        for circle in circles[0, :]:
            x, y, r = circle
            
            # Check if within court mask
            if field_mask is not None:
                h, w = field_mask.shape
                if 0 <= y < h and 0 <= x < w:
                    if field_mask[y, x] == 0:
                        continue
            
            # Score based on brightness (balls are usually bright)
            if 0 <= y < gray.shape[0] and 0 <= x < gray.shape[1]:
                brightness = float(gray[y, x]) / 255.0
                score = brightness * (1.0 / max(r, 1))  # Prefer smaller, brighter circles
                
                if score > best_score:
                    best_score = score
                    best_circle = (int(x), int(y), min(score, 1.0))
        
        return best_circle
    
    def interpolate_trajectory(
        self, 
        positions: List[Tuple[int, int, int]], 
        total_frames: int
    ) -> List[Tuple[int, int, int]]:
        """
        Interpolate ball trajectory for frames where ball wasn't detected.
        
        Args:
            positions: List of detected ball positions (x, y, frame_number)
            total_frames: Total number of frames in video
            
        Returns:
            Complete trajectory with interpolated positions
        """
        if len(positions) < 2:
            return positions
        
        # Extract x, y, frame data
        frames = np.array([p[2] for p in positions])
        x_coords = np.array([p[0] for p in positions])
        y_coords = np.array([p[1] for p in positions])
        
        # Create interpolation functions
        try:
            # Use linear interpolation for simplicity
            f_x = interp1d(frames, x_coords, kind='linear', fill_value='extrapolate')
            f_y = interp1d(frames, y_coords, kind='linear', fill_value='extrapolate')
            
            # Interpolate for all frames within detection range
            min_frame = int(frames.min())
            max_frame = int(frames.max())
            
            interpolated = []
            for frame in range(min_frame, max_frame + 1):
                x = int(f_x(frame))
                y = int(f_y(frame))
                interpolated.append((x, y, frame))
            
            return interpolated
            
        except Exception as e:
            logger.warning(f"Interpolation failed: {e}, using raw positions")
            return positions
    
    def calculate_velocity(
        self, 
        positions: List[Tuple[int, int, int]], 
        fps: float
    ) -> List[Tuple[float, float]]:
        """
        Calculate ball velocity from position data.
        
        Args:
            positions: List of ball positions (x, y, frame_number)
            fps: Video frame rate
            
        Returns:
            List of velocity vectors (vx, vy) in pixels/second
        """
        if len(positions) < 2:
            return []
        
        velocities = []
        
        for i in range(len(positions) - 1):
            x1, y1, f1 = positions[i]
            x2, y2, f2 = positions[i + 1]
            
            # Calculate time difference
            dt = (f2 - f1) / fps if fps > 0 else 1.0
            
            # Calculate velocity
            vx = (x2 - x1) / dt if dt > 0 else 0.0
            vy = (y2 - y1) / dt if dt > 0 else 0.0
            
            velocities.append((vx, vy))
        
        return velocities
    
    def _calculate_in_play(self, velocities: List[Tuple[float, float]]) -> List[bool]:
        """
        Determine when ball is in play based on velocity.
        
        Args:
            velocities: List of velocity vectors
            
        Returns:
            List of boolean values indicating if ball is in play
        """
        # Simple heuristic: ball is in play if moving above threshold
        threshold = 50.0  # pixels/second
        
        in_play = []
        for vx, vy in velocities:
            speed = np.sqrt(vx**2 + vy**2)
            in_play.append(speed > threshold)
        
        return in_play
