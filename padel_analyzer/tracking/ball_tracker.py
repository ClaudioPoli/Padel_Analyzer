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
    
    def track(self, video_data: Dict[str, Any], field_info: Dict[str, Any], player_tracks: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Track the ball throughout the video.
        
        Args:
            video_data: Video data from VideoLoader
            field_info: Field detection information (used to filter detections)
            player_tracks: Optional player tracking data to use for ball detection hints
            
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
        
        # Build player position lookup for proximity hints
        self.player_positions_by_frame = {}
        if player_tracks:
            for player in player_tracks:
                for i, frame_num in enumerate(player.get('frame_numbers', [])):
                    if frame_num not in self.player_positions_by_frame:
                        self.player_positions_by_frame[frame_num] = []
                    if i < len(player.get('positions', [])):
                        self.player_positions_by_frame[frame_num].append(player['positions'][i])
        
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
            
            # Detect ball in frame with temporal and spatial hints
            ball_detection = self.detect_ball_in_frame(
                frame, 
                frame_idx,
                field_mask=field_info.get("court_mask"),
                prev_ball_pos=raw_detections[-1] if raw_detections else None
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
        frame_idx: int,
        field_mask: Optional[np.ndarray] = None,
        prev_ball_pos: Optional[Dict[str, Any]] = None
    ) -> Optional[Tuple[int, int, float]]:
        """
        Detect the ball in a single frame using multiple strategies.
        
        Args:
            frame: Video frame to process
            frame_idx: Current frame index
            field_mask: Optional field mask to limit search area
            prev_ball_pos: Previous ball position for temporal tracking
            
        Returns:
            Tuple of (x, y, confidence) if ball detected, None otherwise
        """
        detections = []
        
        # Get player positions for this frame (proximity hint)
        player_positions = self.player_positions_by_frame.get(frame_idx, [])
        
        # Try model-based detection first
        if self.model is not None:
            model_detection = self._detect_ball_with_model(frame, field_mask, player_positions)
            if model_detection is not None:
                detections.append(model_detection)
        
        # Also try traditional CV methods
        traditional_detection = self._detect_ball_traditional(frame, field_mask, player_positions)
        if traditional_detection is not None:
            detections.append(traditional_detection)
        
        # Try color-based detection for yellow balls (most important for small balls)
        color_detection = self._detect_ball_by_color(frame, field_mask, player_positions)
        if color_detection is not None:
            detections.append(color_detection)
        
        # If we have previous ball position, try motion-based detection
        if prev_ball_pos is not None:
            motion_detection = self._detect_ball_by_motion(frame, prev_ball_pos, field_mask)
            if motion_detection is not None:
                detections.append(motion_detection)
        
        # Return the detection with highest confidence, boosting if near players
        if len(detections) > 0:
            # Boost confidence for detections near players
            boosted_detections = []
            for x, y, conf in detections:
                boost = 1.0
                # Check if near any player (within 300 pixels)
                for px, py in player_positions:
                    dist = np.sqrt((x - px)**2 + (y - py)**2)
                    if dist < 300:
                        boost = 1.2  # 20% confidence boost for being near player
                        break
                boosted_detections.append((x, y, min(conf * boost, 1.0)))
            
            return max(boosted_detections, key=lambda d: d[2])
        
        return None
    
    def _detect_ball_with_model(
        self, 
        frame: np.ndarray, 
        field_mask: Optional[np.ndarray] = None,
        player_positions: Optional[List[Tuple[int, int]]] = None
    ) -> Optional[Tuple[int, int, float]]:
        """
        Detect ball using YOLO model.
        
        Args:
            frame: Video frame
            field_mask: Optional court mask
            player_positions: Optional list of player positions for proximity boost
            
        Returns:
            Ball position and confidence or None
        """
        try:
            # Use very low confidence threshold to catch more balls
            # Small fast balls often have lower detection confidence
            min_conf = min(0.10, self.config.tracking.ball_detection_confidence)  # Lowered from 0.15
            
            # Run detection with sports ball class (class 32 in COCO)
            results = self.model(
                frame,
                classes=[32],  # sports ball
                conf=min_conf,
                verbose=False,
                iou=0.3,  # Lower IoU for overlapping detections
                max_det=5  # Allow multiple ball candidates
            )
            
            candidates = []
            
            for result in results:
                boxes = result.boxes
                
                if boxes is None or len(boxes) == 0:
                    continue
                
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0].cpu().numpy())
                    
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    
                    # Calculate size
                    width = x2 - x1
                    height = y2 - y1
                    area = width * height
                    
                    # Filter by court mask
                    if field_mask is not None:
                        h, w = field_mask.shape
                        if 0 <= center_y < h and 0 <= center_x < w:
                            if field_mask[center_y, center_x] == 0:
                                continue
                    
                    # Prefer smaller detections (balls are small)
                    # Adjust confidence based on size
                    size_factor = 1.0 / (1.0 + area / 500.0)
                    adjusted_conf = confidence * (0.5 + 0.5 * size_factor)
                    
                    candidates.append((center_x, center_y, adjusted_conf))
            
            # Return best candidate
            if len(candidates) > 0:
                return max(candidates, key=lambda c: c[2])
            
            return None
            
        except Exception as e:
            logger.warning(f"Error in model-based ball detection: {e}")
            return None
    
    def _detect_ball_traditional(
        self, 
        frame: np.ndarray, 
        field_mask: Optional[np.ndarray] = None,
        player_positions: Optional[List[Tuple[int, int]]] = None
    ) -> Optional[Tuple[int, int, float]]:
        """
        Detect ball using traditional CV methods (circle detection).
        
        Args:
            frame: Video frame
            field_mask: Optional court mask
            player_positions: Optional list of player positions
            
        Returns:
            Ball position and confidence or None
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        
        # Try multiple parameter sets for better detection
        circles_list = []
        
        # Parameter set 1: More sensitive for small balls
        circles1 = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=10,  # Reduced from 15 for small balls
            param1=40,
            param2=18,  # More sensitive (reduced from 20)
            minRadius=2,  # Reduced from 3 for very small balls
            maxRadius=20  # Reduced from 25 to focus on small objects
        )
        if circles1 is not None:
            circles_list.extend(circles1[0])
        
        # Parameter set 2: Less sensitive
        circles2 = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=20,
            param1=50,
            param2=30,
            minRadius=5,
            maxRadius=30
        )
        if circles2 is not None:
            circles_list.extend(circles2[0])
        
        if len(circles_list) == 0:
            return None
        
        circles_list = np.array(circles_list)
        
        # Find best circle based on multiple factors
        best_circle = None
        best_score = 0.0
        
        for circle in circles_list:
            x, y, r = circle
            x, y, r = int(x), int(y), int(r)
            
            # Check if within frame bounds
            if x < r or y < r or x >= frame.shape[1] - r or y >= frame.shape[0] - r:
                continue
            
            # Check if within court mask
            if field_mask is not None:
                h, w = field_mask.shape
                if 0 <= y < h and 0 <= x < w:
                    if field_mask[y, x] == 0:
                        continue
            
            # Calculate score based on brightness, size, and color
            if 0 <= y < gray.shape[0] and 0 <= x < gray.shape[1]:
                # Get region of interest
                roi_y1, roi_y2 = max(0, y-r), min(gray.shape[0], y+r)
                roi_x1, roi_x2 = max(0, x-r), min(gray.shape[1], x+r)
                
                # Brightness score
                brightness = float(gray[y, x]) / 255.0
                
                # Check if it's a bright object (balls are usually bright)
                if brightness < 0.3:
                    continue
                
                # Size score (prefer smaller circles for ball)
                size_score = 1.0 / (1.0 + r / 10.0)
                
                # Combined score
                score = brightness * size_score
                
                if score > best_score:
                    best_score = score
                    best_circle = (x, y, min(score, 1.0))
        
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
    
    def _detect_ball_by_color(
        self, 
        frame: np.ndarray, 
        field_mask: Optional[np.ndarray] = None,
        player_positions: Optional[List[Tuple[int, int]]] = None
    ) -> Optional[Tuple[int, int, float]]:
        """
        Detect ball using color-based segmentation.
        Looks for yellow/green (typical padel ball colors) or bright white objects.
        
        Args:
            frame: Video frame
            field_mask: Optional court mask
            player_positions: Optional list of player positions
            
        Returns:
            Ball position and confidence or None
        """
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define color ranges for typical padel balls - broadened for better detection
        # Yellow-green ball (most common)
        lower_yellow = np.array([18, 60, 60])  # Broader range
        upper_yellow = np.array([48, 255, 255])
        
        # Bright yellow
        lower_bright_yellow = np.array([12, 80, 120])  # More lenient
        upper_bright_yellow = np.array([38, 255, 255])
        
        # White/bright ball (alternative)
        lower_white = np.array([0, 0, 180])  # More lenient
        upper_white = np.array([180, 40, 255])
        
        # Create masks for each color range
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        mask_bright = cv2.inRange(hsv, lower_bright_yellow, upper_bright_yellow)
        mask_white = cv2.inRange(hsv, lower_white, upper_white)
        
        # Combine masks
        combined_mask = cv2.bitwise_or(mask_yellow, mask_bright)
        combined_mask = cv2.bitwise_or(combined_mask, mask_white)
        
        # Apply field mask if provided
        if field_mask is not None:
            combined_mask = cv2.bitwise_and(combined_mask, field_mask)
        
        # Apply morphological operations to reduce noise (smaller kernel for small balls)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))  # Reduced from (3,3)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            return None
        
        # Find best contour (circular, small, bright)
        best_ball = None
        best_score = 0.0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by size (ball should be small but visible) - lowered minimum
            if area < 10 or area > 1500:  # Reduced from 20 min and 2000 max
                continue
            
            # Calculate circularity
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            # Ball should be reasonably circular - more lenient
            if circularity < 0.35:  # Reduced from 0.4
                continue
            
            # Get centroid
            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
            
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            # Check bounds
            if cx < 0 or cy < 0 or cx >= frame.shape[1] or cy >= frame.shape[0]:
                continue
            
            # Calculate score based on circularity and size
            # Prefer smaller, more circular objects
            size_score = 1.0 / (1.0 + area / 100.0)
            score = circularity * size_score
            
            if score > best_score:
                best_score = score
                best_ball = (cx, cy, min(score, 1.0))
        
        return best_ball
    
    def _detect_ball_by_motion(
        self,
        frame: np.ndarray,
        prev_ball_pos: Dict[str, Any],
        field_mask: Optional[np.ndarray] = None
    ) -> Optional[Tuple[int, int, float]]:
        """
        Detect ball using motion and previous position.
        Useful for fast-moving balls that might be blurred.
        
        Args:
            frame: Current video frame
            prev_ball_pos: Previous ball detection dict with 'x', 'y', 'frame'
            field_mask: Optional court mask
            
        Returns:
            Ball position and confidence or None
        """
        prev_x = prev_ball_pos.get('x')
        prev_y = prev_ball_pos.get('y')
        
        if prev_x is None or prev_y is None:
            return None
        
        # Define search region around previous position
        # Fast balls can move up to 100 pixels per frame
        search_radius = 100
        
        # Crop region of interest
        h, w = frame.shape[:2]
        x1 = max(0, prev_x - search_radius)
        y1 = max(0, prev_y - search_radius)
        x2 = min(w, prev_x + search_radius)
        y2 = min(h, prev_y + search_radius)
        
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return None
        
        # Convert to HSV
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Look for yellow ball in ROI
        lower_yellow = np.array([18, 60, 60])
        upper_yellow = np.array([48, 255, 255])
        mask = cv2.inRange(hsv_roi, lower_yellow, upper_yellow)
        
        # Also look for bright objects (ball might be overexposed)
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, bright_mask = cv2.threshold(gray_roi, 200, 255, cv2.THRESH_BINARY)
        
        # Combine masks
        combined = cv2.bitwise_or(mask, bright_mask)
        
        # Find contours
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            return None
        
        # Find contour closest to previous position
        best_ball = None
        min_dist = float('inf')
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 8 or area > 1500:
                continue
            
            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
            
            # Get centroid in ROI coordinates
            cx_roi = int(M["m10"] / M["m00"])
            cy_roi = int(M["m01"] / M["m00"])
            
            # Convert to frame coordinates
            cx = x1 + cx_roi
            cy = y1 + cy_roi
            
            # Distance from previous position
            dist = np.sqrt((cx - prev_x)**2 + (cy - prev_y)**2)
            
            if dist < min_dist:
                min_dist = dist
                # Confidence based on proximity to previous position
                confidence = max(0.3, 1.0 - (dist / search_radius))
                best_ball = (cx, cy, confidence)
        
        return best_ball
