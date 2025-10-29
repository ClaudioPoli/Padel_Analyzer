"""
Player tracking module for detecting and tracking players in video frames.
"""

from typing import Dict, Any, List, Optional, Tuple
import logging
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)


class PlayerTracker:
    """
    Tracks player movements throughout a padel match video.
    
    Uses YOLO for person detection and tracking across frames.
    """
    
    def __init__(self, config: Any):
        """
        Initialize the PlayerTracker.
        
        Args:
            config: Configuration object containing tracking settings
        """
        self.config = config
        self.model = None
        self._load_model()
        
    def _load_model(self):
        """Load YOLO model for person detection."""
        try:
            from ultralytics import YOLO
            from padel_analyzer.utils.device import get_device
            
            model_name = self.config.model.player_model
            logger.info(f"Loading player detection model: {model_name}")
            
            # Load YOLO model
            self.model = YOLO(model_name)
            
            # Set device
            device = get_device(self.config.model.device)
            self.model.to(device)
            
            logger.info(f"Player tracker initialized with device: {device}")
            
        except ImportError:
            logger.warning(
                "Ultralytics YOLO not installed. Install with: pip install ultralytics"
            )
            self.model = None
        except Exception as e:
            logger.error(f"Failed to load player tracking model: {e}")
            self.model = None
    
    def track(self, video_data: Dict[str, Any], field_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Track players throughout the video.
        
        Args:
            video_data: Video data from VideoLoader
            field_info: Field detection information (used to filter detections)
            
        Returns:
            List of player tracking data:
            - player_id: Unique identifier for each player
            - positions: List of (x, y, frame_number) tuples
            - bounding_boxes: List of bounding boxes for each frame
            - team: Team assignment (if applicable)
            - confidence_scores: Detection confidence per frame
        """
        logger.info("Starting player tracking")
        
        if self.model is None:
            logger.warning("Player tracking model not loaded, returning empty tracks")
            return []
        
        capture = video_data.get("capture")
        if capture is None:
            raise ValueError("Video not properly loaded")
        
        metadata = video_data.get("metadata", {})
        frame_count = metadata.get("frame_count", 0)
        
        # Reset video to beginning
        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # Track players frame by frame
        all_detections = []
        
        logger.info(f"Processing {frame_count} frames for player tracking...")
        
        frame_idx = 0
        while True:
            ret, frame = capture.read()
            if not ret:
                break
            
            # Detect players in frame
            detections = self.detect_players_in_frame(
                frame, 
                frame_idx, 
                field_mask=field_info.get("court_mask")
            )
            all_detections.extend(detections)
            
            frame_idx += 1
            
            # Log progress
            if frame_idx % 100 == 0:
                logger.info(f"Processed {frame_idx}/{frame_count} frames")
        
        logger.info(f"Total detections: {len(all_detections)}")
        
        # Associate detections into tracks
        player_tracks = self._associate_tracks(all_detections, field_info)
        
        logger.info(f"Created {len(player_tracks)} player tracks")
        
        return player_tracks
    
    def detect_players_in_frame(
        self, 
        frame: np.ndarray, 
        frame_number: int,
        field_mask: Optional[np.ndarray] = None
    ) -> List[Dict[str, Any]]:
        """
        Detect players in a single frame using YOLO.
        
        Args:
            frame: Video frame to process
            frame_number: Frame index in video
            field_mask: Optional court mask to filter detections
            
        Returns:
            List of detected players with bounding boxes and confidence scores
        """
        if self.model is None:
            return []
        
        try:
            # Run YOLO detection with lower confidence for better recall
            # Use a lower threshold than config to catch more players initially
            min_conf = min(0.25, self.config.tracking.player_detection_confidence)
            
            results = self.model.track(
                frame, 
                persist=True,
                classes=[0],  # 0 = person in COCO dataset
                conf=min_conf,
                verbose=False,
                iou=0.5,  # IoU threshold for NMS
                max_det=10  # Maximum detections per frame
            )
            
            detections = []
            
            for result in results:
                boxes = result.boxes
                
                if boxes is None or len(boxes) == 0:
                    continue
                
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0].cpu().numpy())
                    
                    # Get track ID if available
                    track_id = int(box.id[0].cpu().numpy()) if box.id is not None else None
                    
                    # Calculate center point
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    
                    # Calculate bottom center (feet position) which is more reliable for court filtering
                    bottom_center_x = int((x1 + x2) / 2)
                    bottom_center_y = int(y2)  # Bottom of bounding box
                    
                    # Filter by court mask if provided - use bottom center (feet) instead of bbox center
                    # Also expand the mask slightly to avoid cutting off players at boundaries
                    if field_mask is not None:
                        h, w = field_mask.shape
                        # Check if feet position is within expanded court area
                        if 0 <= bottom_center_y < h and 0 <= bottom_center_x < w:
                            # Don't filter out if feet are on or near the court
                            # Only filter if clearly outside (allowing 20px margin for error)
                            margin = 20
                            if field_mask[bottom_center_y, bottom_center_x] == 0:
                                # Check surrounding area (margin) before filtering
                                y_min = max(0, bottom_center_y - margin)
                                y_max = min(h, bottom_center_y + margin)
                                x_min = max(0, bottom_center_x - margin)
                                x_max = min(w, bottom_center_x + margin)
                                
                                # If no part of the margin area is on court, skip
                                if np.sum(field_mask[y_min:y_max, x_min:x_max]) == 0:
                                    continue
                    
                    detections.append({
                        "frame_number": frame_number,
                        "bbox": [int(x1), int(y1), int(x2), int(y2)],
                        "center": (center_x, center_y),
                        "confidence": confidence,
                        "track_id": track_id
                    })
            
            return detections
            
        except Exception as e:
            logger.warning(f"Error detecting players in frame {frame_number}: {e}")
            return []
    
    def _associate_tracks(
        self, 
        all_detections: List[Dict[str, Any]], 
        field_info: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Associate detections into continuous player tracks.
        
        Args:
            all_detections: All detections from all frames
            field_info: Field information for context
            
        Returns:
            List of player tracks
        """
        # Group detections by track_id (YOLO's built-in tracking)
        tracks_dict = defaultdict(lambda: {
            "positions": [],
            "bounding_boxes": [],
            "confidence_scores": [],
            "frame_numbers": []
        })
        
        for detection in all_detections:
            track_id = detection.get("track_id")
            if track_id is None:
                continue
            
            tracks_dict[track_id]["positions"].append(detection["center"])
            tracks_dict[track_id]["bounding_boxes"].append(detection["bbox"])
            tracks_dict[track_id]["confidence_scores"].append(detection["confidence"])
            tracks_dict[track_id]["frame_numbers"].append(detection["frame_number"])
        
        # Convert to list format and filter short tracks
        # Use a shorter minimum to catch players who appear briefly
        min_track_length = 5  # Reduced from 10 to catch more players
        player_tracks = []
        
        for track_id, track_data in tracks_dict.items():
            if len(track_data["positions"]) < min_track_length:
                continue
            
            player_tracks.append({
                "player_id": track_id,
                "positions": track_data["positions"],
                "bounding_boxes": track_data["bounding_boxes"],
                "confidence_scores": track_data["confidence_scores"],
                "frame_numbers": track_data["frame_numbers"],
                "team": None  # To be assigned later
            })
        
        # Assign teams based on court position
        if len(player_tracks) >= 2:
            player_tracks = self.assign_teams(player_tracks, field_info)
        
        return player_tracks
    
    def assign_teams(
        self, 
        player_tracks: List[Dict[str, Any]], 
        field_info: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Assign players to teams based on their positions on the court.
        
        Args:
            player_tracks: List of player tracking data
            field_info: Field information for court context
            
        Returns:
            Updated player tracks with team assignments
        """
        # Simple heuristic: divide court in half and assign teams
        # Calculate average y-position for each player
        
        player_avg_positions = []
        for track in player_tracks:
            positions = track["positions"]
            if len(positions) == 0:
                continue
            
            avg_y = np.mean([pos[1] for pos in positions])
            player_avg_positions.append((track["player_id"], avg_y))
        
        if len(player_avg_positions) < 2:
            return player_tracks
        
        # Sort by y-position
        player_avg_positions.sort(key=lambda x: x[1])
        
        # Assign teams (top half vs bottom half)
        mid_point = len(player_avg_positions) // 2
        team_a_ids = [pid for pid, _ in player_avg_positions[:mid_point]]
        team_b_ids = [pid for pid, _ in player_avg_positions[mid_point:]]
        
        # Update tracks with team assignments
        for track in player_tracks:
            if track["player_id"] in team_a_ids:
                track["team"] = "A"
            elif track["player_id"] in team_b_ids:
                track["team"] = "B"
        
        logger.info(f"Team A: {len(team_a_ids)} players, Team B: {len(team_b_ids)} players")
        
        return player_tracks


# Import cv2 here to avoid circular imports at module level
try:
    import cv2
except ImportError:
    logger.warning("OpenCV not installed")
