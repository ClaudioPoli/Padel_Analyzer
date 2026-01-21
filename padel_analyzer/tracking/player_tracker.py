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
        self.detection_model = None
        self._load_models()
        
    def _load_models(self):
        """Load YOLO model for player detection."""
        try:
            from ultralytics import YOLO
            from padel_analyzer.utils.device import get_device
            
            # Get device
            device = get_device(self.config.model.device)
            
            # Load detection model
            detection_model_name = self.config.model.player_model
            logger.info(f"Loading player detection model: {detection_model_name}")
            self.detection_model = YOLO(detection_model_name)
            self.detection_model.to(device)
            logger.info(f"Player tracker initialized with device: {device}")
            
        except ImportError:
            logger.warning(
                "Ultralytics YOLO not installed. Install with: pip install ultralytics"
            )
            self.detection_model = None
        except Exception as e:
            logger.error(f"Failed to load player tracking model: {e}")
            self.detection_model = None
    
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
        
        if self.detection_model is None:
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
        Detect players in a single frame using hybrid approach:
        1. Detection model for bounding boxes
        2. Pose model for keypoints (if enabled)
        
        Args:
            frame: Video frame to process
            frame_number: Frame index in video
            field_mask: Optional court mask to filter detections
            
        Returns:
            List of detected players with bounding boxes, keypoints, and confidence scores
        """
        if self.detection_model is None:
            return []
        
        try:
            # Run YOLO detection with lower confidence for better recall
            min_conf = min(0.2, self.config.tracking.player_detection_confidence)
            
            results = self.detection_model.track(
                frame, 
                persist=True,
                classes=[0],  # 0 = person in COCO dataset
                conf=min_conf,
                verbose=False,
                iou=0.4,
                max_det=15,
                tracker="bytetrack.yaml"
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
                    
                    # More lenient filtering for field mask
                    # Only filter if clearly outside court (e.g., spectators in stands)
                    # For padel, players can be near walls/glass, so be very permissive
                    if field_mask is not None:
                        h, w = field_mask.shape
                        # Allow very generous margin - 50px for players near boundaries
                        margin = 50
                        
                        # Check if feet position is within expanded court area
                        if 0 <= bottom_center_y < h and 0 <= bottom_center_x < w:
                            # Check surrounding area before filtering
                            y_min = max(0, bottom_center_y - margin)
                            y_max = min(h, bottom_center_y + margin)
                            x_min = max(0, bottom_center_x - margin)
                            x_max = min(w, bottom_center_x + margin)
                            
                            # Only filter if NO part of the margin area is on court
                            # This is very permissive to avoid losing players
                            if np.sum(field_mask[y_min:y_max, x_min:x_max]) == 0:
                                # Additional check: if confidence is high, keep it anyway
                                # (might be a player outside court temporarily)
                                if confidence < 0.6:
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
            "frame_numbers": [],
            "keypoints_sequence": [],  # Store keypoints over time
            "keypoints_conf_sequence": []
        })
        
        for detection in all_detections:
            track_id = detection.get("track_id")
            if track_id is None:
                continue
            
            tracks_dict[track_id]["positions"].append(detection["center"])
            tracks_dict[track_id]["bounding_boxes"].append(detection["bbox"])
            tracks_dict[track_id]["confidence_scores"].append(detection["confidence"])
            tracks_dict[track_id]["frame_numbers"].append(detection["frame_number"])
            tracks_dict[track_id]["keypoints_sequence"].append(detection.get("keypoints"))
            tracks_dict[track_id]["keypoints_conf_sequence"].append(detection.get("keypoints_conf"))
        
        # Convert to list format and filter short tracks
        min_track_length = 3  # Minimum frames to be considered a valid track
        all_tracks = []
        
        for track_id, track_data in tracks_dict.items():
            if len(track_data["positions"]) < min_track_length:
                continue
            
            all_tracks.append({
                "player_id": track_id,
                "positions": track_data["positions"],
                "bounding_boxes": track_data["bounding_boxes"],
                "confidence_scores": track_data["confidence_scores"],
                "frame_numbers": track_data["frame_numbers"],
                "team": None  # To be assigned later
            })
        
        # For padel doubles, we expect 4 players
        # Keep the 4 tracks with highest coverage/consistency
        # Sort by number of frames (coverage) and keep top 4
        if len(all_tracks) > 4:
            # Score tracks by coverage and average confidence
            scored_tracks = []
            for track in all_tracks:
                coverage_score = len(track["positions"])
                avg_conf = np.mean(track["confidence_scores"]) if track["confidence_scores"] else 0
                # Combined score: prioritize coverage, but consider confidence
                score = coverage_score * 0.8 + avg_conf * coverage_score * 0.2
                scored_tracks.append((track, score))
            
            # Sort by score and keep top 4
            scored_tracks.sort(key=lambda x: x[1], reverse=True)
            player_tracks = [track for track, score in scored_tracks[:4]]
            
            logger.info(f"Filtered {len(all_tracks)} tracks down to top 4 players")
        else:
            player_tracks = all_tracks
        
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
        Assign players to teams based on court position and optionally shirt color from keypoints.
        Hybrid approach:
        1. Primary: Court position (net divides teams)
        2. Secondary (if keypoints available): Shirt color clustering
        
        Args:
            player_tracks: List of player tracking data
            field_info: Field information for court context
            
        Returns:
            Updated player tracks with team assignments
        """
        # Try intelligent team assignment with keypoints first
        if self.config.tracking.use_keypoints_for_team:
            team_assigned = self._assign_teams_by_color(player_tracks)
            if team_assigned:
                logger.info("Team assignment: Using shirt color from keypoints")
                return player_tracks
        
        # Fallback: Position-based team assignment
        logger.info("Team assignment: Using court position (fallback)")
        return self._assign_teams_by_position(player_tracks, field_info)
    
    def _assign_teams_by_position(
        self, 
        player_tracks: List[Dict[str, Any]], 
        field_info: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Assign teams based on Y-position (net divides court horizontally).
        
        Args:
            player_tracks: List of player tracking data
            field_info: Field information for court context
            
        Returns:
            Updated player tracks with team assignments
        """
        # Calculate average y-position for each player to determine which side of net
        player_avg_positions = []
        for track in player_tracks:
            positions = track["positions"]
            if len(positions) == 0:
                continue
            
            # Use median y-position (more robust to outliers)
            avg_y = np.median([pos[1] for pos in positions])
            avg_x = np.median([pos[0] for pos in positions])
            player_avg_positions.append((track["player_id"], avg_y, avg_x))
        
        if len(player_avg_positions) < 2:
            return player_tracks
        
        # Sort by y-position (vertical position on court)
        player_avg_positions.sort(key=lambda x: x[1])
        
        # For padel: net divides court horizontally
        # Split players into two groups based on y-position (top/bottom of court)
        # Each group is a team (2 players per team for standard padel doubles)
        
        if len(player_avg_positions) == 2:
            # Only 2 players detected - assign to different teams
            team_a_ids = [player_avg_positions[0][0]]
            team_b_ids = [player_avg_positions[1][0]]
        elif len(player_avg_positions) == 3:
            # 3 players - assign based on gaps in y-position
            # Find the largest gap to determine net position
            gaps = []
            for i in range(len(player_avg_positions) - 1):
                gap = player_avg_positions[i+1][1] - player_avg_positions[i][1]
                gaps.append((i, gap))
            
            # Split at largest gap (likely the net)
            max_gap_idx = max(gaps, key=lambda x: x[1])[0]
            team_a_ids = [pid for pid, _, _ in player_avg_positions[:max_gap_idx+1]]
            team_b_ids = [pid for pid, _, _ in player_avg_positions[max_gap_idx+1:]]
        else:
            # 4 or more players
            # For 4 players: split in half (2 per team)
            # For more: try to identify the 4 main players and assign others
            mid_point = len(player_avg_positions) // 2
            
            # Calculate y-position gaps to find natural split (the net)
            if len(player_avg_positions) >= 4:
                # Find largest gap in y-positions - this should be the net
                gaps = []
                for i in range(len(player_avg_positions) - 1):
                    gap = player_avg_positions[i+1][1] - player_avg_positions[i][1]
                    gaps.append((i, gap))
                
                # If there's a clear gap, use it to split teams
                max_gap = max(gaps, key=lambda x: x[1])
                if max_gap[1] > 50:  # Significant gap (likely the net)
                    mid_point = max_gap[0] + 1
            
            team_a_ids = [pid for pid, _, _ in player_avg_positions[:mid_point]]
            team_b_ids = [pid for pid, _, _ in player_avg_positions[mid_point:]]
        
        # Update tracks with team assignments
        for track in player_tracks:
            if track["player_id"] in team_a_ids:
                track["team"] = "A"
            elif track["player_id"] in team_b_ids:
                track["team"] = "B"
        
        logger.info(f"Team A: {len(team_a_ids)} players {team_a_ids}, Team B: {len(team_b_ids)} players {team_b_ids}")
        
        return player_tracks
    
    def _assign_teams_by_color(self, player_tracks: List[Dict[str, Any]]) -> bool:
        """
        Assign teams based on shirt color extracted from torso keypoints.
        Uses k-means clustering to identify two dominant colors.
        
        Args:
            player_tracks: List of player tracking data with keypoints
            
        Returns:
            True if successful, False if not enough data
        """
        # This is a placeholder for future implementation
        # Requires:
        # 1. Extract torso region using shoulder/hip keypoints
        # 2. Sample dominant color from torso
        # 3. Cluster players into 2 groups by color
        # 4. Assign teams based on clusters
        
        logger.debug("Color-based team assignment not yet implemented")
        return False


# Import cv2 here to avoid circular imports at module level
try:
    import cv2
except ImportError:
    logger.warning("OpenCV not installed")
