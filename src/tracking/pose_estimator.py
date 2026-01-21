"""
Pose estimation module for detecting player body keypoints.

Uses YOLOv8-Pose for zero-shot pose estimation without fine-tuning.
This approach is recommended for padel analysis because:
1. YOLOv8-Pose is pre-trained on COCO which includes diverse human poses
2. Padel player poses (serving, volleying, smashing) are similar to general sports poses
3. Zero-shot inference is sufficient for action classification based on keypoint geometry
4. Fine-tuning would require a large annotated padel-specific dataset

For more specialized shot recognition (e.g., distinguishing forehand topspin from slice),
fine-tuning on a padel-specific dataset would be beneficial. Available datasets:
- Custom annotation using CVAT/LabelStudio on padel match videos
- Transfer learning from tennis pose datasets (similar biomechanics)
"""

from typing import Dict, Any, List, Optional, Tuple
import logging
import numpy as np

logger = logging.getLogger(__name__)

# COCO Keypoint indices for YOLOv8-Pose
# https://docs.ultralytics.com/tasks/pose/
KEYPOINT_NAMES = [
    "nose",           # 0
    "left_eye",       # 1
    "right_eye",      # 2
    "left_ear",       # 3
    "right_ear",      # 4
    "left_shoulder",  # 5
    "right_shoulder", # 6
    "left_elbow",     # 7
    "right_elbow",    # 8
    "left_wrist",     # 9
    "right_wrist",    # 10
    "left_hip",       # 11
    "right_hip",      # 12
    "left_knee",      # 13
    "right_knee",     # 14
    "left_ankle",     # 15
    "right_ankle",    # 16
]

# Keypoint connections for skeleton visualization
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2),    # Nose to eyes
    (1, 3), (2, 4),    # Eyes to ears
    (5, 6),            # Shoulders
    (5, 7), (7, 9),    # Left arm
    (6, 8), (8, 10),   # Right arm
    (5, 11), (6, 12),  # Torso
    (11, 12),          # Hips
    (11, 13), (13, 15),  # Left leg
    (12, 14), (14, 16),  # Right leg
]


class PoseEstimator:
    """
    Estimates human body pose keypoints for detected players.
    
    Uses YOLOv8-Pose model for zero-shot pose estimation.
    The model outputs 17 COCO keypoints per detected person.
    """
    
    def __init__(self, config: Any):
        """
        Initialize the PoseEstimator.
        
        Args:
            config: Configuration object containing pose estimation settings
        """
        self.config = config
        self.pose_model = None
        self._load_model()
    
    def _load_model(self):
        """Load YOLOv8-Pose model for pose estimation."""
        try:
            from ultralytics import YOLO
            from src.utils.device import get_device
            
            # Get device
            device = get_device(self.config.model.device)
            
            # Load pose model
            pose_model_name = getattr(self.config.pose, 'pose_model', 'yolov8n-pose.pt')
            logger.info(f"Loading pose estimation model: {pose_model_name}")
            
            self.pose_model = YOLO(pose_model_name)
            self.pose_model.to(device)
            self.device = device
            
            logger.info(f"Pose estimator initialized with device: {device}")
            
        except ImportError:
            logger.warning(
                "Ultralytics YOLO not installed. Install with: pip install ultralytics"
            )
            self.pose_model = None
        except Exception as e:
            logger.error(f"Failed to load pose estimation model: {e}")
            self.pose_model = None
    
    def estimate_pose(
        self, 
        frame: np.ndarray, 
        bboxes: Optional[List[List[int]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Estimate poses for all persons in a frame.
        
        Args:
            frame: Video frame (BGR format from OpenCV)
            bboxes: Optional list of bounding boxes to focus on specific regions.
                    Format: [[x1, y1, x2, y2], ...]
                    If provided, only estimate poses within these boxes.
            
        Returns:
            List of pose estimations, each containing:
            - keypoints: np.ndarray of shape (17, 2) with (x, y) coordinates
            - keypoints_conf: np.ndarray of shape (17,) with confidence scores
            - bbox: [x1, y1, x2, y2] bounding box
            - confidence: Overall detection confidence
        """
        if self.pose_model is None:
            logger.warning("Pose model not loaded, returning empty results")
            return []
        
        try:
            # Get confidence threshold from config
            min_conf = getattr(self.config.pose, 'min_confidence', 0.25)
            
            # Run YOLOv8-Pose inference
            results = self.pose_model(
                frame,
                conf=min_conf,
                verbose=False
            )
            
            poses = []
            
            for result in results:
                if result.keypoints is None:
                    continue
                
                keypoints_data = result.keypoints
                boxes = result.boxes
                
                if boxes is None or len(boxes) == 0:
                    continue
                
                # Process each detection
                for i in range(len(boxes)):
                    # Get bounding box
                    box = boxes[i]
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0].cpu().numpy())
                    
                    # If bboxes filter is provided, check if this detection overlaps
                    if bboxes is not None:
                        if not self._bbox_overlaps_any(
                            [x1, y1, x2, y2], 
                            bboxes, 
                            iou_threshold=0.3
                        ):
                            continue
                    
                    # Get keypoints for this detection
                    if i < len(keypoints_data):
                        kpts = keypoints_data[i]
                        
                        # Extract xy coordinates and confidence
                        if hasattr(kpts, 'xy'):
                            kpts_xy = kpts.xy[0].cpu().numpy()  # Shape: (17, 2)
                        else:
                            kpts_xy = kpts.data[0, :, :2].cpu().numpy()
                        
                        if hasattr(kpts, 'conf'):
                            kpts_conf = kpts.conf[0].cpu().numpy()  # Shape: (17,)
                        else:
                            kpts_conf = kpts.data[0, :, 2].cpu().numpy()
                        
                        poses.append({
                            "keypoints": kpts_xy,
                            "keypoints_conf": kpts_conf,
                            "bbox": [int(x1), int(y1), int(x2), int(y2)],
                            "confidence": confidence
                        })
            
            return poses
            
        except Exception as e:
            logger.warning(f"Error estimating poses: {e}")
            return []
    
    def estimate_pose_for_player(
        self,
        frame: np.ndarray,
        bbox: List[int],
        padding: int = 20
    ) -> Optional[Dict[str, Any]]:
        """
        Estimate pose for a single player given their bounding box.
        
        This crops the region around the player for more focused pose estimation.
        
        Args:
            frame: Full video frame
            bbox: Bounding box [x1, y1, x2, y2] of the player
            padding: Pixels to add around the bounding box
            
        Returns:
            Pose estimation dict or None if no pose detected
        """
        if self.pose_model is None:
            return None
        
        try:
            x1, y1, x2, y2 = bbox
            h, w = frame.shape[:2]
            
            # Add padding with bounds checking
            x1_pad = max(0, x1 - padding)
            y1_pad = max(0, y1 - padding)
            x2_pad = min(w, x2 + padding)
            y2_pad = min(h, y2 + padding)
            
            # Crop the region
            cropped = frame[y1_pad:y2_pad, x1_pad:x2_pad]
            
            if cropped.size == 0:
                return None
            
            # Run pose estimation on cropped region
            poses = self.estimate_pose(cropped)
            
            if not poses:
                return None
            
            # Get the pose with highest confidence
            best_pose = max(poses, key=lambda p: p["confidence"])
            
            # Adjust keypoints back to full frame coordinates
            best_pose["keypoints"][:, 0] += x1_pad
            best_pose["keypoints"][:, 1] += y1_pad
            best_pose["bbox"][0] += x1_pad
            best_pose["bbox"][1] += y1_pad
            best_pose["bbox"][2] += x1_pad
            best_pose["bbox"][3] += y1_pad
            
            return best_pose
            
        except Exception as e:
            logger.warning(f"Error estimating pose for player: {e}")
            return None
    
    def _bbox_overlaps_any(
        self, 
        bbox: List[float], 
        bboxes: List[List[int]], 
        iou_threshold: float = 0.3
    ) -> bool:
        """
        Check if a bounding box overlaps with any box in a list.
        
        Args:
            bbox: Query bounding box [x1, y1, x2, y2]
            bboxes: List of bounding boxes to check against
            iou_threshold: Minimum IoU for overlap
            
        Returns:
            True if bbox overlaps with any box in bboxes
        """
        for other_bbox in bboxes:
            iou = self._calculate_iou(bbox, other_bbox)
            if iou >= iou_threshold:
                return True
        return False
    
    def _calculate_iou(
        self, 
        bbox1: List[float], 
        bbox2: List[float]
    ) -> float:
        """Calculate Intersection over Union between two bounding boxes."""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def get_keypoint_name(self, index: int) -> str:
        """Get the name of a keypoint by its index."""
        if 0 <= index < len(KEYPOINT_NAMES):
            return KEYPOINT_NAMES[index]
        return f"unknown_{index}"
    
    def get_keypoint_index(self, name: str) -> Optional[int]:
        """Get the index of a keypoint by its name."""
        try:
            return KEYPOINT_NAMES.index(name)
        except ValueError:
            return None
    
    def extract_body_angles(
        self, 
        keypoints: np.ndarray, 
        keypoints_conf: np.ndarray,
        min_conf: float = 0.3
    ) -> Dict[str, Optional[float]]:
        """
        Extract important body angles from keypoints for action analysis.
        
        Angles are computed for key joints useful in padel action recognition:
        - Elbow angles (arm extension during swing)
        - Shoulder angles (arm position relative to torso)
        - Knee angles (stance and movement)
        - Hip angles (torso rotation)
        
        Args:
            keypoints: Array of shape (17, 2) with (x, y) coordinates
            keypoints_conf: Array of shape (17,) with confidence scores
            min_conf: Minimum confidence threshold for valid keypoints
            
        Returns:
            Dictionary of angle names to values (in degrees) or None if not computable
        """
        angles = {}
        
        def get_valid_point(idx: int) -> Optional[np.ndarray]:
            """Get keypoint if confidence is above threshold."""
            if keypoints_conf[idx] >= min_conf:
                return keypoints[idx]
            return None
        
        def compute_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
            """Compute angle at p2 formed by p1-p2-p3."""
            v1 = p1 - p2
            v2 = p3 - p2
            
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle)
            return np.degrees(angle)
        
        # Left elbow angle (shoulder - elbow - wrist)
        left_shoulder = get_valid_point(5)
        left_elbow = get_valid_point(7)
        left_wrist = get_valid_point(9)
        if all(p is not None for p in [left_shoulder, left_elbow, left_wrist]):
            angles["left_elbow"] = compute_angle(left_shoulder, left_elbow, left_wrist)
        else:
            angles["left_elbow"] = None
        
        # Right elbow angle
        right_shoulder = get_valid_point(6)
        right_elbow = get_valid_point(8)
        right_wrist = get_valid_point(10)
        if all(p is not None for p in [right_shoulder, right_elbow, right_wrist]):
            angles["right_elbow"] = compute_angle(right_shoulder, right_elbow, right_wrist)
        else:
            angles["right_elbow"] = None
        
        # Left knee angle (hip - knee - ankle)
        left_hip = get_valid_point(11)
        left_knee = get_valid_point(13)
        left_ankle = get_valid_point(15)
        if all(p is not None for p in [left_hip, left_knee, left_ankle]):
            angles["left_knee"] = compute_angle(left_hip, left_knee, left_ankle)
        else:
            angles["left_knee"] = None
        
        # Right knee angle
        right_hip = get_valid_point(12)
        right_knee = get_valid_point(14)
        right_ankle = get_valid_point(16)
        if all(p is not None for p in [right_hip, right_knee, right_ankle]):
            angles["right_knee"] = compute_angle(right_hip, right_knee, right_ankle)
        else:
            angles["right_knee"] = None
        
        # Shoulder angle (arm height) - angle of arm relative to vertical
        if left_shoulder is not None and left_elbow is not None:
            # Vertical reference (straight up from shoulder)
            vertical = np.array([0, -1])
            arm_vec = left_elbow - left_shoulder
            cos_angle = np.dot(vertical, arm_vec) / (np.linalg.norm(arm_vec) + 1e-6)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angles["left_arm_raise"] = np.degrees(np.arccos(cos_angle))
        else:
            angles["left_arm_raise"] = None
        
        if right_shoulder is not None and right_elbow is not None:
            vertical = np.array([0, -1])
            arm_vec = right_elbow - right_shoulder
            cos_angle = np.dot(vertical, arm_vec) / (np.linalg.norm(arm_vec) + 1e-6)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angles["right_arm_raise"] = np.degrees(np.arccos(cos_angle))
        else:
            angles["right_arm_raise"] = None
        
        # Torso angle (vertical posture)
        if left_shoulder is not None and left_hip is not None:
            vertical = np.array([0, 1])  # Down
            torso_vec = left_hip - left_shoulder
            cos_angle = np.dot(vertical, torso_vec) / (np.linalg.norm(torso_vec) + 1e-6)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angles["torso_lean"] = np.degrees(np.arccos(cos_angle))
        else:
            angles["torso_lean"] = None
        
        return angles
    
    def get_wrist_velocity(
        self,
        keypoints_sequence: List[np.ndarray],
        keypoints_conf_sequence: List[np.ndarray],
        fps: float = 30.0,
        hand: str = "right"
    ) -> List[Optional[float]]:
        """
        Calculate wrist velocity over a sequence of frames.
        
        High wrist velocity indicates active shot execution.
        
        Args:
            keypoints_sequence: List of keypoint arrays over time
            keypoints_conf_sequence: List of confidence arrays over time
            fps: Video frames per second
            hand: "left", "right", or "dominant" (highest velocity)
            
        Returns:
            List of velocity values (pixels/second) or None for invalid frames
        """
        if len(keypoints_sequence) == 0:
            return []
        
        wrist_idx = 10 if hand == "right" else 9  # Right or left wrist
        min_conf = 0.3
        
        velocities = [None]  # First frame has no velocity
        
        for i in range(1, len(keypoints_sequence)):
            prev_kpts = keypoints_sequence[i - 1]
            curr_kpts = keypoints_sequence[i]
            prev_conf = keypoints_conf_sequence[i - 1]
            curr_conf = keypoints_conf_sequence[i]
            
            if prev_kpts is None or curr_kpts is None:
                velocities.append(None)
                continue
            
            if prev_conf[wrist_idx] < min_conf or curr_conf[wrist_idx] < min_conf:
                velocities.append(None)
                continue
            
            # Calculate displacement
            prev_pos = prev_kpts[wrist_idx]
            curr_pos = curr_kpts[wrist_idx]
            displacement = np.linalg.norm(curr_pos - prev_pos)
            
            # Convert to velocity (pixels per second)
            velocity = displacement * fps
            velocities.append(velocity)
        
        return velocities
