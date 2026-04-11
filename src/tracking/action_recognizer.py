"""
Action recognition module for classifying padel shots and movements.

This module analyzes pose keypoints to classify player actions into categories:
- Serve: High arm position, ball toss, follow-through
- Smash: Overhead motion, explosive arm extension
- Volley: Quick, compact motion at net
- Forehand: Lateral arm swing from dominant side
- Backhand: Lateral arm swing from non-dominant side
- Ready position: Balanced stance, arms in front

Approach Decision: Zero-Shot vs Fine-Tuning
============================================
Current implementation uses RULE-BASED classification with geometric analysis
of keypoints. This is recommended as a starting point because:

1. Zero-Shot Approach (Current):
   - No training data required
   - Immediately functional
   - Interpretable rules based on biomechanics
   - Works well for coarse action categories (serve, smash, volley)
   - Limitations: May struggle with subtle variations and transitions

2. Fine-Tuning Approach (Recommended for Production):
   For more accurate shot classification, consider fine-tuning on padel data:
   
   Available Datasets:
   - Create custom dataset: Record padel matches, annotate with LabelStudio/CVAT
   - Tennis datasets (similar biomechanics): UCF-Sports, THETIS
   - General sports action recognition: Kinetics, AVA
   
   Recommended Models for Fine-Tuning:
   - SlowFast (video-based): Good for temporal action recognition
   - I3D/R3D: 3D convolutions for motion patterns
   - Pose-based LSTM/Transformer: Use keypoint sequences as input
   
   For padel-specific shots, a pose-based sequence model is recommended:
   - Input: Sequence of 17 COCO keypoints over N frames (e.g., 16 frames)
   - Architecture: Transformer encoder or LSTM
   - Output: Shot classification (serve, smash, volley, forehand, backhand)
   
   Training requires ~500-1000 examples per class for good performance.
"""

from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
import logging
import numpy as np

logger = logging.getLogger(__name__)

# Small epsilon value to prevent division by zero in angle calculations
EPSILON = 1e-6


class PadelAction(Enum):
    """Enumeration of padel actions/shots."""
    UNKNOWN = "unknown"
    READY = "ready"
    SERVE = "serve"
    SMASH = "smash"
    VOLLEY = "volley"
    FOREHAND = "forehand"
    BACKHAND = "backhand"
    LOB = "lob"
    BANDEJA = "bandeja"  # Padel-specific overhead shot
    VIBORA = "vibora"    # Padel-specific side-spin overhead
    MOVING = "moving"


class ActionRecognizer:
    """
    Recognizes padel actions from pose keypoints.
    
    Uses geometric analysis of body pose to classify shots.
    Can be extended with ML-based classification for better accuracy.
    """
    
    # Keypoint indices (COCO format)
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16
    
    def __init__(self, config: Any):
        """
        Initialize the ActionRecognizer.
        
        Args:
            config: Configuration object containing action recognition settings
        """
        self.config = config
        
        # Thresholds for action classification (can be tuned)
        self.arm_raised_threshold = 45  # degrees from vertical
        self.overhead_threshold = 30    # degrees for overhead shots
        self.extended_arm_threshold = 150  # elbow angle for extended arm
        self.bent_knee_threshold = 140  # knee angle for movement detection
        self.wrist_velocity_threshold = 500  # pixels/second for active shot
        
        # Sequence buffer for temporal analysis
        self.pose_buffer_size = getattr(config.action_recognition, 'buffer_size', 16)
        self.pose_buffers: Dict[int, List[Dict[str, Any]]] = {}  # player_id -> pose history
        
        self.ml_model = None
        self._load_ml_model_if_available()
    
    def _load_ml_model_if_available(self):
        """
        Load ML-based action recognition model if available.
        
        This is a placeholder for future fine-tuned model integration.
        """
        use_ml = getattr(self.config.action_recognition, 'use_ml_model', False)
        model_path = getattr(self.config.action_recognition, 'model_path', None)
        
        if use_ml and model_path:
            try:
                # Placeholder for loading a fine-tuned model
                # E.g., a PyTorch LSTM/Transformer model
                logger.info(f"Loading action recognition model from: {model_path}")
                # self.ml_model = torch.load(model_path)
                logger.warning("ML-based action recognition not yet implemented")
            except Exception as e:
                logger.warning(f"Failed to load action recognition model: {e}")
                self.ml_model = None
    
    def classify_action(
        self,
        keypoints: np.ndarray,
        keypoints_conf: np.ndarray,
        player_id: Optional[int] = None,
        angles: Optional[Dict[str, Optional[float]]] = None
    ) -> Dict[str, Any]:
        """
        Classify the current action based on pose keypoints.
        
        Args:
            keypoints: Array of shape (17, 2) with (x, y) coordinates
            keypoints_conf: Array of shape (17,) with confidence scores
            player_id: Optional player ID for temporal tracking
            angles: Pre-computed body angles (optional, will compute if not provided)
            
        Returns:
            Dictionary containing:
            - action: PadelAction enum value
            - confidence: Classification confidence (0-1)
            - details: Additional information about the classification
        """
        min_conf = 0.3
        
        # Validate input
        if keypoints is None or len(keypoints) < 17:
            return {
                "action": PadelAction.UNKNOWN,
                "confidence": 0.0,
                "details": {"reason": "invalid_keypoints"}
            }
        
        # Compute angles if not provided
        if angles is None:
            angles = self._compute_angles(keypoints, keypoints_conf, min_conf)
        
        # Add to pose buffer for temporal analysis
        if player_id is not None:
            self._update_pose_buffer(player_id, keypoints, keypoints_conf, angles)
        
        # Use ML model if available
        if self.ml_model is not None and player_id is not None:
            return self._classify_with_ml(player_id)
        
        # Rule-based classification
        return self._classify_rule_based(keypoints, keypoints_conf, angles, min_conf)
    
    def _classify_rule_based(
        self,
        keypoints: np.ndarray,
        keypoints_conf: np.ndarray,
        angles: Dict[str, Optional[float]],
        min_conf: float
    ) -> Dict[str, Any]:
        """
        Rule-based action classification using geometric analysis.
        
        The classification hierarchy:
        1. Check for overhead shots (serve, smash, bandeja)
        2. Check for ground strokes (forehand, backhand)
        3. Check for volleys
        4. Default to ready/moving
        """
        details = {"angles": angles, "features": {}}
        
        # Extract key features
        features = self._extract_features(keypoints, keypoints_conf, min_conf)
        details["features"] = features
        
        # Classification logic
        confidence = 0.0
        action = PadelAction.UNKNOWN
        
        # 1. Check for overhead shots
        if features.get("arm_above_head", False):
            # Determine type of overhead shot
            if features.get("both_arms_up", False):
                # Could be serve (ball toss) or smash preparation
                if features.get("torso_tilted_back", False):
                    action = PadelAction.SERVE
                    confidence = 0.75
                else:
                    action = PadelAction.SMASH
                    confidence = 0.7
            elif features.get("arm_extended", False):
                # Extended arm overhead = smash or serve follow-through
                action = PadelAction.SMASH
                confidence = 0.7
            else:
                # Compact overhead = bandeja or vibora
                if features.get("arm_sideways", False):
                    action = PadelAction.VIBORA
                    confidence = 0.6
                else:
                    action = PadelAction.BANDEJA
                    confidence = 0.6
        
        # 2. Check for lateral strokes (forehand/backhand)
        elif features.get("arm_lateral", False):
            if features.get("right_arm_active", False):
                # Right arm swing
                action = PadelAction.FOREHAND
                confidence = 0.65
            else:
                # Left arm swing
                action = PadelAction.BACKHAND
                confidence = 0.65
        
        # 3. Check for volley (compact, forward motion at net)
        elif features.get("compact_arm_position", False) and features.get("forward_stance", False):
            action = PadelAction.VOLLEY
            confidence = 0.6
        
        # 4. Check for lob (arm rising, weight back)
        elif features.get("arm_rising", False) and features.get("weight_back", False):
            action = PadelAction.LOB
            confidence = 0.55
        
        # 5. Check for movement vs ready position
        elif features.get("knees_bent", False):
            if features.get("asymmetric_stance", False):
                action = PadelAction.MOVING
                confidence = 0.5
            else:
                action = PadelAction.READY
                confidence = 0.6
        
        else:
            action = PadelAction.UNKNOWN
            confidence = 0.3
        
        return {
            "action": action,
            "confidence": confidence,
            "details": details
        }
    
    def _extract_features(
        self,
        keypoints: np.ndarray,
        keypoints_conf: np.ndarray,
        min_conf: float
    ) -> Dict[str, Any]:
        """
        Extract geometric features from keypoints for classification.
        """
        features = {}
        
        def get_point(idx: int) -> Optional[np.ndarray]:
            if keypoints_conf[idx] >= min_conf:
                return keypoints[idx]
            return None
        
        # Get key points
        nose = get_point(self.NOSE)
        left_shoulder = get_point(self.LEFT_SHOULDER)
        right_shoulder = get_point(self.RIGHT_SHOULDER)
        left_elbow = get_point(self.LEFT_ELBOW)
        right_elbow = get_point(self.RIGHT_ELBOW)
        left_wrist = get_point(self.LEFT_WRIST)
        right_wrist = get_point(self.RIGHT_WRIST)
        left_hip = get_point(self.LEFT_HIP)
        right_hip = get_point(self.RIGHT_HIP)
        left_knee = get_point(self.LEFT_KNEE)
        right_knee = get_point(self.RIGHT_KNEE)
        
        # Feature: Arm above head
        if right_wrist is not None and right_shoulder is not None:
            features["right_arm_above_shoulder"] = right_wrist[1] < right_shoulder[1]
            if nose is not None:
                features["right_arm_above_head"] = right_wrist[1] < nose[1]
        
        if left_wrist is not None and left_shoulder is not None:
            features["left_arm_above_shoulder"] = left_wrist[1] < left_shoulder[1]
            if nose is not None:
                features["left_arm_above_head"] = left_wrist[1] < nose[1]
        
        features["arm_above_head"] = (
            features.get("right_arm_above_head", False) or 
            features.get("left_arm_above_head", False)
        )
        
        features["both_arms_up"] = (
            features.get("right_arm_above_shoulder", False) and 
            features.get("left_arm_above_shoulder", False)
        )
        
        # Feature: Arm extended (elbow angle > threshold)
        if right_shoulder is not None and right_elbow is not None and right_wrist is not None:
            right_elbow_angle = self._compute_angle(right_shoulder, right_elbow, right_wrist)
            features["right_arm_extended"] = right_elbow_angle > self.extended_arm_threshold
            features["right_elbow_angle"] = right_elbow_angle
        
        if left_shoulder is not None and left_elbow is not None and left_wrist is not None:
            left_elbow_angle = self._compute_angle(left_shoulder, left_elbow, left_wrist)
            features["left_arm_extended"] = left_elbow_angle > self.extended_arm_threshold
            features["left_elbow_angle"] = left_elbow_angle
        
        features["arm_extended"] = (
            features.get("right_arm_extended", False) or 
            features.get("left_arm_extended", False)
        )
        
        # Feature: Arm lateral (wrist far from body centerline)
        if left_shoulder is not None and right_shoulder is not None:
            center_x = (left_shoulder[0] + right_shoulder[0]) / 2
            shoulder_width = abs(right_shoulder[0] - left_shoulder[0])
            
            if right_wrist is not None:
                right_lateral_dist = abs(right_wrist[0] - center_x)
                features["right_arm_lateral"] = right_lateral_dist > shoulder_width * 1.2
                features["right_arm_active"] = right_wrist[0] > center_x
            
            if left_wrist is not None:
                left_lateral_dist = abs(left_wrist[0] - center_x)
                features["left_arm_lateral"] = left_lateral_dist > shoulder_width * 1.2
                features["left_arm_active"] = left_wrist[0] < center_x
        
        features["arm_lateral"] = (
            features.get("right_arm_lateral", False) or 
            features.get("left_arm_lateral", False)
        )
        
        # Feature: Compact arm position
        features["compact_arm_position"] = (
            not features.get("arm_extended", False) and
            not features.get("arm_lateral", False)
        )
        
        # Feature: Knees bent - also get ankle positions for later use
        left_ankle = get_point(self.LEFT_ANKLE)
        right_ankle = get_point(self.RIGHT_ANKLE)
        
        if left_hip is not None and left_knee is not None and left_ankle is not None:
            left_knee_angle = self._compute_angle(left_hip, left_knee, left_ankle)
            features["left_knee_bent"] = left_knee_angle < self.bent_knee_threshold
            features["left_knee_angle"] = left_knee_angle
        
        if right_hip is not None and right_knee is not None and right_ankle is not None:
            right_knee_angle = self._compute_angle(right_hip, right_knee, right_ankle)
            features["right_knee_bent"] = right_knee_angle < self.bent_knee_threshold
            features["right_knee_angle"] = right_knee_angle
        
        features["knees_bent"] = (
            features.get("left_knee_bent", False) or 
            features.get("right_knee_bent", False)
        )
        
        # Feature: Torso tilted back (for serve)
        if left_shoulder is not None and left_hip is not None:
            torso_vector = left_hip - left_shoulder
            vertical = np.array([0, 1])
            torso_angle = self._compute_angle_2d(torso_vector, vertical)
            features["torso_angle"] = torso_angle
            features["torso_tilted_back"] = torso_angle > 15  # degrees
        
        # Feature: Asymmetric stance (one leg forward)
        if left_ankle is not None and right_ankle is not None:
            ankle_diff_y = abs(left_ankle[1] - right_ankle[1])
            features["asymmetric_stance"] = ankle_diff_y > 30  # pixels
        
        # Feature: Weight distribution (heuristic based on hip position)
        if left_hip is not None and right_hip is not None and left_ankle is not None:
            hip_center = (left_hip + right_hip) / 2
            features["weight_back"] = hip_center[1] > left_ankle[1] - 50  # hips higher (further back in frame)
        
        # Feature: Forward stance
        features["forward_stance"] = features.get("knees_bent", False) and not features.get("weight_back", False)
        
        return features
    
    def _compute_angle(
        self, 
        p1: np.ndarray, 
        p2: np.ndarray, 
        p3: np.ndarray
    ) -> float:
        """Compute angle at p2 formed by p1-p2-p3 in degrees."""
        v1 = p1 - p2
        v2 = p3 - p2
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + EPSILON)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return np.degrees(np.arccos(cos_angle))
    
    def _compute_angle_2d(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Compute angle between two 2D vectors in degrees."""
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + EPSILON)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return np.degrees(np.arccos(cos_angle))
    
    def _compute_angles(
        self,
        keypoints: np.ndarray,
        keypoints_conf: np.ndarray,
        min_conf: float
    ) -> Dict[str, Optional[float]]:
        """Compute body angles from keypoints."""
        angles = {}
        
        def get_point(idx: int) -> Optional[np.ndarray]:
            if keypoints_conf[idx] >= min_conf:
                return keypoints[idx]
            return None
        
        # Left elbow
        left_shoulder = get_point(self.LEFT_SHOULDER)
        left_elbow = get_point(self.LEFT_ELBOW)
        left_wrist = get_point(self.LEFT_WRIST)
        if all(p is not None for p in [left_shoulder, left_elbow, left_wrist]):
            angles["left_elbow"] = self._compute_angle(left_shoulder, left_elbow, left_wrist)
        
        # Right elbow
        right_shoulder = get_point(self.RIGHT_SHOULDER)
        right_elbow = get_point(self.RIGHT_ELBOW)
        right_wrist = get_point(self.RIGHT_WRIST)
        if all(p is not None for p in [right_shoulder, right_elbow, right_wrist]):
            angles["right_elbow"] = self._compute_angle(right_shoulder, right_elbow, right_wrist)
        
        # Left knee
        left_hip = get_point(self.LEFT_HIP)
        left_knee = get_point(self.LEFT_KNEE)
        left_ankle = get_point(self.LEFT_ANKLE)
        if all(p is not None for p in [left_hip, left_knee, left_ankle]):
            angles["left_knee"] = self._compute_angle(left_hip, left_knee, left_ankle)
        
        # Right knee
        right_hip = get_point(self.RIGHT_HIP)
        right_knee = get_point(self.RIGHT_KNEE)
        right_ankle = get_point(self.RIGHT_ANKLE)
        if all(p is not None for p in [right_hip, right_knee, right_ankle]):
            angles["right_knee"] = self._compute_angle(right_hip, right_knee, right_ankle)
        
        return angles
    
    def _update_pose_buffer(
        self,
        player_id: int,
        keypoints: np.ndarray,
        keypoints_conf: np.ndarray,
        angles: Dict[str, Optional[float]]
    ):
        """Update pose history buffer for temporal analysis."""
        if player_id not in self.pose_buffers:
            self.pose_buffers[player_id] = []
        
        self.pose_buffers[player_id].append({
            "keypoints": keypoints.copy(),
            "keypoints_conf": keypoints_conf.copy(),
            "angles": angles.copy()
        })
        
        # Keep buffer at maximum size
        if len(self.pose_buffers[player_id]) > self.pose_buffer_size:
            self.pose_buffers[player_id].pop(0)
    
    def _classify_with_ml(self, player_id: int) -> Dict[str, Any]:
        """
        Classify action using ML model on pose sequence.
        
        Placeholder for future implementation.
        """
        if self.ml_model is None:
            return {
                "action": PadelAction.UNKNOWN,
                "confidence": 0.0,
                "details": {"reason": "ml_model_not_loaded"}
            }
        
        # Get pose buffer
        buffer = self.pose_buffers.get(player_id, [])
        if len(buffer) < self.pose_buffer_size:
            return {
                "action": PadelAction.UNKNOWN,
                "confidence": 0.0,
                "details": {"reason": "insufficient_sequence_length"}
            }

        
        return {
            "action": PadelAction.UNKNOWN,
            "confidence": 0.0,
            "details": {"reason": "ml_classification_not_implemented"}
        }
    
    def classify_action_sequence(
        self,
        keypoints_sequence: List[np.ndarray],
        keypoints_conf_sequence: List[np.ndarray],
        fps: float = 30.0
    ) -> Dict[str, Any]:
        """
        Classify action from a sequence of poses.
        
        This method uses temporal information for more accurate classification.
        Useful for identifying complete shot motions (preparation -> execution -> follow-through).
        
        Args:
            keypoints_sequence: List of keypoint arrays over time
            keypoints_conf_sequence: List of confidence arrays over time
            fps: Video frames per second
            
        Returns:
            Dictionary containing action classification with temporal analysis
        """
        if len(keypoints_sequence) < 2:
            return {
                "action": PadelAction.UNKNOWN,
                "confidence": 0.0,
                "details": {"reason": "sequence_too_short"}
            }
        
        # Classify each frame
        frame_actions = []
        for kpts, conf in zip(keypoints_sequence, keypoints_conf_sequence):
            if kpts is not None:
                result = self.classify_action(kpts, conf)
                frame_actions.append(result["action"])
            else:
                frame_actions.append(PadelAction.UNKNOWN)
        
        # Find dominant action in sequence (excluding UNKNOWN and READY)
        action_counts = {}
        for action in frame_actions:
            if action not in [PadelAction.UNKNOWN, PadelAction.READY]:
                action_counts[action] = action_counts.get(action, 0) + 1
        
        if not action_counts:
            # If no specific action detected, check for READY
            ready_count = sum(1 for a in frame_actions if a == PadelAction.READY)
            if ready_count > len(frame_actions) * 0.5:
                return {
                    "action": PadelAction.READY,
                    "confidence": 0.6,
                    "details": {"frame_actions": [a.value for a in frame_actions]}
                }
            return {
                "action": PadelAction.UNKNOWN,
                "confidence": 0.3,
                "details": {"frame_actions": [a.value for a in frame_actions]}
            }
        
        # Get most common action
        dominant_action = max(action_counts, key=action_counts.get)
        confidence = action_counts[dominant_action] / len(frame_actions)
        
        # Boost confidence if action is consistent
        if confidence > 0.5:
            confidence = min(0.9, confidence + 0.1)
        
        return {
            "action": dominant_action,
            "confidence": confidence,
            "details": {
                "frame_actions": [a.value for a in frame_actions],
                "action_counts": {k.value: v for k, v in action_counts.items()}
            }
        }
    
    def get_action_name(self, action: PadelAction) -> str:
        """Get human-readable name for an action."""
        names = {
            PadelAction.UNKNOWN: "Unknown",
            PadelAction.READY: "Ready Position",
            PadelAction.SERVE: "Serve",
            PadelAction.SMASH: "Smash",
            PadelAction.VOLLEY: "Volley",
            PadelAction.FOREHAND: "Forehand",
            PadelAction.BACKHAND: "Backhand",
            PadelAction.LOB: "Lob",
            PadelAction.BANDEJA: "Bandeja",
            PadelAction.VIBORA: "Vibora",
            PadelAction.MOVING: "Moving"
        }
        return names.get(action, action.value)
    
    def clear_buffers(self):
        """Clear all pose buffers."""
        self.pose_buffers.clear()
