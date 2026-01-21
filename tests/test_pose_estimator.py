"""
Tests for pose estimation functionality.
"""

import pytest
import numpy as np
from src.utils.config import Config, PoseConfig
from src.tracking.pose_estimator import PoseEstimator, KEYPOINT_NAMES, SKELETON_CONNECTIONS


class TestPoseEstimator:
    """Test cases for PoseEstimator class."""
    
    def test_initialization_default_config(self):
        """Test pose estimator initialization with default config."""
        config = Config()
        estimator = PoseEstimator(config)
        
        assert estimator.config is not None
        # Model may or may not be loaded depending on environment
    
    def test_initialization_pose_disabled(self):
        """Test pose estimator with pose disabled still initializes."""
        config = Config()
        config.pose.enabled = False
        
        estimator = PoseEstimator(config)
        assert estimator.config is not None
    
    def test_keypoint_names_count(self):
        """Test that COCO keypoint names are correctly defined."""
        assert len(KEYPOINT_NAMES) == 17
        assert "nose" in KEYPOINT_NAMES
        assert "left_wrist" in KEYPOINT_NAMES
        assert "right_ankle" in KEYPOINT_NAMES
    
    def test_skeleton_connections(self):
        """Test skeleton connections are valid indices."""
        for connection in SKELETON_CONNECTIONS:
            assert len(connection) == 2
            assert 0 <= connection[0] < 17
            assert 0 <= connection[1] < 17
    
    def test_get_keypoint_name(self):
        """Test keypoint name retrieval."""
        config = Config()
        estimator = PoseEstimator(config)
        
        assert estimator.get_keypoint_name(0) == "nose"
        assert estimator.get_keypoint_name(10) == "right_wrist"
        assert estimator.get_keypoint_name(100).startswith("unknown")
    
    def test_get_keypoint_index(self):
        """Test keypoint index retrieval."""
        config = Config()
        estimator = PoseEstimator(config)
        
        assert estimator.get_keypoint_index("nose") == 0
        assert estimator.get_keypoint_index("right_wrist") == 10
        assert estimator.get_keypoint_index("invalid") is None
    
    def test_extract_body_angles_valid_keypoints(self):
        """Test body angle extraction with valid keypoints."""
        config = Config()
        estimator = PoseEstimator(config)
        
        # Create synthetic keypoints forming a simple pose
        keypoints = np.zeros((17, 2))
        keypoints_conf = np.ones(17) * 0.9
        
        # Set up a basic pose (approximate positions)
        # Shoulders at y=100
        keypoints[5] = [200, 100]  # left shoulder
        keypoints[6] = [300, 100]  # right shoulder
        
        # Elbows at y=150 (arms pointing down)
        keypoints[7] = [180, 150]  # left elbow
        keypoints[8] = [320, 150]  # right elbow
        
        # Wrists at y=200
        keypoints[9] = [170, 200]  # left wrist
        keypoints[10] = [330, 200]  # right wrist
        
        # Hips at y=200
        keypoints[11] = [220, 200]  # left hip
        keypoints[12] = [280, 200]  # right hip
        
        # Knees at y=300
        keypoints[13] = [220, 300]  # left knee
        keypoints[14] = [280, 300]  # right knee
        
        # Ankles at y=400
        keypoints[15] = [220, 400]  # left ankle
        keypoints[16] = [280, 400]  # right ankle
        
        angles = estimator.extract_body_angles(keypoints, keypoints_conf)
        
        # Should have computed angles for key joints
        assert "left_elbow" in angles
        assert "right_elbow" in angles
        assert "left_knee" in angles
        assert "right_knee" in angles
        
        # Angles should be reasonable (not None)
        assert angles["left_knee"] is not None
        assert angles["right_knee"] is not None
    
    def test_extract_body_angles_low_confidence(self):
        """Test body angle extraction with low confidence keypoints."""
        config = Config()
        estimator = PoseEstimator(config)
        
        keypoints = np.zeros((17, 2))
        # All keypoints have low confidence
        keypoints_conf = np.ones(17) * 0.1
        
        angles = estimator.extract_body_angles(keypoints, keypoints_conf, min_conf=0.5)
        
        # All angles should be None due to low confidence
        assert angles["left_elbow"] is None
        assert angles["right_elbow"] is None
    
    def test_calculate_iou(self):
        """Test IoU calculation between bounding boxes."""
        config = Config()
        estimator = PoseEstimator(config)
        
        # Identical boxes
        bbox1 = [0, 0, 100, 100]
        bbox2 = [0, 0, 100, 100]
        assert estimator._calculate_iou(bbox1, bbox2) == 1.0
        
        # Non-overlapping boxes
        bbox3 = [0, 0, 50, 50]
        bbox4 = [100, 100, 150, 150]
        assert estimator._calculate_iou(bbox3, bbox4) == 0.0
        
        # Partially overlapping boxes
        bbox5 = [0, 0, 100, 100]
        bbox6 = [50, 50, 150, 150]
        iou = estimator._calculate_iou(bbox5, bbox6)
        assert 0 < iou < 1
    
    def test_estimate_pose_no_model(self):
        """Test pose estimation when model is not loaded."""
        config = Config()
        estimator = PoseEstimator(config)
        estimator.pose_model = None  # Force no model
        
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = estimator.estimate_pose(frame)
        
        assert result == []
    
    def test_wrist_velocity_empty_sequence(self):
        """Test wrist velocity with empty sequence."""
        config = Config()
        estimator = PoseEstimator(config)
        
        velocities = estimator.get_wrist_velocity([], [], fps=30.0)
        assert velocities == []


class TestPoseConfig:
    """Test cases for PoseConfig."""
    
    def test_default_values(self):
        """Test default pose configuration values."""
        config = PoseConfig()
        
        assert config.enabled is True
        assert config.pose_model == "yolov8n-pose.pt"
        assert config.min_confidence == 0.25
        assert config.estimate_for_all_frames is False
        assert config.frame_sample_rate == 3
    
    def test_custom_values(self):
        """Test custom pose configuration values."""
        config = PoseConfig(
            enabled=False,
            pose_model="yolov8m-pose.pt",
            min_confidence=0.5,
            estimate_for_all_frames=True,
            frame_sample_rate=1
        )
        
        assert config.enabled is False
        assert config.pose_model == "yolov8m-pose.pt"
        assert config.min_confidence == 0.5
        assert config.estimate_for_all_frames is True
        assert config.frame_sample_rate == 1
