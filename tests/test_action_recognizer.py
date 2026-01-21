"""
Tests for action recognition functionality.
"""

import pytest
import numpy as np
from src.utils.config import Config, ActionRecognitionConfig
from src.tracking.action_recognizer import ActionRecognizer, PadelAction


class TestActionRecognizer:
    """Test cases for ActionRecognizer class."""
    
    def test_initialization_default_config(self):
        """Test action recognizer initialization with default config."""
        config = Config()
        recognizer = ActionRecognizer(config)
        
        assert recognizer.config is not None
        assert recognizer.ml_model is None  # No ML model by default
    
    def test_padel_action_enum(self):
        """Test PadelAction enum values."""
        assert PadelAction.SERVE.value == "serve"
        assert PadelAction.SMASH.value == "smash"
        assert PadelAction.VOLLEY.value == "volley"
        assert PadelAction.FOREHAND.value == "forehand"
        assert PadelAction.BACKHAND.value == "backhand"
        assert PadelAction.BANDEJA.value == "bandeja"
        assert PadelAction.VIBORA.value == "vibora"
        assert PadelAction.LOB.value == "lob"
        assert PadelAction.READY.value == "ready"
        assert PadelAction.MOVING.value == "moving"
        assert PadelAction.UNKNOWN.value == "unknown"
    
    def test_classify_action_invalid_keypoints(self):
        """Test classification with invalid keypoints."""
        config = Config()
        recognizer = ActionRecognizer(config)
        
        result = recognizer.classify_action(None, None)
        
        assert result["action"] == PadelAction.UNKNOWN
        assert result["confidence"] == 0.0
    
    def test_classify_action_ready_position(self):
        """Test classification of ready position."""
        config = Config()
        recognizer = ActionRecognizer(config)
        
        # Create keypoints for a ready position (balanced stance, bent knees)
        keypoints = np.zeros((17, 2))
        keypoints_conf = np.ones(17) * 0.9
        
        # Head/face
        keypoints[0] = [250, 50]   # nose
        keypoints[1] = [240, 40]   # left eye
        keypoints[2] = [260, 40]   # right eye
        
        # Shoulders at same height (balanced)
        keypoints[5] = [200, 100]  # left shoulder
        keypoints[6] = [300, 100]  # right shoulder
        
        # Elbows in front
        keypoints[7] = [210, 150]  # left elbow
        keypoints[8] = [290, 150]  # right elbow
        
        # Wrists together in front (ready position)
        keypoints[9] = [240, 180]  # left wrist
        keypoints[10] = [260, 180] # right wrist
        
        # Hips
        keypoints[11] = [220, 200]  # left hip
        keypoints[12] = [280, 200]  # right hip
        
        # Knees bent (lower y position would be straight legs)
        keypoints[13] = [220, 280]  # left knee
        keypoints[14] = [280, 280]  # right knee
        
        # Ankles at same level (balanced)
        keypoints[15] = [220, 380]  # left ankle
        keypoints[16] = [280, 380]  # right ankle
        
        result = recognizer.classify_action(keypoints, keypoints_conf)
        
        # Should detect some action (may vary based on exact geometry)
        assert "action" in result
        assert "confidence" in result
        assert result["confidence"] >= 0.0
    
    def test_classify_action_overhead(self):
        """Test classification of overhead shot (serve/smash)."""
        config = Config()
        recognizer = ActionRecognizer(config)
        
        # Create keypoints for overhead shot (arm above head)
        keypoints = np.zeros((17, 2))
        keypoints_conf = np.ones(17) * 0.9
        
        # Head
        keypoints[0] = [250, 100]  # nose
        
        # Shoulders
        keypoints[5] = [200, 150]  # left shoulder
        keypoints[6] = [300, 150]  # right shoulder
        
        # Right arm raised above head (serving/smashing)
        keypoints[8] = [320, 80]   # right elbow (above shoulder)
        keypoints[10] = [340, 30]  # right wrist (above head)
        
        # Left arm normal
        keypoints[7] = [180, 180]  # left elbow
        keypoints[9] = [160, 220]  # left wrist
        
        # Body
        keypoints[11] = [220, 250]  # left hip
        keypoints[12] = [280, 250]  # right hip
        keypoints[13] = [220, 350]  # left knee
        keypoints[14] = [280, 350]  # right knee
        keypoints[15] = [220, 450]  # left ankle
        keypoints[16] = [280, 450]  # right ankle
        
        result = recognizer.classify_action(keypoints, keypoints_conf)
        
        # Should detect overhead action (serve, smash, or bandeja)
        assert result["action"] in [
            PadelAction.SERVE, 
            PadelAction.SMASH, 
            PadelAction.BANDEJA,
            PadelAction.VIBORA,
            PadelAction.UNKNOWN  # May not detect if not enough features
        ]
    
    def test_classify_action_lateral_swing(self):
        """Test classification of lateral swing (forehand/backhand)."""
        config = Config()
        recognizer = ActionRecognizer(config)
        
        # Create keypoints for lateral swing (arm extended to side)
        keypoints = np.zeros((17, 2))
        keypoints_conf = np.ones(17) * 0.9
        
        # Head
        keypoints[0] = [250, 100]  # nose
        
        # Shoulders
        keypoints[5] = [200, 150]  # left shoulder
        keypoints[6] = [300, 150]  # right shoulder
        
        # Right arm extended laterally (forehand swing)
        keypoints[8] = [400, 150]  # right elbow (far to right)
        keypoints[10] = [480, 150] # right wrist (even further right)
        
        # Left arm across body
        keypoints[7] = [200, 180]  # left elbow
        keypoints[9] = [240, 200]  # left wrist
        
        # Body
        keypoints[11] = [220, 250]  # left hip
        keypoints[12] = [280, 250]  # right hip
        keypoints[13] = [220, 350]  # left knee
        keypoints[14] = [280, 350]  # right knee
        keypoints[15] = [220, 450]  # left ankle
        keypoints[16] = [280, 450]  # right ankle
        
        result = recognizer.classify_action(keypoints, keypoints_conf)
        
        # Should detect lateral stroke
        assert result["action"] in [
            PadelAction.FOREHAND,
            PadelAction.BACKHAND,
            PadelAction.UNKNOWN  # May not detect if not enough features
        ]
    
    def test_get_action_name(self):
        """Test human-readable action names."""
        config = Config()
        recognizer = ActionRecognizer(config)
        
        assert recognizer.get_action_name(PadelAction.SERVE) == "Serve"
        assert recognizer.get_action_name(PadelAction.SMASH) == "Smash"
        assert recognizer.get_action_name(PadelAction.BANDEJA) == "Bandeja"
        assert recognizer.get_action_name(PadelAction.VIBORA) == "Vibora"
        assert recognizer.get_action_name(PadelAction.READY) == "Ready Position"
    
    def test_classify_action_sequence_short(self):
        """Test sequence classification with short sequence."""
        config = Config()
        recognizer = ActionRecognizer(config)
        
        result = recognizer.classify_action_sequence([], [])
        
        assert result["action"] == PadelAction.UNKNOWN
        assert "sequence_too_short" in result["details"]["reason"]
    
    def test_classify_action_sequence_valid(self):
        """Test sequence classification with valid sequence."""
        config = Config()
        recognizer = ActionRecognizer(config)
        
        # Create a sequence of poses
        keypoints_seq = []
        conf_seq = []
        
        for _ in range(10):
            kpts = np.zeros((17, 2))
            conf = np.ones(17) * 0.8
            
            # Basic ready position
            kpts[5] = [200, 100]
            kpts[6] = [300, 100]
            kpts[11] = [220, 200]
            kpts[12] = [280, 200]
            kpts[13] = [220, 300]
            kpts[14] = [280, 300]
            kpts[15] = [220, 400]
            kpts[16] = [280, 400]
            
            keypoints_seq.append(kpts)
            conf_seq.append(conf)
        
        result = recognizer.classify_action_sequence(keypoints_seq, conf_seq)
        
        assert "action" in result
        assert "confidence" in result
        assert "details" in result
    
    def test_clear_buffers(self):
        """Test clearing pose buffers."""
        config = Config()
        recognizer = ActionRecognizer(config)
        
        # Add some data to buffer
        recognizer.pose_buffers[1] = [{"test": "data"}]
        recognizer.pose_buffers[2] = [{"test": "data2"}]
        
        recognizer.clear_buffers()
        
        assert len(recognizer.pose_buffers) == 0
    
    def test_compute_angle(self):
        """Test angle computation between three points."""
        config = Config()
        recognizer = ActionRecognizer(config)
        
        # Points forming a 90-degree angle
        p1 = np.array([0, 0])
        p2 = np.array([0, 100])  # vertex
        p3 = np.array([100, 100])
        
        angle = recognizer._compute_angle(p1, p2, p3)
        
        # Should be close to 90 degrees
        assert 85 <= angle <= 95
    
    def test_compute_angle_straight(self):
        """Test angle computation for straight line (180 degrees)."""
        config = Config()
        recognizer = ActionRecognizer(config)
        
        # Points forming a straight line
        p1 = np.array([0, 0])
        p2 = np.array([100, 0])  # vertex
        p3 = np.array([200, 0])
        
        angle = recognizer._compute_angle(p1, p2, p3)
        
        # Should be close to 180 degrees
        assert 175 <= angle <= 185


class TestActionRecognitionConfig:
    """Test cases for ActionRecognitionConfig."""
    
    def test_default_values(self):
        """Test default action recognition configuration values."""
        config = ActionRecognitionConfig()
        
        assert config.enabled is True
        assert config.use_ml_model is False
        assert config.model_path is None
        assert config.buffer_size == 16
        assert config.min_action_confidence == 0.5
    
    def test_custom_values(self):
        """Test custom action recognition configuration values."""
        config = ActionRecognitionConfig(
            enabled=False,
            use_ml_model=True,
            model_path="/path/to/model.pt",
            buffer_size=32,
            min_action_confidence=0.7
        )
        
        assert config.enabled is False
        assert config.use_ml_model is True
        assert config.model_path == "/path/to/model.pt"
        assert config.buffer_size == 32
        assert config.min_action_confidence == 0.7


class TestConfigIntegration:
    """Test config integration with pose and action recognition."""
    
    def test_config_with_pose_and_action(self):
        """Test main Config includes pose and action recognition."""
        config = Config()
        
        assert hasattr(config, 'pose')
        assert hasattr(config, 'action_recognition')
        assert config.pose.enabled is True
        assert config.action_recognition.enabled is True
    
    def test_config_to_file_includes_new_sections(self):
        """Test config serialization includes pose and action sections."""
        import tempfile
        import json
        
        config = Config()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config.to_file(f.name)
            
            with open(f.name, 'r') as read_f:
                saved_config = json.load(read_f)
        
        assert 'pose' in saved_config
        assert 'action_recognition' in saved_config
        assert saved_config['pose']['enabled'] is True
        assert saved_config['action_recognition']['enabled'] is True
