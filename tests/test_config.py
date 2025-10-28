"""
Tests for the Config class.
"""

import pytest
import json
import tempfile
from pathlib import Path
from padel_analyzer.utils.config import Config, VideoConfig, TrackingConfig


class TestConfig:
    """Test cases for Config class."""
    
    def test_default_initialization(self):
        """Test config initialization with defaults."""
        config = Config()
        
        assert config.video is not None
        assert config.tracking is not None
        assert config.field_detection is not None
        assert config.model is not None
    
    def test_video_config_defaults(self):
        """Test VideoConfig default values."""
        config = Config()
        
        assert '.mp4' in config.video.supported_formats
        assert '.mov' in config.video.supported_formats
        assert config.video.target_fps is None
        assert config.video.target_resolution is None
    
    def test_tracking_config_defaults(self):
        """Test TrackingConfig default values."""
        config = Config()
        
        assert config.tracking.player_detection_confidence == 0.5
        assert config.tracking.ball_detection_confidence == 0.3
        assert config.tracking.max_tracking_distance == 100
        assert config.tracking.interpolate_missing is True
    
    def test_model_config_defaults(self):
        """Test ModelConfig default values."""
        config = Config()
        
        assert config.model.device == "auto"
        assert config.model.batch_size == 1
    
    def test_save_and_load_config(self):
        """Test saving and loading config from file."""
        config = Config()
        config.model.device = "cuda"
        config.tracking.player_detection_confidence = 0.8
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_path = f.name
        
        try:
            # Save config
            config.to_file(config_path)
            
            # Load config
            loaded_config = Config.from_file(config_path)
            
            assert loaded_config.model.device == "cuda"
            assert loaded_config.tracking.player_detection_confidence == 0.8
        finally:
            Path(config_path).unlink()
    
    def test_update_config(self):
        """Test updating config parameters."""
        config = Config()
        initial_device = config.model.device
        
        new_model_config = Config().model
        new_model_config.device = "cuda"
        config.update(model=new_model_config)
        
        assert config.model.device == "cuda"
