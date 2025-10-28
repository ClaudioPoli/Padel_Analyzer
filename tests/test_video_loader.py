"""
Tests for the VideoLoader class.
"""

import pytest
import tempfile
from pathlib import Path
from padel_analyzer.video.video_loader import VideoLoader
from padel_analyzer.utils.config import Config


class TestVideoLoader:
    """Test cases for VideoLoader class."""
    
    def test_initialization(self):
        """Test video loader initialization."""
        config = Config()
        loader = VideoLoader(config)
        
        assert loader.config is not None
        assert len(loader.SUPPORTED_FORMATS) > 0
    
    def test_supported_formats(self):
        """Test that common video formats are supported."""
        config = Config()
        loader = VideoLoader(config)
        
        assert '.mp4' in loader.SUPPORTED_FORMATS
        assert '.mov' in loader.SUPPORTED_FORMATS
        assert '.avi' in loader.SUPPORTED_FORMATS
    
    def test_load_nonexistent_file(self):
        """Test loading a non-existent file raises FileNotFoundError."""
        config = Config()
        loader = VideoLoader(config)
        
        with pytest.raises(FileNotFoundError):
            loader.load(Path("nonexistent_video.mp4"))
    
    def test_load_unsupported_format(self):
        """Test loading unsupported format raises ValueError."""
        config = Config()
        loader = VideoLoader(config)
        
        # Create a temporary file with unsupported extension
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
            temp_path = Path(f.name)
        
        try:
            with pytest.raises(ValueError, match="Unsupported video format"):
                loader.load(temp_path)
        finally:
            temp_path.unlink()
    
    def test_load_supported_format_structure(self):
        """Test that load raises ValueError for invalid video files."""
        config = Config()
        loader = VideoLoader(config)
        
        # Create a temporary file with supported extension but invalid content
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            temp_path = Path(f.name)
        
        try:
            # Should raise ValueError because file is not a valid video
            with pytest.raises(ValueError, match="Failed to open video file"):
                loader.load(temp_path)
        finally:
            temp_path.unlink()
