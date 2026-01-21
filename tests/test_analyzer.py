"""
Tests for the PadelAnalyzer main class.
"""

import pytest
from pathlib import Path
from src import PadelAnalyzer
from src.utils.config import Config


class TestPadelAnalyzer:
    """Test cases for PadelAnalyzer class."""
    
    def test_initialization_default_config(self):
        """Test analyzer initialization with default config."""
        analyzer = PadelAnalyzer()
        assert analyzer.config is not None
        assert analyzer.video_loader is not None
        assert analyzer.field_detector is not None
        assert analyzer.player_tracker is not None
        assert analyzer.ball_tracker is not None
    
    def test_initialization_custom_config(self):
        """Test analyzer initialization with custom config."""
        config = Config()
        config.model.device = "cpu"
        
        analyzer = PadelAnalyzer(config)
        assert analyzer.config.model.device == "cpu"
    
    def test_analyze_video_file_not_found(self):
        """Test that FileNotFoundError is raised for non-existent video."""
        analyzer = PadelAnalyzer()
        
        with pytest.raises(FileNotFoundError):
            analyzer.analyze_video("nonexistent_video.mp4")
    
    def test_analyze_video_batch_empty_list(self):
        """Test batch analysis with empty list."""
        analyzer = PadelAnalyzer()
        results = analyzer.analyze_video_batch([])
        assert results == []
    
    def test_analyze_video_batch_with_errors(self):
        """Test batch analysis handles errors gracefully."""
        analyzer = PadelAnalyzer()
        results = analyzer.analyze_video_batch(["nonexistent1.mp4", "nonexistent2.mp4"])
        
        assert len(results) == 2
        assert all("error" in result for result in results)
