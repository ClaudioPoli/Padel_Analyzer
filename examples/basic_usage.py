"""
Example script demonstrating how to use the Padel Analyzer.

This script shows the basic workflow for analyzing a padel match video.
"""

from padel_analyzer import PadelAnalyzer
from padel_analyzer.utils.config import Config
import json
from pathlib import Path


def main():
    """Main function demonstrating basic usage."""
    
    # Example 1: Basic usage with default configuration
    print("Example 1: Basic usage with default configuration")
    print("-" * 50)
    
    analyzer = PadelAnalyzer()
    
    # Note: Replace with actual video path when testing
    video_path = "path/to/your/padel_match.mp4"
    
    # Check if video exists (for demo purposes, we skip actual analysis)
    if Path(video_path).exists():
        try:
            results = analyzer.analyze_video(video_path)
            print(f"Analysis completed successfully!")
            print(f"Field info: {results['field_info']}")
            print(f"Player tracks: {len(results['player_tracks'])} players detected")
            print(f"Ball tracks: {len(results['ball_tracks']['positions'])} ball positions")
        except Exception as e:
            print(f"Error during analysis: {e}")
    else:
        print(f"Video file not found: {video_path}")
        print("Please provide a valid video path to run the analysis.")
    
    print("\n")
    
    # Example 2: Using custom configuration
    print("Example 2: Using custom configuration")
    print("-" * 50)
    
    # Create custom configuration
    config = Config()
    config.model.device = "cpu"  # Use "cuda" if GPU is available
    config.tracking.player_detection_confidence = 0.7
    config.tracking.ball_detection_confidence = 0.5
    config.field_detection.auto_calibrate = True
    
    # Initialize analyzer with custom config
    custom_analyzer = PadelAnalyzer(config)
    
    print(f"Configured device: {config.model.device}")
    print(f"Player detection confidence: {config.tracking.player_detection_confidence}")
    print(f"Ball detection confidence: {config.tracking.ball_detection_confidence}")
    
    print("\n")
    
    # Example 3: Saving and loading configuration
    print("Example 3: Saving and loading configuration")
    print("-" * 50)
    
    # Save configuration to file
    config_path = "/tmp/padel_analyzer_config.json"
    config.to_file(config_path)
    print(f"Configuration saved to: {config_path}")
    
    # Load configuration from file
    loaded_config = Config.from_file(config_path)
    print(f"Configuration loaded from: {config_path}")
    print(f"Loaded device setting: {loaded_config.model.device}")
    
    print("\n")
    
    # Example 4: Batch processing multiple videos
    print("Example 4: Batch processing multiple videos")
    print("-" * 50)
    
    video_paths = [
        "path/to/match1.mp4",
        "path/to/match2.mp4",
        "path/to/match3.mov",
    ]
    
    print(f"Processing {len(video_paths)} videos...")
    print("Note: This is a demonstration. Replace with actual video paths.")
    
    # Uncomment below to run batch analysis
    # results = analyzer.analyze_video_batch(video_paths)
    # for i, result in enumerate(results):
    #     if "error" in result:
    #         print(f"Video {i+1} failed: {result['error']}")
    #     else:
    #         print(f"Video {i+1} analyzed successfully")
    
    print("\n")
    
    # Example 5: Understanding the results structure
    print("Example 5: Understanding the results structure")
    print("-" * 50)
    
    # Mock results structure for demonstration
    example_results = {
        "field_info": {
            "boundaries": "Court boundary coordinates",
            "lines": "Detected court lines",
            "corners": "Corner points",
            "homography_matrix": "Perspective transformation matrix",
            "court_mask": "Binary mask of court area"
        },
        "player_tracks": [
            {
                "player_id": "Unique player identifier",
                "positions": "List of (x, y, frame_number) positions",
                "bounding_boxes": "Bounding boxes for each frame",
                "team": "Team assignment",
                "confidence_scores": "Detection confidence per frame"
            }
        ],
        "ball_tracks": {
            "positions": "List of ball positions",
            "velocities": "Velocity vectors",
            "in_play": "Boolean flags for ball in play",
            "trajectory": "Interpolated trajectory",
            "confidence_scores": "Detection confidence per frame"
        },
        "metadata": {
            "fps": "Video frame rate",
            "resolution": "Video resolution",
            "duration": "Video duration",
            "frame_count": "Total number of frames"
        }
    }
    
    print("Results structure:")
    print(json.dumps(example_results, indent=2))


if __name__ == "__main__":
    main()
