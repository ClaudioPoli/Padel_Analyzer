"""
Demo script to test the video processing implementation.

This script demonstrates:
1. Device detection (CUDA/MPS/CPU)
2. Configuration setup
3. Video analysis pipeline (with synthetic video if needed)
"""

import sys
from pathlib import Path
import numpy as np
import cv2
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from padel_analyzer import PadelAnalyzer
from padel_analyzer.utils.config import Config
from padel_analyzer.utils.device import get_device_info

def print_device_info():
    """Print information about available devices."""
    print("\n" + "="*60)
    print("Device Information")
    print("="*60)
    
    info = get_device_info()
    
    print(f"CPU Available: {info['cpu_available']}")
    print(f"CUDA Available: {info['cuda_available']}")
    if info['cuda_available']:
        print(f"  - Device Count: {info.get('cuda_device_count', 'N/A')}")
        print(f"  - Device Name: {info.get('cuda_device_name', 'N/A')}")
    
    print(f"MPS Available: {info['mps_available']}")
    print(f"\nRecommended Device: {info['recommended_device']}")
    print("="*60 + "\n")


def create_synthetic_video(output_path: Path, duration_seconds: int = 5, fps: int = 30):
    """
    Create a synthetic video for testing (simulating a padel court).
    
    Args:
        output_path: Where to save the video
        duration_seconds: Length of video
        fps: Frames per second
    """
    print(f"Creating synthetic test video at {output_path}...")
    
    width, height = 1280, 720
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    total_frames = duration_seconds * fps
    
    for frame_idx in range(total_frames):
        # Create frame with green background (simulating court)
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :] = (50, 150, 50)  # Green background
        
        # Draw court lines (white)
        # Outer boundaries
        cv2.rectangle(frame, (100, 100), (1180, 620), (255, 255, 255), 3)
        # Center line
        cv2.line(frame, (640, 100), (640, 620), (255, 255, 255), 2)
        # Service lines
        cv2.line(frame, (100, 360), (1180, 360), (255, 255, 255), 2)
        
        # Simulate 4 players (circles moving around)
        t = frame_idx / fps  # Time in seconds
        
        # Player 1 (top-left)
        p1_x = int(320 + 50 * np.sin(t * 2))
        p1_y = int(230 + 30 * np.cos(t * 1.5))
        cv2.circle(frame, (p1_x, p1_y), 30, (0, 0, 255), -1)  # Red
        
        # Player 2 (top-right)
        p2_x = int(960 + 50 * np.sin(t * 1.8))
        p2_y = int(230 + 30 * np.cos(t * 2.2))
        cv2.circle(frame, (p2_x, p2_y), 30, (0, 0, 255), -1)  # Red
        
        # Player 3 (bottom-left)
        p3_x = int(320 + 50 * np.sin(t * 1.5))
        p3_y = int(490 + 30 * np.cos(t * 2))
        cv2.circle(frame, (p3_x, p3_y), 30, (255, 0, 0), -1)  # Blue
        
        # Player 4 (bottom-right)
        p4_x = int(960 + 50 * np.sin(t * 2.2))
        p4_y = int(490 + 30 * np.cos(t * 1.8))
        cv2.circle(frame, (p4_x, p4_y), 30, (255, 0, 0), -1)  # Blue
        
        # Simulate ball (small yellow circle)
        ball_x = int(640 + 300 * np.sin(t * 3))
        ball_y = int(360 + 150 * np.cos(t * 4))
        cv2.circle(frame, (ball_x, ball_y), 8, (0, 255, 255), -1)  # Yellow
        
        writer.write(frame)
    
    writer.release()
    print(f"Created {total_frames} frames at {fps} FPS")


def test_video_analysis(video_path: Path):
    """
    Test video analysis pipeline.
    
    Args:
        video_path: Path to video file
    """
    print("\n" + "="*60)
    print("Testing Video Analysis Pipeline")
    print("="*60)
    
    # Create configuration
    config = Config()
    config.model.device = "auto"  # Auto-detect best device
    config.tracking.player_detection_confidence = 0.3  # Lower for synthetic video
    config.tracking.ball_detection_confidence = 0.2
    
    print(f"\nConfiguration:")
    print(f"  Device: {config.model.device}")
    print(f"  Player Model: {config.model.player_model}")
    print(f"  Player Confidence: {config.tracking.player_detection_confidence}")
    print(f"  Ball Confidence: {config.tracking.ball_detection_confidence}")
    
    # Initialize analyzer
    print("\nInitializing PadelAnalyzer...")
    analyzer = PadelAnalyzer(config)
    
    # Analyze video
    print(f"\nAnalyzing video: {video_path}")
    print("This may take a moment...\n")
    
    try:
        results = analyzer.analyze_video(str(video_path))
        
        print("\n" + "="*60)
        print("Analysis Results")
        print("="*60)
        
        # Metadata
        metadata = results.get('metadata', {})
        print(f"\nVideo Metadata:")
        print(f"  FPS: {metadata.get('fps', 'N/A')}")
        print(f"  Resolution: {metadata.get('width', 'N/A')}x{metadata.get('height', 'N/A')}")
        print(f"  Duration: {metadata.get('duration', 'N/A'):.2f}s")
        print(f"  Frame Count: {metadata.get('frame_count', 'N/A')}")
        
        # Field detection
        field_info = results.get('field_info', {})
        print(f"\nField Detection:")
        print(f"  Confidence: {field_info.get('confidence', 0):.2f}")
        print(f"  Lines Detected: {len(field_info.get('lines', []))}")
        print(f"  Corners Detected: {len(field_info.get('corners', []))}")
        
        # Player tracking
        player_tracks = results.get('player_tracks', [])
        print(f"\nPlayer Tracking:")
        print(f"  Players Detected: {len(player_tracks)}")
        for i, track in enumerate(player_tracks):
            print(f"  Player {track['player_id']}:")
            print(f"    Team: {track.get('team', 'Unknown')}")
            print(f"    Frames Tracked: {len(track['positions'])}")
            if len(track['confidence_scores']) > 0:
                avg_conf = sum(track['confidence_scores']) / len(track['confidence_scores'])
                print(f"    Avg Confidence: {avg_conf:.3f}")
        
        # Ball tracking
        ball_tracks = results.get('ball_tracks', {})
        positions = ball_tracks.get('positions', [])
        trajectory = ball_tracks.get('trajectory', [])
        print(f"\nBall Tracking:")
        print(f"  Raw Detections: {len(positions)}")
        print(f"  Trajectory Points: {len(trajectory)}")
        if len(ball_tracks.get('confidence_scores', [])) > 0:
            avg_conf = sum(ball_tracks['confidence_scores']) / len(ball_tracks['confidence_scores'])
            print(f"  Avg Confidence: {avg_conf:.3f}")
        
        print("\n" + "="*60)
        print("Analysis Complete!")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"\nError during analysis: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function."""
    print("\n")
    print("#" * 60)
    print("# Padel Analyzer - Video Processing Demo")
    print("#" * 60)
    
    # Print device information
    print_device_info()
    
    # Check if video path provided
    if len(sys.argv) > 1:
        video_path = Path(sys.argv[1])
        if not video_path.exists():
            print(f"Error: Video file not found: {video_path}")
            sys.exit(1)
    else:
        # Create synthetic video for testing in a temporary directory
        import tempfile
        temp_dir = Path(tempfile.gettempdir())
        video_path = temp_dir / "test_padel_video.mp4"
        create_synthetic_video(video_path, duration_seconds=3, fps=30)
    
    # Test analysis
    success = test_video_analysis(video_path)
    
    if success:
        print("\n✅ Demo completed successfully!")
    else:
        print("\n❌ Demo failed. Check logs above for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
