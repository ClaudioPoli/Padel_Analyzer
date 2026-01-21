#!/usr/bin/env python3
"""
Test hybrid approach: custom detection + standard pose estimation.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src import PadelAnalyzer
from src.utils.config import Config
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_hybrid_detection(video_path: str, use_custom_model: bool = False):
    """
    Test hybrid approach with optional custom trained model.
    
    Args:
        video_path: Path to test video
        use_custom_model: If True, use custom trained model (after training completes)
    """
    # Create config
    config = Config()
    
    if use_custom_model:
        # Use custom trained detection model
        custom_model_path = "runs/detect/padel_yolov8n/weights/best.pt"
        if not Path(custom_model_path).exists():
            logger.warning(f"Custom model not found at {custom_model_path}")
            logger.info("Using standard YOLOv8n. Train custom model first!")
            config.model.player_model = "yolov8n.pt"
        else:
            config.model.player_model = custom_model_path
            logger.info(f"Using custom trained model: {custom_model_path}")
    else:
        config.model.player_model = "yolov8n.pt"
        logger.info("Using standard YOLOv8n detection")
    
    # Enable pose extraction
    config.model.use_pose = True
    config.model.pose_model = "yolov8n-pose.pt"
    config.tracking.extract_keypoints = True
    
    # Set device
    config.model.device = "mps"  # For M1/M2/M3
    
    logger.info("\n📊 Configuration:")
    logger.info(f"   Detection model: {config.model.player_model}")
    logger.info(f"   Pose model: {config.model.pose_model}")
    logger.info(f"   Extract keypoints: {config.tracking.extract_keypoints}")
    logger.info(f"   Device: {config.model.device}")
    
    # Create analyzer
    analyzer = PadelAnalyzer(config)
    
    # Analyze video
    logger.info(f"\n🎬 Analyzing video: {video_path}")
    results = analyzer.analyze_video(video_path)
    
    # Display results
    logger.info("\n✅ Analysis complete!")
    logger.info(f"\n📈 Results:")
    
    player_tracks = results.get("player_tracks", [])
    logger.info(f"   Players detected: {len(player_tracks)}")
    
    for i, track in enumerate(player_tracks):
        player_id = track.get("player_id")
        team = track.get("team", "Unknown")
        num_frames = len(track.get("frame_numbers", []))
        has_keypoints = any(kp is not None for kp in track.get("keypoints_sequence", []))
        num_keypoints = sum(1 for kp in track.get("keypoints_sequence", []) if kp is not None)
        
        logger.info(f"\n   Player {i+1} (ID: {player_id}):")
        logger.info(f"      Team: {team}")
        logger.info(f"      Frames tracked: {num_frames}")
        logger.info(f"      Keypoints extracted: {num_keypoints}/{num_frames} frames")
        
        if has_keypoints:
            # Show sample keypoint
            for kp in track.get("keypoints_sequence", []):
                if kp is not None:
                    logger.info(f"      Sample keypoints shape: {kp.shape} (17 points, x/y coords)")
                    break
    
    ball_tracks = results.get("ball_tracks", {})
    ball_positions = ball_tracks.get("positions", [])
    logger.info(f"\n   Ball positions detected: {len(ball_positions)}")
    
    field_info = results.get("field_info", {})
    field_confidence = field_info.get("confidence", 0.0)
    logger.info(f"   Field detection confidence: {field_confidence:.2f}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test hybrid detection + pose approach")
    parser.add_argument(
        "video_path",
        type=str,
        help="Path to video file"
    )
    parser.add_argument(
        "--custom",
        action="store_true",
        help="Use custom trained model (default: standard YOLOv8n)"
    )
    
    args = parser.parse_args()
    
    video_path = Path(args.video_path)
    if not video_path.exists():
        print(f"❌ Video not found: {video_path}")
        sys.exit(1)
    
    test_hybrid_detection(str(video_path), use_custom_model=args.custom)
