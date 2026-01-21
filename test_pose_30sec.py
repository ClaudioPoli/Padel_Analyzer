"""
Test script for pose estimation on the first 30 seconds of each video in data/personal/
Output is saved to data/output/
"""

import sys
from pathlib import Path
import logging
import json
import cv2
import numpy as np
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils.config import Config
from src.tracking.pose_estimator import PoseEstimator
from src.utils.device import get_device

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def draw_pose(frame, pose_data):
    """
    Draw pose keypoints and skeleton on a frame.
    
    Args:
        frame: Video frame
        pose_data: Pose data dictionary with keypoints and confidence
    
    Returns:
        Frame with drawn pose
    """
    from src.tracking.pose_estimator import SKELETON_CONNECTIONS, KEYPOINT_NAMES
    
    keypoints = pose_data['keypoints']
    keypoints_conf = pose_data['keypoints_conf']
    
    # Draw skeleton connections
    for conn in SKELETON_CONNECTIONS:
        pt1_idx, pt2_idx = conn
        
        # Check if both keypoints are visible
        if keypoints_conf[pt1_idx] > 0.3 and keypoints_conf[pt2_idx] > 0.3:
            pt1 = tuple(map(int, keypoints[pt1_idx]))
            pt2 = tuple(map(int, keypoints[pt2_idx]))
            
            cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
    
    # Draw keypoints
    for i, (kp, conf) in enumerate(zip(keypoints, keypoints_conf)):
        if conf > 0.3:
            x, y = map(int, kp)
            # Different colors for different body parts
            if i <= 4:  # Head
                color = (255, 0, 0)
            elif i <= 10:  # Arms
                color = (0, 255, 255)
            else:  # Legs
                color = (255, 255, 0)
            
            cv2.circle(frame, (x, y), 3, color, -1)
    
    # Draw bounding box
    bbox = pose_data['bbox']
    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 255), 2)
    
    # Draw confidence score
    conf_text = f"Conf: {pose_data['confidence']:.2f}"
    cv2.putText(frame, conf_text, (bbox[0], bbox[1] - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
    
    return frame


def process_video(video_path: Path, output_dir: Path, config: Config, duration_seconds: int = 30):
    """
    Process the first N seconds of a video and save pose estimation results.
    
    Args:
        video_path: Path to video file
        output_dir: Directory to save output
        config: Configuration object
        duration_seconds: Number of seconds to process
    """
    logger.info(f"Processing video: {video_path.name}")
    
    # Initialize pose estimator
    pose_estimator = PoseEstimator(config)
    
    if pose_estimator.pose_model is None:
        logger.error(f"Failed to load pose model for {video_path.name}")
        return None
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path.name}")
        return None
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate frames to process
    max_frames = min(fps * duration_seconds, total_frames)
    
    logger.info(f"Video properties: {width}x{height} @ {fps}fps, processing {max_frames} frames ({duration_seconds}s)")
    
    # Prepare output video
    output_video_path = output_dir / f"{video_path.stem}_pose_output.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
    
    # Statistics
    stats = {
        "video_name": video_path.name,
        "duration_seconds": duration_seconds,
        "fps": fps,
        "resolution": f"{width}x{height}",
        "frames_processed": 0,
        "frames_with_poses": 0,
        "total_poses_detected": 0,
        "avg_poses_per_frame": 0,
        "processing_time_seconds": 0,
        "device": str(pose_estimator.device)
    }
    
    frame_idx = 0
    start_time = datetime.now()
    
    try:
        while frame_idx < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Estimate poses
            poses = pose_estimator.estimate_pose(frame)
            
            # Update statistics
            stats["frames_processed"] += 1
            if len(poses) > 0:
                stats["frames_with_poses"] += 1
                stats["total_poses_detected"] += len(poses)
            
            # Draw poses on frame
            output_frame = frame.copy()
            for pose_data in poses:
                output_frame = draw_pose(output_frame, pose_data)
            
            # Add frame info
            info_text = f"Frame: {frame_idx}/{max_frames} | Poses: {len(poses)}"
            cv2.putText(output_frame, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Write output frame
            out.write(output_frame)
            
            frame_idx += 1
            
            # Progress update every 30 frames
            if frame_idx % 30 == 0:
                logger.info(f"Processed {frame_idx}/{max_frames} frames...")
    
    except KeyboardInterrupt:
        logger.warning("Processing interrupted by user")
    
    finally:
        cap.release()
        out.release()
        
        # Calculate final statistics
        end_time = datetime.now()
        stats["processing_time_seconds"] = (end_time - start_time).total_seconds()
        
        if stats["frames_processed"] > 0:
            stats["avg_poses_per_frame"] = stats["total_poses_detected"] / stats["frames_processed"]
            stats["fps_processing"] = stats["frames_processed"] / stats["processing_time_seconds"]
        
        # Save statistics
        stats_path = output_dir / f"{video_path.stem}_pose_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Completed {video_path.name}:")
        logger.info(f"  - Processed: {stats['frames_processed']} frames")
        logger.info(f"  - Frames with poses: {stats['frames_with_poses']}")
        logger.info(f"  - Total poses detected: {stats['total_poses_detected']}")
        logger.info(f"  - Avg poses per frame: {stats['avg_poses_per_frame']:.2f}")
        logger.info(f"  - Processing time: {stats['processing_time_seconds']:.2f}s")
        logger.info(f"  - Processing FPS: {stats.get('fps_processing', 0):.2f}")
        logger.info(f"  - Output video: {output_video_path}")
        logger.info(f"  - Stats file: {stats_path}")
        
        return stats


def main():
    """Main function to process all videos in data/personal/"""
    
    # Setup paths
    project_root = Path(__file__).parent
    personal_dir = project_root / "data" / "personal"
    output_dir = project_root / "data" / "output"
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create configuration
    config = Config()
    config.pose.enabled = True
    config.pose.min_confidence = 0.25
    config.pose.pose_model = "yolov8n-pose.pt"
    
    # Detect device
    device = get_device(config.model.device)
    logger.info(f"Using device: {device}")
    
    # Find all video files
    video_extensions = ['.mp4', '.mov', '.MP4', '.MOV', '.avi', '.mkv']
    video_files = []
    for ext in video_extensions:
        video_files.extend(personal_dir.glob(f"*{ext}"))
    
    if not video_files:
        logger.error(f"No video files found in {personal_dir}")
        return
    
    logger.info(f"Found {len(video_files)} video files to process")
    
    # Process each video
    all_stats = []
    for video_path in sorted(video_files):
        try:
            stats = process_video(video_path, output_dir, config, duration_seconds=30)
            if stats:
                all_stats.append(stats)
        except Exception as e:
            logger.error(f"Error processing {video_path.name}: {e}", exc_info=True)
            continue
    
    # Save combined statistics
    if all_stats:
        combined_stats_path = output_dir / "all_videos_pose_stats.json"
        with open(combined_stats_path, 'w') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "total_videos": len(all_stats),
                "videos": all_stats
            }, f, indent=2)
        
        logger.info(f"\n{'='*60}")
        logger.info("SUMMARY:")
        logger.info(f"Total videos processed: {len(all_stats)}")
        logger.info(f"Combined stats saved to: {combined_stats_path}")
        logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
