"""
Test script for hybrid player tracking + pose estimation approach.
This script uses player detection to find bounding boxes, then applies
pose estimation only on those specific regions for improved accuracy.

Processes the first 30 seconds of each video in data/personal/
and saves annotated videos to data/output/
"""

import os
import sys
import cv2
import json
import numpy as np
from pathlib import Path
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.utils.config import Config
from src.tracking.player_tracker import PlayerTracker
from src.tracking.pose_estimator import KEYPOINT_NAMES, SKELETON_CONNECTIONS


def draw_pose_on_frame(frame, poses, color=(0, 255, 0), thickness=2):
    """
    Draw pose keypoints and skeleton on frame.
    
    Args:
        frame: Frame to draw on
        poses: List of pose dictionaries from pose estimator
        color: Color for drawing (BGR)
        thickness: Line thickness
    """
    for pose in poses:
        keypoints = pose["keypoints"]
        keypoints_conf = pose["keypoints_conf"]
        bbox = pose["bbox"]
        
        # Draw bounding box
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        
        # Draw skeleton connections
        for conn in SKELETON_CONNECTIONS:
            idx1, idx2 = conn
            
            # Check if both keypoints are valid (confidence > 0.5)
            if (keypoints_conf[idx1] > 0.5 and keypoints_conf[idx2] > 0.5):
                pt1 = tuple(keypoints[idx1].astype(int))
                pt2 = tuple(keypoints[idx2].astype(int))
                
                cv2.line(frame, pt1, pt2, color, thickness)
        
        # Draw keypoints
        for i, (kpt, conf) in enumerate(zip(keypoints, keypoints_conf)):
            if conf > 0.5:  # Only draw confident keypoints
                x, y = int(kpt[0]), int(kpt[1])
                cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)


def draw_player_tracks_with_poses(frame, player_tracks, frame_idx):
    """
    Draw player tracks with pose information on frame.
    
    Args:
        frame: Frame to draw on
        player_tracks: List of player track dictionaries
        frame_idx: Current frame index
    """
    colors = [
        (255, 0, 0),    # Blue
        (0, 255, 0),    # Green
        (0, 0, 255),    # Red
        (255, 255, 0),  # Cyan
    ]
    
    for i, track in enumerate(player_tracks):
        color = colors[i % len(colors)]
        
        # Find the detection for this frame
        try:
            frame_indices = track["frame_numbers"]
            if frame_idx in frame_indices:
                idx_in_track = frame_indices.index(frame_idx)
                
                # Draw bounding box
                bbox = track["bounding_boxes"][idx_in_track]
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                
                # Draw player ID
                player_id = track["player_id"]
                cv2.putText(
                    frame,
                    f"Player {player_id}",
                    (bbox[0], bbox[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )
                
                # Draw pose if available
                keypoints = track["keypoints_sequence"][idx_in_track]
                keypoints_conf = track["keypoints_conf_sequence"][idx_in_track]
                
                if keypoints is not None and keypoints_conf is not None:
                    # Draw skeleton connections
                    for conn in SKELETON_CONNECTIONS:
                        idx1, idx2 = conn
                        
                        if (keypoints_conf[idx1] > 0.5 and keypoints_conf[idx2] > 0.5):
                            pt1 = tuple(keypoints[idx1].astype(int))
                            pt2 = tuple(keypoints[idx2].astype(int))
                            cv2.line(frame, pt1, pt2, color, 2)
                    
                    # Draw keypoints
                    for kpt, conf in zip(keypoints, keypoints_conf):
                        if conf > 0.5:
                            x, y = int(kpt[0]), int(kpt[1])
                            cv2.circle(frame, (x, y), 3, (255, 255, 255), -1)
                            cv2.circle(frame, (x, y), 4, color, 1)
        
        except (ValueError, IndexError):
            continue


def process_video(video_path, output_dir, config, duration_seconds=30):
    """
    Process a single video with hybrid player tracking + pose estimation.
    
    Args:
        video_path: Path to input video
        output_dir: Directory to save output
        config: Configuration object
        duration_seconds: Duration to process (in seconds)
    """
    print(f"\n{'='*80}")
    print(f"Processing: {video_path}")
    print(f"{'='*80}")
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate frames to process
    max_frames = min(fps * duration_seconds, total_frames)
    
    print(f"Video info: {width}x{height} @ {fps} FPS")
    print(f"Processing first {max_frames} frames ({duration_seconds}s)")
    
    # Initialize player tracker with pose estimation enabled
    print("\nInitializing hybrid player tracker + pose estimation...")
    tracker = PlayerTracker(config, use_pose_estimation=True)
    
    # Collect all frames for tracking
    frames = []
    frame_idx = 0
    
    print("Loading frames...")
    while frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        frame_idx += 1
        
        if frame_idx % 100 == 0:
            print(f"  Loaded {frame_idx}/{max_frames} frames")
    
    print(f"Total frames loaded: {len(frames)}")
    
    # Simple field info (no field detection for now)
    field_info = {
        "court_mask": None
    }
    
    # Track players with poses - process frames directly
    print("\nTracking players with pose estimation...")
    start_time = time.time()
    
    all_detections = []
    for frame_idx, frame in enumerate(frames):
        detections = tracker.detect_players_in_frame(frame, frame_idx, field_mask=None)
        all_detections.extend(detections)
        
        if (frame_idx + 1) % 100 == 0:
            elapsed = time.time() - start_time
            fps_so_far = (frame_idx + 1) / elapsed
            print(f"  Processed {frame_idx + 1}/{len(frames)} frames ({fps_so_far:.1f} fps)")
    
    # Associate detections into tracks
    print("\nAssociating detections into player tracks...")
    player_tracks = tracker._associate_tracks(all_detections, field_info)
    
    elapsed = time.time() - start_time
    fps_processing = len(frames) / elapsed if elapsed > 0 else 0
    
    print(f"\nTracking completed in {elapsed:.2f}s ({fps_processing:.1f} fps)")
    print(f"Found {len(player_tracks)} player tracks")
    
    # Print statistics for each track
    for i, track in enumerate(player_tracks):
        num_frames = len(track["frame_numbers"])
        num_poses = sum(1 for kp in track["keypoints_sequence"] if kp is not None)
        pose_percentage = (num_poses / num_frames * 100) if num_frames > 0 else 0
        
        print(f"  Player {track['player_id']}: {num_frames} frames, "
              f"{num_poses} poses ({pose_percentage:.1f}% coverage)")
    
    # Create output video
    output_path = output_dir / f"{video_path.stem}_hybrid_pose.mp4"
    print(f"\nCreating output video: {output_path}")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    # Annotate frames
    print("Annotating frames...")
    for frame_idx, frame in enumerate(frames):
        annotated = frame.copy()
        
        # Draw player tracks with poses
        draw_player_tracks_with_poses(annotated, player_tracks, frame_idx)
        
        # Add frame info
        cv2.putText(
            annotated,
            f"Frame: {frame_idx}/{len(frames)} | Players: {len(player_tracks)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
        
        out.write(annotated)
        
        if (frame_idx + 1) % 100 == 0:
            print(f"  Annotated {frame_idx + 1}/{len(frames)} frames")
    
    out.release()
    cap.release()
    
    print(f"✓ Output saved to: {output_path}")
    
    # Save statistics
    stats = {
        "video_name": video_path.name,
        "duration_seconds": duration_seconds,
        "frames_processed": len(frames),
        "fps": fps,
        "resolution": f"{width}x{height}",
        "player_tracks": []
    }
    
    for track in player_tracks:
        num_frames = len(track["frame_numbers"])
        num_poses = sum(1 for kp in track["keypoints_sequence"] if kp is not None)
        
        stats["player_tracks"].append({
            "player_id": int(track["player_id"]),
            "frames_detected": num_frames,
            "poses_detected": num_poses,
            "pose_coverage_percent": round(num_poses / num_frames * 100, 2) if num_frames > 0 else 0,
            "avg_confidence": float(np.mean(track["confidence_scores"])) if track["confidence_scores"] else 0
        })
    
    stats_path = output_dir / f"{video_path.stem}_hybrid_pose_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"✓ Statistics saved to: {stats_path}")
    
    return stats


def main():
    """Main function to process all videos in data/personal/"""
    
    # Setup paths
    project_root = Path(__file__).parent
    personal_dir = project_root / "data" / "personal"
    output_dir = project_root / "data" / "output"
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load configuration
    config_path = project_root / "config.example.json"
    if config_path.exists():
        config = Config.from_file(str(config_path))
        print(f"Loaded config from: {config_path}")
    else:
        config = Config()
        print("Using default configuration")
    
    # Find all video files
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    video_files = []
    
    if personal_dir.exists():
        for ext in video_extensions:
            video_files.extend(personal_dir.glob(f"*{ext}"))
    else:
        print(f"Error: Directory not found: {personal_dir}")
        return
    
    if not video_files:
        print(f"No video files found in {personal_dir}")
        return
    
    print(f"\nFound {len(video_files)} video(s) to process")
    print(f"Output directory: {output_dir}")
    
    # Process each video
    results = []
    for i, video_path in enumerate(sorted(video_files), 1):
        print(f"\n[{i}/{len(video_files)}] Processing: {video_path.name}")
        
        try:
            stats = process_video(video_path, output_dir, config, duration_seconds=30)
            if stats:
                results.append(stats)
        except Exception as e:
            print(f"✗ Error processing {video_path.name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print(f"\n{'='*80}")
    print("PROCESSING COMPLETE")
    print(f"{'='*80}")
    print(f"Processed {len(results)}/{len(video_files)} videos successfully")
    
    if results:
        total_players = sum(len(r["player_tracks"]) for r in results)
        total_frames = sum(r["frames_processed"] for r in results)
        
        print(f"Total frames processed: {total_frames}")
        print(f"Total player tracks: {total_players}")
        
        # Calculate average pose coverage
        all_coverages = []
        for result in results:
            for track in result["player_tracks"]:
                all_coverages.append(track["pose_coverage_percent"])
        
        if all_coverages:
            avg_coverage = np.mean(all_coverages)
            print(f"Average pose detection coverage: {avg_coverage:.1f}%")
    
    print(f"\nAll output files saved to: {output_dir}")


if __name__ == "__main__":
    main()
