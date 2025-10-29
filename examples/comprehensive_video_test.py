"""
Comprehensive video analysis test for padel matches.

This script tests all major components:
- Field detection
- Player tracking and recognition
- Ball tracking and trajectory
- Player movement analysis
- Shot type detection (basic implementation)

Usage:
    python examples/comprehensive_video_test.py <path_to_video>
"""

import sys
import logging
from pathlib import Path
import numpy as np
import cv2
from typing import List, Dict, Any, Tuple

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from padel_analyzer import PadelAnalyzer
from padel_analyzer.utils.config import Config
from padel_analyzer.utils.device import get_device_info


class ShotTypeDetector:
    """
    Basic shot type detection based on player movement and ball trajectory.
    
    Shot types detected:
    - Serve: Ball starts from back of court, high trajectory
    - Volley: Ball at net, fast reaction
    - Smash: High ball position, downward trajectory
    - Groundstroke: Ball from back court, medium height
    """
    
    def __init__(self):
        self.shot_types = []
    
    def detect_shot_type(
        self, 
        player_pos: Tuple[int, int],
        ball_pos: Tuple[int, int],
        ball_velocity: Tuple[float, float],
        court_info: Dict[str, Any]
    ) -> str:
        """
        Detect the type of shot based on positions and velocities.
        
        Args:
            player_pos: (x, y) position of player
            ball_pos: (x, y) position of ball
            ball_velocity: (vx, vy) velocity of ball
            court_info: Information about court boundaries
            
        Returns:
            Shot type as string
        """
        # Simple heuristic-based detection
        # In production, this would use ML models
        
        px, py = player_pos
        bx, by = ball_pos
        vx, vy = ball_velocity
        
        # Calculate distance from player to ball
        distance = np.sqrt((bx - px)**2 + (by - py)**2)
        
        # Calculate ball speed
        speed = np.sqrt(vx**2 + vy**2)
        
        # Detect shot type based on heuristics
        if distance < 100 and speed > 500:
            # Close to player, high speed - likely volley or smash
            if vy > 200:  # Downward trajectory
                return "smash"
            else:
                return "volley"
        elif distance < 150 and speed < 300:
            return "serve"
        elif speed > 300:
            return "groundstroke"
        else:
            return "unknown"
    
    def analyze_rally(
        self,
        player_tracks: List[Dict[str, Any]],
        ball_tracks: Dict[str, Any],
        field_info: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Analyze a rally to detect all shots.
        
        Args:
            player_tracks: Player tracking data
            ball_tracks: Ball tracking data
            field_info: Field detection data
            
        Returns:
            List of detected shots with metadata
        """
        shots = []
        
        # Get ball trajectory
        trajectory = ball_tracks.get('trajectory', [])
        velocities = ball_tracks.get('velocities', [])
        
        if len(trajectory) < 2 or len(velocities) == 0:
            return shots
        
        # For each ball position, find nearest player and detect shot
        for i, (bx, by, frame) in enumerate(trajectory):
            if i >= len(velocities):
                break
            
            vx, vy = velocities[i]
            
            # Find nearest player at this frame
            min_dist = float('inf')
            nearest_player = None
            
            for player in player_tracks:
                # Find player position at this frame
                if frame not in player.get('frame_numbers', []):
                    continue
                
                frame_idx = player['frame_numbers'].index(frame)
                if frame_idx >= len(player['positions']):
                    continue
                
                px, py = player['positions'][frame_idx]
                dist = np.sqrt((bx - px)**2 + (by - py)**2)
                
                if dist < min_dist:
                    min_dist = dist
                    nearest_player = player
            
            # If player is close enough, this might be a shot
            if nearest_player and min_dist < 200:
                px, py = nearest_player['positions'][frame_idx]
                shot_type = self.detect_shot_type(
                    (px, py), (bx, by), (vx, vy), field_info
                )
                
                shots.append({
                    'frame': frame,
                    'player_id': nearest_player['player_id'],
                    'team': nearest_player.get('team', 'Unknown'),
                    'shot_type': shot_type,
                    'ball_position': (bx, by),
                    'player_position': (px, py),
                    'ball_speed': np.sqrt(vx**2 + vy**2)
                })
        
        return shots


def visualize_analysis(
    video_path: Path,
    results: Dict[str, Any],
    output_path: Path = None,
    max_frames: int = 300
):
    """
    Create a visualization of the analysis results.
    
    Args:
        video_path: Path to original video
        results: Analysis results
        output_path: Path to save visualization (optional)
        max_frames: Maximum frames to process
    """
    logger.info("Creating visualization...")
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        logger.error(f"Could not open video: {video_path}")
        return
    
    fps = results['metadata'].get('fps', 30)
    width = results['metadata'].get('width', 1280)
    height = results['metadata'].get('height', 720)
    
    # Setup video writer if output requested
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    # Draw analysis on frames
    frame_idx = 0
    player_tracks = results.get('player_tracks', [])
    ball_trajectory = results.get('ball_tracks', {}).get('trajectory', [])
    field_info = results.get('field_info', {})
    
    # Create lookup for ball positions by frame
    ball_by_frame = {frame: (x, y) for x, y, frame in ball_trajectory}
    
    while frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Draw field boundaries
        if field_info.get('corners'):
            corners = field_info['corners'][:4]
            if len(corners) == 4:
                pts = np.array(corners, dtype=np.int32)
                cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
        
        # Draw players
        for player in player_tracks:
            if frame_idx not in player.get('frame_numbers', []):
                continue
            
            idx = player['frame_numbers'].index(frame_idx)
            if idx >= len(player['bounding_boxes']):
                continue
            
            bbox = player['bounding_boxes'][idx]
            x1, y1, x2, y2 = bbox
            
            # Color by team
            color = (0, 0, 255) if player.get('team') == 'A' else (255, 0, 0)
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw player ID and team
            label = f"P{player['player_id']} ({player.get('team', '?')})"
            cv2.putText(frame, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw ball
        if frame_idx in ball_by_frame:
            bx, by = ball_by_frame[frame_idx]
            cv2.circle(frame, (int(bx), int(by)), 8, (0, 255, 255), -1)
            cv2.circle(frame, (int(bx), int(by)), 12, (0, 255, 255), 2)
        
        # Draw trajectory trail (last 10 positions)
        trail_frames = [f for f in range(max(0, frame_idx - 10), frame_idx) 
                       if f in ball_by_frame]
        if len(trail_frames) > 1:
            trail_points = [ball_by_frame[f] for f in trail_frames]
            for i in range(len(trail_points) - 1):
                pt1 = (int(trail_points[i][0]), int(trail_points[i][1]))
                pt2 = (int(trail_points[i+1][0]), int(trail_points[i+1][1]))
                cv2.line(frame, pt1, pt2, (255, 255, 0), 2)
        
        # Write frame info
        info_text = f"Frame: {frame_idx}"
        cv2.putText(frame, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if writer:
            writer.write(frame)
        
        frame_idx += 1
        
        if frame_idx % 100 == 0:
            logger.info(f"Processed {frame_idx} frames")
    
    cap.release()
    if writer:
        writer.release()
        logger.info(f"Visualization saved to: {output_path}")


def test_video_analysis(video_path: Path):
    """
    Comprehensive test of video analysis on a real padel video.
    
    Args:
        video_path: Path to padel video file
    """
    print("\n" + "=" * 80)
    print("COMPREHENSIVE PADEL VIDEO ANALYSIS TEST")
    print("=" * 80)
    
    # Check video exists
    if not video_path.exists():
        print(f"\n❌ ERROR: Video file not found: {video_path}")
        sys.exit(1)
    
    print(f"\n📹 Video: {video_path}")
    print(f"   Size: {video_path.stat().st_size / (1024*1024):.2f} MB")
    
    # Print device info
    print("\n" + "-" * 80)
    print("DEVICE INFORMATION")
    print("-" * 80)
    device_info = get_device_info()
    print(f"CPU Available: {device_info['cpu_available']}")
    print(f"CUDA Available: {device_info['cuda_available']}")
    if device_info['cuda_available']:
        print(f"  └─ Devices: {device_info.get('cuda_device_count', 0)}")
        print(f"  └─ Name: {device_info.get('cuda_device_name', 'N/A')}")
    print(f"MPS Available: {device_info['mps_available']}")
    print(f"Recommended: {device_info['recommended_device']}")
    
    # Configure analyzer
    print("\n" + "-" * 80)
    print("ANALYZER CONFIGURATION")
    print("-" * 80)
    config = Config()
    config.model.device = "auto"
    config.tracking.player_detection_confidence = 0.4  # Lower for real videos
    config.tracking.ball_detection_confidence = 0.25
    
    print(f"Device: {config.model.device}")
    print(f"Player Model: {config.model.player_model}")
    print(f"Player Confidence: {config.tracking.player_detection_confidence}")
    print(f"Ball Confidence: {config.tracking.ball_detection_confidence}")
    
    # Initialize analyzer
    print("\n" + "-" * 80)
    print("INITIALIZING ANALYZER")
    print("-" * 80)
    analyzer = PadelAnalyzer(config)
    print("✓ Analyzer initialized")
    
    # Analyze video
    print("\n" + "-" * 80)
    print("ANALYZING VIDEO")
    print("-" * 80)
    print("This may take several minutes depending on video length...")
    
    try:
        results = analyzer.analyze_video(str(video_path))
    except Exception as e:
        print(f"\n❌ ERROR during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Test 1: Field Detection
    print("\n" + "=" * 80)
    print("TEST 1: FIELD DETECTION")
    print("=" * 80)
    field_info = results.get('field_info', {})
    field_confidence = field_info.get('confidence', 0)
    lines_detected = len(field_info.get('lines', []))
    corners_detected = len(field_info.get('corners', []))
    
    print(f"Confidence: {field_confidence:.2f}")
    print(f"Lines Detected: {lines_detected}")
    print(f"Corners Detected: {corners_detected}")
    print(f"Homography: {'✓ Available' if field_info.get('homography_matrix') is not None else '✗ Not available'}")
    
    if field_confidence > 0.5 and corners_detected >= 4:
        print("✅ PASS: Field detected successfully")
    elif field_confidence > 0.3:
        print("⚠️  PARTIAL: Field partially detected")
    else:
        print("❌ FAIL: Field detection insufficient")
    
    # Test 2: Player Tracking
    print("\n" + "=" * 80)
    print("TEST 2: PLAYER TRACKING")
    print("=" * 80)
    player_tracks = results.get('player_tracks', [])
    print(f"Players Detected: {len(player_tracks)}")
    
    for i, player in enumerate(player_tracks):
        frames_tracked = len(player.get('positions', []))
        avg_conf = 0
        if len(player.get('confidence_scores', [])) > 0:
            avg_conf = sum(player['confidence_scores']) / len(player['confidence_scores'])
        
        print(f"\nPlayer {player['player_id']}:")
        print(f"  Team: {player.get('team', 'Unknown')}")
        print(f"  Frames Tracked: {frames_tracked}")
        print(f"  Avg Confidence: {avg_conf:.3f}")
        print(f"  Total Detections: {len(player.get('confidence_scores', []))}")
    
    if len(player_tracks) >= 2:
        print("\n✅ PASS: Players tracked successfully")
    elif len(player_tracks) > 0:
        print("\n⚠️  PARTIAL: Some players tracked")
    else:
        print("\n❌ FAIL: No players tracked")
    
    # Test 3: Ball Tracking
    print("\n" + "=" * 80)
    print("TEST 3: BALL TRACKING")
    print("=" * 80)
    ball_tracks = results.get('ball_tracks', {})
    raw_detections = len(ball_tracks.get('positions', []))
    trajectory_points = len(ball_tracks.get('trajectory', []))
    velocities = len(ball_tracks.get('velocities', []))
    
    print(f"Raw Detections: {raw_detections}")
    print(f"Trajectory Points: {trajectory_points}")
    print(f"Velocity Calculations: {velocities}")
    
    if len(ball_tracks.get('confidence_scores', [])) > 0:
        avg_conf = sum(ball_tracks['confidence_scores']) / len(ball_tracks['confidence_scores'])
        max_conf = max(ball_tracks['confidence_scores'])
        print(f"Avg Confidence: {avg_conf:.3f}")
        print(f"Max Confidence: {max_conf:.3f}")
    
    total_frames = results['metadata'].get('frame_count', 0)
    coverage = (trajectory_points / total_frames * 100) if total_frames > 0 else 0
    print(f"Coverage: {coverage:.1f}% of frames")
    
    if raw_detections > 50:
        print("\n✅ PASS: Ball tracked successfully")
    elif raw_detections > 10:
        print("\n⚠️  PARTIAL: Ball partially tracked")
    else:
        print("\n❌ FAIL: Insufficient ball tracking")
    
    # Test 4: Movement Analysis
    print("\n" + "=" * 80)
    print("TEST 4: PLAYER MOVEMENT ANALYSIS")
    print("=" * 80)
    
    for player in player_tracks:
        if len(player.get('positions', [])) < 2:
            continue
        
        positions = player['positions']
        
        # Calculate movement statistics
        distances = []
        for i in range(len(positions) - 1):
            x1, y1 = positions[i]
            x2, y2 = positions[i + 1]
            dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            distances.append(dist)
        
        if distances:
            total_distance = sum(distances)
            avg_speed = np.mean(distances)
            max_speed = max(distances)
            
            print(f"\nPlayer {player['player_id']} ({player.get('team', '?')}):")
            print(f"  Total Distance Moved: {total_distance:.1f} pixels")
            print(f"  Avg Movement per Frame: {avg_speed:.2f} pixels")
            print(f"  Max Movement: {max_speed:.2f} pixels")
            print(f"  Active Frames: {len(positions)}")
    
    print("\n✅ Movement analysis complete")
    
    # Test 5: Shot Detection (Basic)
    print("\n" + "=" * 80)
    print("TEST 5: SHOT TYPE DETECTION (EXPERIMENTAL)")
    print("=" * 80)
    
    shot_detector = ShotTypeDetector()
    shots = shot_detector.analyze_rally(player_tracks, ball_tracks, field_info)
    
    print(f"Total Shots Detected: {len(shots)}")
    
    # Count shots by type
    shot_counts = {}
    for shot in shots:
        shot_type = shot['shot_type']
        shot_counts[shot_type] = shot_counts.get(shot_type, 0) + 1
    
    print("\nShot Type Distribution:")
    for shot_type, count in sorted(shot_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {shot_type}: {count}")
    
    if len(shots) > 0:
        print("\n✅ Shot detection functional (experimental)")
        
        # Show first few shots
        print("\nSample Shots:")
        for i, shot in enumerate(shots[:5]):
            print(f"\n  Shot {i+1}:")
            print(f"    Frame: {shot['frame']}")
            print(f"    Player: {shot['player_id']} (Team {shot['team']})")
            print(f"    Type: {shot['shot_type']}")
            print(f"    Ball Speed: {shot['ball_speed']:.1f} px/s")
    else:
        print("\n⚠️  No shots detected (needs improvement)")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\n📊 Video Metadata:")
    print(f"   Resolution: {results['metadata']['width']}x{results['metadata']['height']}")
    print(f"   FPS: {results['metadata']['fps']:.2f}")
    print(f"   Duration: {results['metadata']['duration']:.2f}s")
    print(f"   Total Frames: {results['metadata']['frame_count']}")
    
    print(f"\n🎯 Detection Results:")
    print(f"   Field Confidence: {field_confidence:.2f}")
    print(f"   Players Tracked: {len(player_tracks)}")
    print(f"   Ball Detection Coverage: {coverage:.1f}%")
    print(f"   Shots Detected: {len(shots)}")
    
    # Overall assessment
    print("\n" + "=" * 80)
    tests_passed = 0
    if field_confidence > 0.5 and corners_detected >= 4:
        tests_passed += 1
    if len(player_tracks) >= 2:
        tests_passed += 1
    if raw_detections > 50:
        tests_passed += 1
    
    print(f"Tests Passed: {tests_passed}/3 core tests")
    
    if tests_passed == 3:
        print("\n✅ ALL CORE TESTS PASSED - System is working well!")
    elif tests_passed >= 2:
        print("\n⚠️  MOST TESTS PASSED - System functional with some limitations")
    else:
        print("\n❌ INSUFFICIENT PERFORMANCE - System needs improvement")
    
    print("\n" + "=" * 80)
    
    # Ask if user wants visualization
    print("\n📺 To generate a visualization video with annotations, run:")
    print(f"   python -c \"from examples.comprehensive_video_test import visualize_analysis; "
          f"visualize_analysis('{video_path}', results, 'output_annotated.mp4')\"")
    
    return results


def main():
    """Main function."""
    if len(sys.argv) < 2:
        print("\nUsage: python examples/comprehensive_video_test.py <path_to_video>")
        print("\nExample:")
        print("  python examples/comprehensive_video_test.py /path/to/padel_match.mp4")
        sys.exit(1)
    
    video_path = Path(sys.argv[1])
    results = test_video_analysis(video_path)


if __name__ == "__main__":
    main()
