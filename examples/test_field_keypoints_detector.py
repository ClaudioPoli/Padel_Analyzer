"""
Test script for the KeypointFieldDetector.

Processes the first 30 seconds of each video in data/personal/,
runs field keypoint detection with geometric interpolation,
and saves annotated output videos + summary images.

Usage:
    python test_field_keypoints_detector.py
    python test_field_keypoints_detector.py --video data/personal/rally.mp4
    python test_field_keypoints_detector.py --max-seconds 10
"""

import sys
import argparse
import time
from pathlib import Path

import cv2
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.config import Config
from src.detection.keypoint_field_detector import KeypointFieldDetector


# ──────────────────────────────────────────────────────────────────────────
# Setup
# ──────────────────────────────────────────────────────────────────────────

VIDEO_DIR = Path("data/personal")
OUTPUT_DIR = Path("data/output/field_keypoints_test")
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}


def find_videos(video_dir: Path) -> list:
    """Find all video files in a directory (case-insensitive extension)."""
    videos = []
    for f in sorted(video_dir.iterdir()):
        if f.suffix.lower() in VIDEO_EXTENSIONS:
            videos.append(f)
    return videos


def create_config() -> Config:
    """Create config optimized for testing."""
    config = Config()
    config.field_keypoints.enabled = True
    config.field_keypoints.model_path = "models/field_skeleton_yolo11m_pose5/weights/best.pt"
    config.field_keypoints.min_keypoint_confidence = 0.5
    config.field_keypoints.min_detection_confidence = 0.25
    config.field_keypoints.interpolate_missing = True
    config.field_keypoints.temporal_smoothing = True
    config.field_keypoints.smoothing_window = 5
    config.model.device = "auto"
    return config


# ──────────────────────────────────────────────────────────────────────────
# Per-video processing
# ──────────────────────────────────────────────────────────────────────────

def process_video(
    video_path: Path,
    detector: KeypointFieldDetector,
    output_dir: Path,
    max_seconds: float = 30.0,
    save_video: bool = True,
    save_snapshots: bool = True,
    snapshot_interval: int = 5,
):
    """
    Process a single video: detect field keypoints with temporal smoothing,
    save annotated video + snapshot images.

    Args:
        video_path: Path to input video
        detector: Initialized KeypointFieldDetector
        output_dir: Directory for output files
        max_seconds: Max duration to process
        save_video: Whether to save annotated video
        save_snapshots: Whether to save annotated snapshots every N seconds
        snapshot_interval: Seconds between snapshots
    """
    stem = video_path.stem
    print(f"\n{'='*70}")
    print(f"Processing: {video_path.name}")
    print(f"{'='*70}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  ✗ Could not open video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0

    max_frames = int(min(max_seconds * fps, total_frames))
    snapshot_every = int(snapshot_interval * fps) if fps > 0 else 150

    print(f"  Resolution: {width}x{height}, FPS: {fps:.1f}, Duration: {duration:.1f}s")
    print(f"  Processing first {max_seconds}s → {max_frames} frames")

    # Setup video writer
    writer = None
    if save_video:
        out_video = output_dir / f"{stem}_annotated.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_video), fourcc, fps, (width, height))
        print(f"  Output video: {out_video}")

    # Snapshots directory
    snap_dir = output_dir / f"{stem}_snapshots"
    if save_snapshots:
        snap_dir.mkdir(parents=True, exist_ok=True)

    # Processing stats
    prev_field_info = None
    frame_idx = 0
    stats = {
        "total_frames": 0,
        "detected_frames": 0,
        "reliable_counts": [],
        "interpolated_counts": [],
        "confidences": [],
        "per_kpt_reliable_count": np.zeros(10, dtype=int),
        "per_kpt_interpolated_count": np.zeros(10, dtype=int),
    }
    t_start = time.time()

    while frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect with temporal smoothing
        field_info = detector.detect_in_video_stream(frame, prev_field_info)
        prev_field_info = field_info

        stats["total_frames"] += 1

        detected = field_info.get("detection_confidence", 0) > 0
        if detected:
            stats["detected_frames"] += 1
            stats["reliable_counts"].append(field_info.get("num_reliable", 0))
            stats["interpolated_counts"].append(field_info.get("num_interpolated", 0))
            stats["confidences"].append(field_info.get("detection_confidence", 0))

            # Per-keypoint stats
            status_list = field_info.get("keypoints_status", [])
            for i, s in enumerate(status_list):
                if s == "reliable":
                    stats["per_kpt_reliable_count"][i] += 1
                elif s == "interpolated":
                    stats["per_kpt_interpolated_count"][i] += 1

        # Draw annotations
        annotated = detector.draw_field_info(frame, field_info)

        # Add frame counter
        cv2.putText(
            annotated,
            f"Frame {frame_idx}/{max_frames}  ({frame_idx/fps:.1f}s)",
            (10, height - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1,
        )

        # Write video
        if writer is not None:
            writer.write(annotated)

        # Save snapshot
        if save_snapshots and frame_idx % snapshot_every == 0:
            snap_path = snap_dir / f"frame_{frame_idx:06d}.jpg"
            cv2.imwrite(str(snap_path), annotated)

        frame_idx += 1

        # Progress every 5 seconds of video
        if frame_idx % int(5 * fps) == 0:
            elapsed = time.time() - t_start
            speed = frame_idx / elapsed if elapsed > 0 else 0
            print(
                f"  [{frame_idx:5d}/{max_frames}] "
                f"{frame_idx/fps:.1f}s processed, "
                f"{speed:.1f} fps, "
                f"det_rate={stats['detected_frames']/stats['total_frames']*100:.0f}%"
            )

    cap.release()
    if writer is not None:
        writer.release()

    elapsed = time.time() - t_start
    
    # Save a summary image for the last detected frame
    if save_snapshots and prev_field_info and prev_field_info.get("detection_confidence", 0) > 0:
        # Re-open video to get a "best" frame (middle of the clip)
        cap2 = cv2.VideoCapture(str(video_path))
        mid = min(max_frames // 2, total_frames - 1)
        cap2.set(cv2.CAP_PROP_POS_FRAMES, mid)
        ret2, mid_frame = cap2.read()
        cap2.release()
        if ret2:
            mid_info = detector.detect_in_frame(mid_frame)
            if mid_info is not None:
                summary = detector.draw_field_info(mid_frame, mid_info)
                cv2.imwrite(str(output_dir / f"{stem}_summary.jpg"), summary)

    # ── Print stats ──
    print(f"\n  Results for {stem}:")
    print(f"    Frames processed: {stats['total_frames']}")
    print(f"    Detection rate:   {stats['detected_frames']}/{stats['total_frames']} "
          f"({stats['detected_frames']/max(stats['total_frames'],1)*100:.1f}%)")
    print(f"    Processing speed: {stats['total_frames']/max(elapsed,0.001):.1f} fps "
          f"(elapsed: {elapsed:.1f}s)")

    if stats["confidences"]:
        print(f"    Avg confidence:   {np.mean(stats['confidences']):.3f}")
        print(f"    Avg reliable:     {np.mean(stats['reliable_counts']):.1f}/10")
        print(f"    Avg interpolated: {np.mean(stats['interpolated_counts']):.1f}")

    # Per-keypoint breakdown
    kpt_names = ["BL", "BR", "FL", "FR", "NTL", "NTR", "NBL", "NBR", "SL", "ST"]
    print(f"\n    Per-keypoint breakdown (over {stats['detected_frames']} detected frames):")
    print(f"    {'KP':<5} {'Reliable':>10} {'Interpolated':>13} {'Missing':>10}")
    print(f"    {'─'*40}")
    for i, name in enumerate(kpt_names):
        rel = stats["per_kpt_reliable_count"][i]
        interp = stats["per_kpt_interpolated_count"][i]
        miss = stats["detected_frames"] - rel - interp
        print(f"    {name:<5} {rel:>10} {interp:>13} {miss:>10}")

    return stats


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Test KeypointFieldDetector")
    parser.add_argument("--video", type=str, default=None,
                        help="Path to a single video (default: all videos in data/personal/)")
    parser.add_argument("--max-seconds", type=float, default=30.0,
                        help="Max seconds to process per video (default: 30)")
    parser.add_argument("--no-video-output", action="store_true",
                        help="Skip saving annotated videos (faster)")
    parser.add_argument("--snapshot-interval", type=int, default=5,
                        help="Seconds between snapshot images (default: 5)")
    parser.add_argument("--confidence", type=float, default=0.5,
                        help="Min keypoint confidence (default: 0.5)")
    args = parser.parse_args()

    # Setup
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config = create_config()
    config.field_keypoints.min_keypoint_confidence = args.confidence

    # Find videos
    if args.video:
        videos = [Path(args.video)]
        if not videos[0].exists():
            print(f"Video not found: {args.video}")
            sys.exit(1)
    else:
        videos = find_videos(VIDEO_DIR)
        if not videos:
            print(f"No videos found in {VIDEO_DIR}")
            sys.exit(1)

    print(f"Found {len(videos)} video(s) to process")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Max duration per video: {args.max_seconds}s")
    print(f"Min keypoint confidence: {args.confidence}")

    # Initialize detector (model loads lazily on first use)
    detector = KeypointFieldDetector(config)

    # Process each video
    all_stats = {}
    for vp in videos:
        try:
            stats = process_video(
                video_path=vp,
                detector=detector,
                output_dir=OUTPUT_DIR,
                max_seconds=args.max_seconds,
                save_video=not args.no_video_output,
                save_snapshots=True,
                snapshot_interval=args.snapshot_interval,
            )
            all_stats[vp.name] = stats
        except Exception as e:
            print(f"\n  ✗ Error processing {vp.name}: {e}")
            import traceback
            traceback.print_exc()

    # ── Global summary ──
    print(f"\n{'='*70}")
    print("GLOBAL SUMMARY")
    print(f"{'='*70}")
    for name, st in all_stats.items():
        if st is None:
            print(f"  {name}: FAILED")
            continue
        det_rate = st["detected_frames"] / max(st["total_frames"], 1) * 100
        avg_conf = np.mean(st["confidences"]) if st["confidences"] else 0
        avg_rel = np.mean(st["reliable_counts"]) if st["reliable_counts"] else 0
        print(f"  {name:<40} det={det_rate:5.1f}%  conf={avg_conf:.3f}  reliable={avg_rel:.1f}/10")

    print(f"\nOutputs saved to: {OUTPUT_DIR}")
    print("Done!")


if __name__ == "__main__":
    main()
