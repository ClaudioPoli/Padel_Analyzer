"""
Example: Generate player heatmaps from a padel match video.

This script demonstrates the full pipeline:
1. Load video
2. Detect field keypoints
3. Track players
4. Generate heatmaps (per-player, per-team, global)
5. Save heatmap images and overlay on a sample frame

Usage:
    python examples/test_heatmap.py <video_path> [--output_dir data/output/heatmaps]
"""

import sys
import argparse
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import Config
from src.video.video_loader import VideoLoader
from src.detection.keypoint_field_detector import KeypointFieldDetector
from src.tracking.player_tracker import PlayerTracker
from src.analytics.heatmap_generator import HeatmapGenerator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Generate player heatmaps")
    parser.add_argument("video_path", help="Path to padel match video")
    parser.add_argument(
        "--output_dir",
        default="data/output/heatmaps",
        help="Directory to save heatmap images",
    )
    parser.add_argument(
        "--sigma", type=float, default=0.5,
        help="Gaussian smoothing sigma in meters (default: 0.5)",
    )
    parser.add_argument(
        "--colormap", default="jet",
        choices=["jet", "hot", "inferno", "magma", "viridis", "plasma", "turbo"],
        help="Colormap for heatmap rendering",
    )
    args = parser.parse_args()

    video_path = Path(args.video_path)
    if not video_path.exists():
        logger.error(f"Video not found: {video_path}")
        sys.exit(1)

    output_dir = Path(args.output_dir)

    # ── Config ────────────────────────────────────────────────────────────
    config = Config()
    config.heatmap.sigma = args.sigma
    config.heatmap.colormap = args.colormap
    config.heatmap.output_width = 800

    # ── Step 1: Load video ────────────────────────────────────────────────
    logger.info(f"Loading video: {video_path}")
    video_loader = VideoLoader(config)
    video_data = video_loader.load(video_path)
    metadata = video_data.get("metadata", {})
    logger.info(
        f"Video: {metadata.get('width')}x{metadata.get('height')}, "
        f"{metadata.get('fps'):.1f} fps, {metadata.get('frame_count')} frames"
    )

    try:
        # ── Step 2: Detect field keypoints ────────────────────────────────
        logger.info("Detecting field keypoints...")
        field_detector = KeypointFieldDetector(config)
        field_info = field_detector.detect(video_data)

        if field_info.get("homography_matrix") is None:
            logger.error(
                "Field detection failed – no homography matrix computed. "
                "Cannot generate heatmaps."
            )
            sys.exit(1)

        logger.info(
            f"Field detected: confidence={field_info['detection_confidence']:.3f}, "
            f"reliable_kpts={field_info['num_reliable']}/10"
        )

        # ── Step 3: Track players ─────────────────────────────────────────
        logger.info("Tracking players...")
        player_tracker = PlayerTracker(config)
        player_tracks = player_tracker.track(video_data, field_info)
        logger.info(f"Tracked {len(player_tracks)} players")

        for track in player_tracks:
            pid = track["player_id"]
            team = track.get("team", "?")
            npos = len(track["positions"])
            logger.info(f"  Player {pid} (team {team}): {npos} detections")

        # ── Step 4: Generate heatmaps ─────────────────────────────────────
        logger.info("Generating heatmaps...")
        heatmap_gen = HeatmapGenerator(config)
        heatmap_data = heatmap_gen.generate(player_tracks, field_info)

        # ── Step 5: Save heatmap images ───────────────────────────────────
        saved_files = heatmap_gen.save_heatmap(
            heatmap_data,
            output_dir=str(output_dir),
            prefix=video_path.stem,
        )
        for f in saved_files:
            logger.info(f"  Saved: {f}")

        # ── Step 6: Overlay on a sample frame ─────────────────────────────
        import cv2

        capture = video_data.get("capture")
        mid_frame = metadata.get("frame_count", 100) // 2
        capture.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
        ret, frame = capture.read()

        if ret:
            overlay = heatmap_gen.render_overlay(
                frame, heatmap_data, field_info
            )
            overlay_path = str(output_dir / f"{video_path.stem}_overlay.png")
            cv2.imwrite(overlay_path, overlay)
            logger.info(f"  Saved overlay: {overlay_path}")

        logger.info("Done!")

    finally:
        video_loader.release(video_data)


if __name__ == "__main__":
    main()
