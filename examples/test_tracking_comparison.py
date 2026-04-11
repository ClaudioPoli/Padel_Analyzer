#!/usr/bin/env python3
"""
Comparative test: ByteTrack-only vs PlayerIdentityManager tracking.

Processes padel videos and produces:
  1. Two annotated output videos side-by-side (ByteTrack-only | Stabilized)
  2. Per-video JSON report with swap/consistency metrics
  3. Console summary highlighting improvements

Usage:
    python examples/test_tracking_comparison.py                       # all videos, 15s
    python examples/test_tracking_comparison.py --video rally.mp4     # single video
    python examples/test_tracking_comparison.py --duration 30         # 30 seconds
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
from collections import defaultdict, Counter

import cv2
import numpy as np

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import Config
from src.tracking.player_tracker import PlayerTracker, PlayerIdentityManager


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

# Distinct colours per stable ID (BGR)
STABLE_COLORS = {
    1: (255, 120, 0),   # Blue-ish   (Team A)
    2: (255, 220, 0),   # Cyan-ish   (Team A)
    3: (0, 0, 255),     # Red        (Team B)
    4: (0, 120, 255),   # Orange     (Team B)
}
# Random-ish color cycle when IDs are raw ByteTrack values
_BT_PALETTE = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 255), (255, 128, 0),
]


def _color_for_bt(bt_id: int) -> tuple:
    return _BT_PALETTE[bt_id % len(_BT_PALETTE)]


def draw_detections(frame, detections, use_stable_id: bool, frame_idx: int):
    """Draw bounding boxes + labels on *frame* (mutates in place)."""
    for det in detections:
        if use_stable_id:
            pid = det.get("stable_player_id")
            if pid is None:
                continue
            color = STABLE_COLORS.get(pid, (200, 200, 200))
            team = "A" if pid <= 2 else "B"
            label = f"P{pid} Team{team}"
        else:
            pid = det.get("track_id")
            if pid is None:
                continue
            color = _color_for_bt(pid)
            label = f"BT#{pid}"

        x1, y1, x2, y2 = det["bbox"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def compute_metrics(per_frame_ids, mode_label=""):
    """
    Given a list (one entry per frame) of sets-of-IDs, compute:
      - total_unique_ids:  how many distinct IDs appeared
      - id_switches:       frames where the set of IDs changed relative to previous
      - frames_with_4:     frames where exactly 4 players were tracked
      - continuity_score:  fraction of frames where the same 4 IDs persist
    """
    total_unique = set()
    for ids in per_frame_ids:
        total_unique.update(ids)

    id_switches = 0
    frames_with_4 = 0
    continuity_breaks = 0

    prev_ids = None
    reference_set = None  # first frame that has 4 players
    for ids in per_frame_ids:
        if len(ids) == 4:
            frames_with_4 += 1
            if reference_set is None:
                reference_set = ids
        if prev_ids is not None and ids != prev_ids:
            id_switches += 1
        if reference_set is not None and ids != reference_set and len(ids) >= 4:
            continuity_breaks += 1
        prev_ids = ids

    n = len(per_frame_ids)
    continuity_score = 1.0 - (continuity_breaks / max(1, n))

    return {
        "mode": mode_label,
        "total_frames": n,
        "total_unique_ids": len(total_unique),
        "unique_ids_list": sorted(total_unique),
        "id_switches": id_switches,
        "frames_with_4_players": frames_with_4,
        "frames_with_4_pct": round(frames_with_4 / max(1, n) * 100, 1),
        "continuity_score": round(continuity_score, 4),
    }


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def process_video(video_path: Path, output_dir: Path, config: Config,
                  duration_seconds: int = 15):
    print(f"\n{'=' * 78}")
    print(f"  VIDEO: {video_path.name}")
    print(f"{'=' * 78}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  ERROR: cannot open {video_path}")
        return None

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frames = min(fps * duration_seconds, total_frames)

    print(f"  Resolution: {width}x{height}  FPS: {fps}  Frames to process: {max_frames}")

    # Load frames ---------------------------------------------------------
    frames = []
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    print(f"  Loaded {len(frames)} frames")

    # Tracker (shared model, two passes) ----------------------------------
    tracker = PlayerTracker(config, use_pose_estimation=True)

    # =====================================================================
    # PASS 1 — ByteTrack only (no identity manager)
    # =====================================================================
    print("\n  [1/2] ByteTrack-only pass …")
    # Reset ByteTrack internal state by re-creating the model tracker
    tracker_bt = PlayerTracker(config, use_pose_estimation=True)

    bt_per_frame_dets: list[list[dict]] = []
    bt_per_frame_ids: list[set] = []

    t0 = time.time()
    for idx, frame in enumerate(frames):
        dets = tracker_bt.detect_players_in_frame(frame, idx, field_mask=None)
        bt_per_frame_dets.append(dets)
        ids = {d["track_id"] for d in dets if d.get("track_id") is not None}
        bt_per_frame_ids.append(ids)
        if (idx + 1) % 200 == 0:
            print(f"        {idx + 1}/{len(frames)}")
    bt_time = time.time() - t0
    print(f"        Done in {bt_time:.1f}s ({len(frames)/bt_time:.1f} fps)")

    # =====================================================================
    # PASS 2 — With PlayerIdentityManager
    # =====================================================================
    print("\n  [2/2] Stabilized (PlayerIdentityManager) pass …")
    tracker_stable = PlayerTracker(config, use_pose_estimation=True)
    identity_mgr = PlayerIdentityManager(max_players=4)

    st_per_frame_dets: list[list[dict]] = []
    st_per_frame_ids: list[set] = []

    t0 = time.time()
    for idx, frame in enumerate(frames):
        dets = tracker_stable.detect_players_in_frame(frame, idx, field_mask=None)
        dets = identity_mgr.update(frame, dets)
        st_per_frame_dets.append(dets)
        ids = {d["stable_player_id"] for d in dets
               if d.get("stable_player_id") is not None}
        st_per_frame_ids.append(ids)
        if (idx + 1) % 200 == 0:
            print(f"        {idx + 1}/{len(frames)}")
    st_time = time.time() - t0
    print(f"        Done in {st_time:.1f}s ({len(frames)/st_time:.1f} fps)")

    # =====================================================================
    # Metrics
    # =====================================================================
    bt_metrics = compute_metrics(bt_per_frame_ids, "ByteTrack-only")
    st_metrics = compute_metrics(st_per_frame_ids, "Stabilized")

    print("\n  ┌─────────────────────────────────┬────────────────┬────────────────┐")
    print("  │ Metric                          │  ByteTrack     │  Stabilized    │")
    print("  ├─────────────────────────────────┼────────────────┼────────────────┤")
    for key in ["total_unique_ids", "id_switches", "frames_with_4_pct",
                "continuity_score"]:
        bv = bt_metrics[key]
        sv = st_metrics[key]
        label = key.replace("_", " ").title()
        if key == "frames_with_4_pct":
            label = "Frames w/ 4 players (%)"
            bv = f"{bv}%"
            sv = f"{sv}%"
        print(f"  │ {label:<31} │ {str(bv):>14} │ {str(sv):>14} │")
    print("  └─────────────────────────────────┴────────────────┴────────────────┘")

    # Improvement summary
    switch_delta = bt_metrics["id_switches"] - st_metrics["id_switches"]
    id_delta = bt_metrics["total_unique_ids"] - st_metrics["total_unique_ids"]
    print(f"\n  ✦ ID switches reduced by {switch_delta} "
          f"({bt_metrics['id_switches']}→{st_metrics['id_switches']})")
    print(f"  ✦ Unique IDs: {bt_metrics['total_unique_ids']}→{st_metrics['total_unique_ids']} "
          f"(target: 4)")
    print(f"  ✦ Continuity: {bt_metrics['continuity_score']:.2%}→"
          f"{st_metrics['continuity_score']:.2%}")

    # =====================================================================
    # Output video: side-by-side comparison
    # =====================================================================
    side_w = min(width, 960)
    side_h = int(height * side_w / width)
    canvas_w = side_w * 2 + 4  # 4px divider
    canvas_h = side_h + 40     # room for header

    out_path = output_dir / f"{video_path.stem}_tracking_comparison.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(out_path), fourcc, fps, (canvas_w, canvas_h))

    print(f"\n  Writing comparison video → {out_path.name}")
    for idx in range(len(frames)):
        # Left panel: ByteTrack only
        left = frames[idx].copy()
        draw_detections(left, bt_per_frame_dets[idx], use_stable_id=False, frame_idx=idx)
        left = cv2.resize(left, (side_w, side_h))

        # Right panel: Stabilized
        right = frames[idx].copy()
        draw_detections(right, st_per_frame_dets[idx], use_stable_id=True, frame_idx=idx)
        right = cv2.resize(right, (side_w, side_h))

        # Canvas
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        # Header
        cv2.putText(canvas, "ByteTrack Only", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 180, 255), 2)
        cv2.putText(canvas, "Stabilized (PlayerIdentityManager)", (side_w + 14, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)
        # Panels
        y0 = 40
        canvas[y0:y0 + side_h, :side_w] = left
        canvas[y0:y0 + side_h, side_w + 4:] = right
        # Divider
        canvas[y0:y0 + side_h, side_w:side_w + 4] = (80, 80, 80)
        # Frame counter
        cv2.putText(canvas, f"Frame {idx}/{len(frames)}", (canvas_w - 200, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        out.write(canvas)

    out.release()
    print(f"  ✓ Comparison video saved ({len(frames)} frames)")

    # =====================================================================
    # Also write a stabilized-only full-res video (for detailed review)
    # =====================================================================
    stable_path = output_dir / f"{video_path.stem}_stabilized.mp4"
    out2 = cv2.VideoWriter(str(stable_path), fourcc, fps, (width, height))
    for idx in range(len(frames)):
        annotated = frames[idx].copy()
        draw_detections(annotated, st_per_frame_dets[idx], use_stable_id=True, frame_idx=idx)
        # Overlay info bar
        cv2.rectangle(annotated, (0, 0), (width, 32), (0, 0, 0), -1)
        visible = {d.get("stable_player_id") for d in st_per_frame_dets[idx]
                   if d.get("stable_player_id") is not None}
        info = (f"Frame {idx}  |  Players: {sorted(visible)}  |  "
                f"Team A: P1,P2  Team B: P3,P4")
        cv2.putText(annotated, info, (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        out2.write(annotated)
    out2.release()
    print(f"  ✓ Stabilized full-res video saved → {stable_path.name}")

    # =====================================================================
    # JSON report
    # =====================================================================
    report = {
        "video": video_path.name,
        "duration_s": duration_seconds,
        "frames_processed": len(frames),
        "fps": fps,
        "resolution": f"{width}x{height}",
        "bytetrack_only": bt_metrics,
        "stabilized": st_metrics,
        "improvement": {
            "id_switches_reduced": switch_delta,
            "unique_ids_reduced": id_delta,
            "continuity_improvement": round(
                st_metrics["continuity_score"] - bt_metrics["continuity_score"], 4),
        },
    }
    report_path = output_dir / f"{video_path.stem}_tracking_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"  ✓ Report saved → {report_path.name}")

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compare ByteTrack vs Stabilized player tracking on padel videos."
    )
    parser.add_argument("--video", type=str, default=None,
                        help="Process a single video file name (from data/personal/)")
    parser.add_argument("--duration", type=int, default=15,
                        help="Seconds to process per video (default: 15)")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    personal_dir = project_root / "data" / "personal"
    output_dir = project_root / "data" / "output" / "tracking_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Config
    config_path = project_root / "config.example.json"
    config = Config.from_file(str(config_path)) if config_path.exists() else Config()

    # Find videos
    if args.video:
        video_files = [personal_dir / args.video]
        if not video_files[0].exists():
            print(f"ERROR: Video not found: {video_files[0]}")
            return
    else:
        exts = [".mp4", ".mov", ".avi", ".mkv"]
        video_files = sorted(
            f for f in personal_dir.iterdir()
            if f.suffix.lower() in exts and f.is_file()
        )

    if not video_files:
        print(f"No videos found in {personal_dir}")
        return

    print(f"Found {len(video_files)} video(s). Output → {output_dir}")

    reports = []
    for i, vp in enumerate(video_files, 1):
        print(f"\n[{i}/{len(video_files)}]", end="")
        try:
            r = process_video(vp, output_dir, config, args.duration)
            if r:
                reports.append(r)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    # =====================================================================
    # Overall summary
    # =====================================================================
    if reports:
        print(f"\n{'=' * 78}")
        print("  OVERALL SUMMARY")
        print(f"{'=' * 78}")
        total_bt_switches = sum(r["bytetrack_only"]["id_switches"] for r in reports)
        total_st_switches = sum(r["stabilized"]["id_switches"] for r in reports)
        avg_bt_cont = np.mean([r["bytetrack_only"]["continuity_score"] for r in reports])
        avg_st_cont = np.mean([r["stabilized"]["continuity_score"] for r in reports])
        avg_bt_ids = np.mean([r["bytetrack_only"]["total_unique_ids"] for r in reports])
        avg_st_ids = np.mean([r["stabilized"]["total_unique_ids"] for r in reports])

        print(f"  Videos processed:         {len(reports)}")
        print(f"  Total ID switches:        {total_bt_switches} → {total_st_switches} "
              f"(−{total_bt_switches - total_st_switches})")
        print(f"  Avg unique IDs:           {avg_bt_ids:.1f} → {avg_st_ids:.1f} (target: 4)")
        print(f"  Avg continuity:           {avg_bt_cont:.2%} → {avg_st_cont:.2%}")
        print(f"\n  Output directory: {output_dir}")


if __name__ == "__main__":
    main()
