#!/usr/bin/env python3
"""
Test YOLOv8n standard (NON fine-tuned) sui video personali.
"""

import sys
from pathlib import Path
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src import PadelAnalyzer
from src.utils.config import Config
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def test_video(analyzer, video_path: Path):
    """Test su singolo video."""
    logger.info(f"\n{'='*80}")
    logger.info(f"🎬 Video: {video_path.name}")
    logger.info(f"{'='*80}")
    
    try:
        start_time = time.time()
        results = analyzer.analyze_video(str(video_path))
        elapsed = time.time() - start_time
        
        # Estrai statistiche
        player_tracks = results.get("player_tracks", [])
        ball_tracks = results.get("ball_tracks", {})
        field_info = results.get("field_info", {})
        metadata = results.get("metadata", {})
        
        num_frames = metadata.get("total_frames", 0)
        fps = metadata.get("fps", 0)
        duration = num_frames / fps if fps > 0 else 0
        
        logger.info(f"\n📊 Risultati:")
        logger.info(f"   ⏱️  Tempo elaborazione: {elapsed:.1f}s")
        logger.info(f"   🎞️  Frame totali: {num_frames}")
        logger.info(f"   ⏱️  Durata video: {duration:.1f}s")
        logger.info(f"   🚀 FPS processing: {num_frames/elapsed:.1f}")
        
        logger.info(f"\n👥 Giocatori rilevati: {len(player_tracks)}")
        for i, track in enumerate(player_tracks):
            frames = len(track.get("frame_numbers", []))
            team = track.get("team", "Unknown")
            keypoints_count = sum(1 for kp in track.get("keypoints_sequence", []) if kp is not None)
            logger.info(f"   Player {i+1}: {frames} frames, team={team}, keypoints={keypoints_count}")
        
        ball_positions = ball_tracks.get("positions", [])
        logger.info(f"\n⚽ Pallina: {len(ball_positions)} posizioni rilevate")
        
        field_conf = field_info.get("confidence", 0.0)
        logger.info(f"\n🎾 Campo: confidence={field_conf:.2f}")
        
        return {
            "success": True,
            "video": video_path.name,
            "players": len(player_tracks),
            "ball_detections": len(ball_positions),
            "elapsed": elapsed,
            "fps_processing": num_frames/elapsed if elapsed > 0 else 0
        }
        
    except Exception as e:
        logger.error(f"❌ Errore: {e}")
        return {
            "success": False,
            "video": video_path.name,
            "error": str(e)
        }


def main():
    logger.info("="*80)
    logger.info("🧪 TEST YOLOv8n STANDARD (NON fine-tuned)")
    logger.info("="*80)
    
    # Setup config con YOLOv8n standard
    config = Config()
    config.model.player_model = "yolov8n.pt"  # Standard pre-trained
    config.model.ball_model = "yolov8n.pt"    # Standard pre-trained
    config.model.use_pose = True
    config.model.pose_model = "yolov8n-pose.pt"
    config.tracking.extract_keypoints = True
    config.model.device = "mps"  # M1 Pro
    
    logger.info("\n⚙️  Configurazione:")
    logger.info(f"   Detection model: {config.model.player_model}")
    logger.info(f"   Pose model: {config.model.pose_model}")
    logger.info(f"   Device: {config.model.device}")
    
    # Crea analyzer
    analyzer = PadelAnalyzer(config)
    
    # Trova tutti i video
    video_dir = project_root / "data" / "personal"
    video_files = []
    for ext in [".mp4", ".MP4", ".mov", ".MOV"]:
        video_files.extend(video_dir.glob(f"*{ext}"))
    
    video_files = sorted(video_files)
    
    logger.info(f"\n📁 Trovati {len(video_files)} video in data/personal/")
    
    # Testa tutti i video
    results = []
    for video_path in video_files:
        result = test_video(analyzer, video_path)
        results.append(result)
    
    # Riepilogo finale
    logger.info(f"\n{'='*80}")
    logger.info("📊 RIEPILOGO COMPLETO")
    logger.info(f"{'='*80}")
    
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]
    
    logger.info(f"\n✅ Successi: {len(successful)}/{len(results)}")
    logger.info(f"❌ Falliti: {len(failed)}/{len(results)}")
    
    if successful:
        logger.info(f"\n📈 Statistiche:")
        avg_players = sum(r["players"] for r in successful) / len(successful)
        avg_ball = sum(r["ball_detections"] for r in successful) / len(successful)
        avg_fps = sum(r["fps_processing"] for r in successful) / len(successful)
        logger.info(f"   Media giocatori rilevati: {avg_players:.1f}")
        logger.info(f"   Media rilevamenti pallina: {avg_ball:.1f}")
        logger.info(f"   FPS processing medio: {avg_fps:.1f}")
    
    if failed:
        logger.info(f"\n❌ Video falliti:")
        for r in failed:
            logger.info(f"   {r['video']}: {r['error']}")


if __name__ == "__main__":
    main()
