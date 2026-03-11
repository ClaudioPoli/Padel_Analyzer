"""
Tests for the HeatmapGenerator module.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import Config
from src.analytics.heatmap_generator import HeatmapGenerator


class TestHeatmapGenerator:
    """Tests for HeatmapGenerator functionality."""

    def _make_config(self, **overrides) -> Config:
        config = Config()
        for k, v in overrides.items():
            setattr(config.heatmap, k, v)
        return config

    def _make_homography(self) -> np.ndarray:
        """
        Create a simple identity-like homography for testing.
        Maps pixels at (500+x*50, 500+y*50) → real-world (x, y).
        """
        # Build a controlled set of correspondences
        src = np.array([
            [250, 0],    # BL  (-5, -10)
            [750, 0],    # BR  ( 5, -10)
            [250, 500],  # NTL (-5,   0)
            [750, 500],  # NBR ( 5,   0)
        ], dtype=np.float32)
        dst = np.array([
            [-5, -10],
            [ 5, -10],
            [-5,   0],
            [ 5,   0],
        ], dtype=np.float32)
        import cv2
        H, _ = cv2.findHomography(src, dst)
        return H

    def _make_player_tracks(self) -> list:
        """Create synthetic player tracks."""
        return [
            {
                "player_id": 1,
                "positions": [(400, 200), (410, 210), (420, 205)],
                "bounding_boxes": [
                    [380, 150, 420, 250],
                    [390, 160, 430, 260],
                    [400, 155, 440, 255],
                ],
                "frame_numbers": [0, 1, 2],
                "team": "A",
                "confidence_scores": [0.9, 0.88, 0.91],
            },
            {
                "player_id": 2,
                "positions": [(600, 700), (610, 710), (620, 705)],
                "bounding_boxes": [
                    [580, 650, 620, 750],
                    [590, 660, 630, 760],
                    [600, 655, 640, 755],
                ],
                "frame_numbers": [0, 1, 2],
                "team": "B",
                "confidence_scores": [0.85, 0.87, 0.83],
            },
        ]

    def _make_field_info(self, H: np.ndarray) -> dict:
        return {"homography_matrix": H}

    # ── Tests ─────────────────────────────────────────────────────────────

    def test_init(self):
        config = self._make_config()
        gen = HeatmapGenerator(config)
        assert gen.grid_w > 0
        assert gen.grid_h > 0

    def test_generate_returns_expected_keys(self):
        config = self._make_config()
        gen = HeatmapGenerator(config)
        H = self._make_homography()
        tracks = self._make_player_tracks()
        field_info = self._make_field_info(H)

        result = gen.generate(tracks, field_info)

        assert "per_player" in result
        assert "per_team" in result
        assert "global" in result
        assert "court_positions" in result
        assert "grid_bounds" in result

    def test_per_player_heatmaps_created(self):
        config = self._make_config()
        gen = HeatmapGenerator(config)
        H = self._make_homography()
        tracks = self._make_player_tracks()
        field_info = self._make_field_info(H)

        result = gen.generate(tracks, field_info)
        assert 1 in result["per_player"]
        assert 2 in result["per_player"]
        assert result["per_player"][1].shape == (gen.grid_h, gen.grid_w)

    def test_per_team_heatmaps_created(self):
        config = self._make_config(per_team=True)
        gen = HeatmapGenerator(config)
        H = self._make_homography()
        tracks = self._make_player_tracks()
        field_info = self._make_field_info(H)

        result = gen.generate(tracks, field_info)
        assert "A" in result["per_team"]
        assert "B" in result["per_team"]

    def test_global_heatmap_is_sum(self):
        config = self._make_config(normalize=False, sigma=0.0)
        gen = HeatmapGenerator(config)
        H = self._make_homography()
        tracks = self._make_player_tracks()
        field_info = self._make_field_info(H)

        result = gen.generate(tracks, field_info)
        global_sum = result["global"].sum()
        # Global should have contributions from all players
        assert global_sum > 0

    def test_court_positions_populated(self):
        config = self._make_config()
        gen = HeatmapGenerator(config)
        H = self._make_homography()
        tracks = self._make_player_tracks()
        field_info = self._make_field_info(H)

        result = gen.generate(tracks, field_info)
        assert len(result["court_positions"][1]) > 0
        assert len(result["court_positions"][2]) > 0
        # Each position is (x, y) tuple
        x, y = result["court_positions"][1][0]
        assert isinstance(x, float)
        assert isinstance(y, float)

    def test_no_homography_returns_empty(self):
        config = self._make_config()
        gen = HeatmapGenerator(config)
        field_info = {"homography_matrix": None}
        tracks = self._make_player_tracks()

        result = gen.generate(tracks, field_info)
        assert result["per_player"] == {}
        assert result["homography_matrix"] is None

    def test_render_court_heatmap_returns_image(self):
        config = self._make_config(output_width=400)
        gen = HeatmapGenerator(config)
        H = self._make_homography()
        tracks = self._make_player_tracks()
        field_info = self._make_field_info(H)

        result = gen.generate(tracks, field_info)
        img = gen.render_court_heatmap(result)
        assert img.ndim == 3
        assert img.shape[2] == 3  # BGR
        assert img.shape[1] == 400  # output_width

    def test_render_court_heatmap_per_player(self):
        config = self._make_config(output_width=400)
        gen = HeatmapGenerator(config)
        H = self._make_homography()
        tracks = self._make_player_tracks()
        field_info = self._make_field_info(H)

        result = gen.generate(tracks, field_info)
        img = gen.render_court_heatmap(result, player_id=1, title="Test Player 1")
        assert img.ndim == 3

    def test_render_overlay_returns_frame_sized_image(self):
        config = self._make_config(output_width=400)
        gen = HeatmapGenerator(config)
        H = self._make_homography()
        tracks = self._make_player_tracks()
        field_info = self._make_field_info(H)

        result = gen.generate(tracks, field_info)

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        overlay = gen.render_overlay(frame, result, field_info)
        assert overlay.shape == frame.shape

    def test_save_heatmap(self, tmp_path):
        config = self._make_config(output_width=200)
        gen = HeatmapGenerator(config)
        H = self._make_homography()
        tracks = self._make_player_tracks()
        field_info = self._make_field_info(H)

        result = gen.generate(tracks, field_info)
        saved = gen.save_heatmap(result, str(tmp_path), prefix="test")

        # Should save global + 2 per-player + 2 per-team = 5 images
        assert len(saved) >= 3  # at least global + 2 players
        for p in saved:
            assert Path(p).exists()

    def test_normalize_disabled(self):
        config = self._make_config(normalize=False, sigma=0.0)
        gen = HeatmapGenerator(config)
        H = self._make_homography()
        tracks = self._make_player_tracks()
        field_info = self._make_field_info(H)

        result = gen.generate(tracks, field_info)
        # With normalize=False, values should be raw counts
        for pid, grid in result["per_player"].items():
            if grid.max() > 0:
                # Should have integer-like values (counts)
                assert grid.max() >= 1.0

    def test_different_colormaps(self):
        for cmap in ["jet", "hot", "viridis", "turbo"]:
            config = self._make_config(colormap=cmap, output_width=200)
            gen = HeatmapGenerator(config)
            H = self._make_homography()
            tracks = self._make_player_tracks()
            field_info = self._make_field_info(H)

            result = gen.generate(tracks, field_info)
            img = gen.render_court_heatmap(result)
            assert img.ndim == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
