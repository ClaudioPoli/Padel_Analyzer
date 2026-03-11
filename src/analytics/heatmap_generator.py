"""
Player heatmap generator for padel court analysis.

Uses player tracking data and field keypoint detection (homography) to map
player pixel positions onto real-world court coordinates, then builds a
2D density map (heatmap) showing where each player/team spends the most time.

Workflow:
    1. Extract player feet positions (bottom-center of bounding boxes) per frame
    2. Transform pixel positions to real-world court coordinates via homography
    3. Accumulate positions into a 2D histogram on the court plane
    4. Apply Gaussian smoothing for a continuous density surface
    5. Render as top-down court diagram or video-frame overlay
"""

from typing import Dict, Any, List, Optional, Tuple
import logging
import numpy as np
import cv2

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Padel court real-world dimensions (meters, origin at center)
# ──────────────────────────────────────────────────────────────────────────────
COURT_LENGTH = 20.0  # total length (Y axis: -10 to +10)
COURT_WIDTH = 10.0   # total width  (X axis: -5 to +5)
SERVICE_LINE_DIST = 3.0  # distance from back wall to service line


class HeatmapGenerator:
    """
    Generates player position heatmaps on the padel court.

    Uses the homography matrix from KeypointFieldDetector to project
    pixel-space player positions onto the real-world court plane, then
    accumulates a 2D density grid.

    Usage:
        generator = HeatmapGenerator(config)
        heatmap_data = generator.generate(player_tracks, field_info)
        court_img = generator.render_court_heatmap(heatmap_data, player_id=1)
        overlay = generator.render_overlay(frame, heatmap_data, field_info)
    """

    def __init__(self, config: Any):
        """
        Initialize the HeatmapGenerator.

        Args:
            config: Config object containing heatmap settings (config.heatmap).
        """
        self.config = config
        hm = config.heatmap

        # Court bounds in meters (with padding)
        self.pad = hm.court_padding
        self.x_min = -COURT_WIDTH / 2 - self.pad
        self.x_max = COURT_WIDTH / 2 + self.pad
        self.y_min = -COURT_LENGTH / 2 - self.pad
        self.y_max = COURT_LENGTH / 2 + self.pad

        # Grid resolution
        self.res = hm.resolution  # cells per meter
        self.grid_w = int((self.x_max - self.x_min) * self.res)
        self.grid_h = int((self.y_max - self.y_min) * self.res)

        logger.info(
            f"HeatmapGenerator initialized: grid {self.grid_w}x{self.grid_h}, "
            f"sigma={hm.sigma}m, padding={self.pad}m"
        )

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def generate(
        self,
        player_tracks: List[Dict[str, Any]],
        field_info: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Generate heatmap data from player tracks and field info.

        Args:
            player_tracks: List of player track dicts from PlayerTracker.
                Each track has 'player_id', 'positions', 'bounding_boxes',
                'frame_numbers', 'team'.
            field_info: Field info dict from KeypointFieldDetector.
                Must contain 'homography_matrix' (image → real-world).

        Returns:
            Dictionary with:
                - 'per_player': {player_id: 2D numpy array} individual heatmaps
                - 'per_team': {'A': 2D array, 'B': 2D array} team heatmaps
                - 'global': 2D numpy array – combined heatmap of all players
                - 'court_positions': {player_id: list of (x, y) real-world coords}
                - 'grid_bounds': dict with x_min/x_max/y_min/y_max
                - 'homography_matrix': the homography used
        """
        H = field_info.get("homography_matrix")
        if H is None:
            logger.warning(
                "No homography matrix in field_info – cannot generate heatmap"
            )
            return self._empty_result()

        hm_cfg = self.config.heatmap
        use_feet = hm_cfg.use_feet_position
        sigma_px = hm_cfg.sigma * self.res  # sigma in grid cells

        # ── 1. Transform player positions to court coordinates ───────────
        court_positions: Dict[int, List[Tuple[float, float]]] = {}

        for track in player_tracks:
            pid = track["player_id"]
            positions = track.get("positions", [])
            bboxes = track.get("bounding_boxes", [])
            court_pts: List[Tuple[float, float]] = []

            for i, pos in enumerate(positions):
                if use_feet and i < len(bboxes):
                    bbox = bboxes[i]
                    # Bottom-center of bounding box ≈ feet position
                    px = (bbox[0] + bbox[2]) / 2.0
                    py = float(bbox[3])  # y2 = bottom
                else:
                    px, py = pos

                # Apply homography: image pixel → real-world meters
                real_pt = self._pixel_to_court(H, px, py)
                if real_pt is not None:
                    court_pts.append(real_pt)

            court_positions[pid] = court_pts
            logger.debug(
                f"Player {pid}: {len(court_pts)}/{len(positions)} positions "
                f"projected onto court"
            )

        # ── 2. Build per-player heatmaps ─────────────────────────────────
        per_player: Dict[int, np.ndarray] = {}
        for pid, pts in court_positions.items():
            grid = self._accumulate(pts)
            if sigma_px > 0:
                grid = self._smooth(grid, sigma_px)
            if hm_cfg.normalize and grid.max() > 0:
                grid = grid / grid.max()
            per_player[pid] = grid

        # ── 3. Build per-team heatmaps ───────────────────────────────────
        per_team: Dict[str, np.ndarray] = {}
        if hm_cfg.per_team:
            team_grids: Dict[str, np.ndarray] = {}
            for track in player_tracks:
                team = track.get("team")
                if team is None:
                    continue
                pid = track["player_id"]
                if pid not in per_player:
                    continue
                if team not in team_grids:
                    team_grids[team] = np.zeros(
                        (self.grid_h, self.grid_w), dtype=np.float64
                    )
                # Accumulate raw (un-normalized) counts for team
                pts = court_positions.get(pid, [])
                team_grids[team] += self._accumulate(pts)

            for team, grid in team_grids.items():
                if sigma_px > 0:
                    grid = self._smooth(grid, sigma_px)
                if hm_cfg.normalize and grid.max() > 0:
                    grid = grid / grid.max()
                per_team[team] = grid

        # ── 4. Build global heatmap ──────────────────────────────────────
        all_pts: List[Tuple[float, float]] = []
        for pts in court_positions.values():
            all_pts.extend(pts)
        global_grid = self._accumulate(all_pts)
        if sigma_px > 0:
            global_grid = self._smooth(global_grid, sigma_px)
        if hm_cfg.normalize and global_grid.max() > 0:
            global_grid = global_grid / global_grid.max()

        logger.info(
            f"Heatmap generated: {len(per_player)} players, "
            f"{len(per_team)} teams, {len(all_pts)} total court positions"
        )

        return {
            "per_player": per_player,
            "per_team": per_team,
            "global": global_grid,
            "court_positions": court_positions,
            "grid_bounds": {
                "x_min": self.x_min,
                "x_max": self.x_max,
                "y_min": self.y_min,
                "y_max": self.y_max,
            },
            "homography_matrix": H,
        }

    # ──────────────────────────────────────────────────────────────────────
    # Rendering
    # ──────────────────────────────────────────────────────────────────────

    def render_court_heatmap(
        self,
        heatmap_data: Dict[str, Any],
        player_id: Optional[int] = None,
        team: Optional[str] = None,
        title: Optional[str] = None,
    ) -> np.ndarray:
        """
        Render a heatmap on a top-down court diagram (BGR image).

        Args:
            heatmap_data: Output of generate().
            player_id: If given, render that player's heatmap.
            team: If given, render that team's heatmap ('A' or 'B').
            title: Optional title drawn on the image.
                   If None, renders the global heatmap.

        Returns:
            BGR image (numpy array) of the court with heatmap overlay.
        """
        # Select the appropriate grid
        if player_id is not None and player_id in heatmap_data.get("per_player", {}):
            grid = heatmap_data["per_player"][player_id]
            default_title = f"Player {player_id}"
        elif team is not None and team in heatmap_data.get("per_team", {}):
            grid = heatmap_data["per_team"][team]
            default_title = f"Team {team}"
        else:
            grid = heatmap_data.get("global", np.zeros((self.grid_h, self.grid_w)))
            default_title = "All Players"

        title = title or default_title

        # Draw court + heatmap
        court_img = self._draw_court_diagram()
        heatmap_overlay = self._grid_to_color(grid)

        # Blend heatmap onto court
        alpha = self.config.heatmap.alpha
        blended = cv2.addWeighted(heatmap_overlay, alpha, court_img, 1.0 - alpha, 0)

        # Draw title
        cv2.putText(
            blended, title, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA,
        )

        return blended

    def render_overlay(
        self,
        frame: np.ndarray,
        heatmap_data: Dict[str, Any],
        field_info: Dict[str, Any],
        player_id: Optional[int] = None,
        team: Optional[str] = None,
    ) -> np.ndarray:
        """
        Render a heatmap overlaid on a video frame using inverse homography.

        Projects the court-plane heatmap back onto the perspective video frame.

        Args:
            frame: BGR video frame.
            heatmap_data: Output of generate().
            field_info: Field info dict with 'homography_matrix'.
            player_id: Optional single-player heatmap.
            team: Optional team heatmap.

        Returns:
            BGR frame with heatmap overlay.
        """
        H = field_info.get("homography_matrix")
        if H is None:
            logger.warning("No homography for overlay rendering")
            return frame.copy()

        # Select grid
        if player_id is not None and player_id in heatmap_data.get("per_player", {}):
            grid = heatmap_data["per_player"][player_id]
        elif team is not None and team in heatmap_data.get("per_team", {}):
            grid = heatmap_data["per_team"][team]
        else:
            grid = heatmap_data.get("global", np.zeros((self.grid_h, self.grid_w)))

        # Colorize grid
        heatmap_color = self._grid_to_color(grid)

        # Inverse homography: real-world → pixel
        try:
            H_inv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            logger.warning("Cannot invert homography matrix")
            return frame.copy()

        # Build source points (court plane corners) and destination (pixel)
        # We warp the heatmap_color image into the frame's perspective
        h_frame, w_frame = frame.shape[:2]

        # Define the 4 corners of the heatmap image in real-world coords
        src_corners = np.array([
            [self.x_min, self.y_min],
            [self.x_max, self.y_min],
            [self.x_max, self.y_max],
            [self.x_min, self.y_max],
        ], dtype=np.float32)

        # Project to pixel coords via inverse homography
        dst_corners = []
        for pt in src_corners:
            px = self._court_to_pixel(H_inv, pt[0], pt[1])
            if px is None:
                logger.warning("Cannot project heatmap corner to frame")
                return frame.copy()
            dst_corners.append(px)
        dst_corners = np.array(dst_corners, dtype=np.float32)

        # Source corners in heatmap image pixel coords
        hm_h, hm_w = heatmap_color.shape[:2]
        src_img_corners = np.array([
            [0, 0], [hm_w, 0], [hm_w, hm_h], [0, hm_h]
        ], dtype=np.float32)

        # Compute perspective transform for heatmap → frame
        M = cv2.getPerspectiveTransform(src_img_corners, dst_corners)
        warped = cv2.warpPerspective(
            heatmap_color, M, (w_frame, h_frame),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
        )

        # Create mask from non-zero warped pixels
        mask = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY) > 0
        mask_3ch = np.stack([mask] * 3, axis=-1)

        # Blend
        alpha = self.config.heatmap.alpha
        output = frame.copy()
        output[mask_3ch] = cv2.addWeighted(
            warped, alpha, frame, 1.0 - alpha, 0
        )[mask_3ch]

        return output

    def save_heatmap(
        self,
        heatmap_data: Dict[str, Any],
        output_dir: str,
        prefix: str = "heatmap",
    ) -> List[str]:
        """
        Save all heatmap images to disk.

        Args:
            heatmap_data: Output of generate().
            output_dir: Directory to write images.
            prefix: Filename prefix.

        Returns:
            List of saved file paths.
        """
        from pathlib import Path

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        saved: List[str] = []

        # Global
        img = self.render_court_heatmap(heatmap_data, title="All Players")
        path = str(out / f"{prefix}_global.png")
        cv2.imwrite(path, img)
        saved.append(path)

        # Per-player
        for pid in heatmap_data.get("per_player", {}):
            img = self.render_court_heatmap(heatmap_data, player_id=pid)
            path = str(out / f"{prefix}_player_{pid}.png")
            cv2.imwrite(path, img)
            saved.append(path)

        # Per-team
        for team_name in heatmap_data.get("per_team", {}):
            img = self.render_court_heatmap(heatmap_data, team=team_name)
            path = str(out / f"{prefix}_team_{team_name}.png")
            cv2.imwrite(path, img)
            saved.append(path)

        logger.info(f"Saved {len(saved)} heatmap images to {output_dir}")
        return saved

    # ──────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────

    def _pixel_to_court(
        self, H: np.ndarray, px: float, py: float
    ) -> Optional[Tuple[float, float]]:
        """
        Transform a pixel coordinate to real-world court coordinate
        using the homography matrix (image → real-world).

        Returns None if the projected point is far outside the court bounds.
        """
        pt = np.array([px, py, 1.0], dtype=np.float64)
        projected = H @ pt
        if abs(projected[2]) < 1e-8:
            return None
        x = projected[0] / projected[2]
        y = projected[1] / projected[2]

        # Reject points far outside the padded court
        margin = 2.0  # extra tolerance in meters
        if (x < self.x_min - margin or x > self.x_max + margin or
                y < self.y_min - margin or y > self.y_max + margin):
            return None

        return (float(x), float(y))

    def _court_to_pixel(
        self, H_inv: np.ndarray, cx: float, cy: float
    ) -> Optional[Tuple[float, float]]:
        """Transform a real-world court coordinate to pixel via inverse homography."""
        pt = np.array([cx, cy, 1.0], dtype=np.float64)
        projected = H_inv @ pt
        if abs(projected[2]) < 1e-8:
            return None
        px = projected[0] / projected[2]
        py = projected[1] / projected[2]
        return (float(px), float(py))

    def _accumulate(
        self, points: List[Tuple[float, float]]
    ) -> np.ndarray:
        """Accumulate court-coordinate points into a 2D histogram grid."""
        grid = np.zeros((self.grid_h, self.grid_w), dtype=np.float64)
        for x, y in points:
            # Convert real-world coords to grid cell indices
            gx = int((x - self.x_min) * self.res)
            gy = int((y - self.y_min) * self.res)
            if 0 <= gx < self.grid_w and 0 <= gy < self.grid_h:
                grid[gy, gx] += 1.0
        return grid

    def _smooth(self, grid: np.ndarray, sigma_px: float) -> np.ndarray:
        """Apply Gaussian smoothing to the density grid."""
        ksize = int(np.ceil(sigma_px * 6)) | 1  # ensure odd
        return cv2.GaussianBlur(grid, (ksize, ksize), sigma_px)

    def _grid_to_color(self, grid: np.ndarray) -> np.ndarray:
        """
        Convert a normalized [0, 1] density grid to a BGR color image
        using the configured colormap.
        """
        # Map to 0-255 uint8
        norm = np.clip(grid, 0, 1) if grid.max() <= 1 else grid / grid.max()
        gray = (norm * 255).astype(np.uint8)

        # Apply colormap
        cmap_name = self.config.heatmap.colormap.upper()
        cmap_map = {
            "JET": cv2.COLORMAP_JET,
            "HOT": cv2.COLORMAP_HOT,
            "INFERNO": cv2.COLORMAP_INFERNO,
            "MAGMA": cv2.COLORMAP_MAGMA,
            "VIRIDIS": cv2.COLORMAP_VIRIDIS,
            "PLASMA": cv2.COLORMAP_PLASMA,
            "TURBO": cv2.COLORMAP_TURBO,
        }
        cv_cmap = cmap_map.get(cmap_name, cv2.COLORMAP_JET)
        color = cv2.applyColorMap(gray, cv_cmap)

        # Make zero-density areas transparent (black)
        mask = gray == 0
        color[mask] = 0

        # Resize to output dimensions if needed
        out_w = self.config.heatmap.output_width
        aspect = self.grid_h / max(self.grid_w, 1)
        out_h = int(out_w * aspect)
        color = cv2.resize(color, (out_w, out_h), interpolation=cv2.INTER_LINEAR)

        return color

    def _draw_court_diagram(self) -> np.ndarray:
        """
        Draw a clean top-down padel court diagram (BGR image).

        Coordinate mapping:
            real-world x ∈ [x_min, x_max] → image column
            real-world y ∈ [y_min, y_max] → image row
        """
        out_w = self.config.heatmap.output_width
        aspect = (self.y_max - self.y_min) / max(self.x_max - self.x_min, 1e-6)
        out_h = int(out_w * aspect)

        img = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        img[:] = (40, 40, 40)  # dark gray background

        def to_px(x: float, y: float) -> Tuple[int, int]:
            """Convert real-world meters → pixel on diagram."""
            px = int((x - self.x_min) / (self.x_max - self.x_min) * out_w)
            py = int((y - self.y_min) / (self.y_max - self.y_min) * out_h)
            return (px, py)

        line_color = (200, 200, 200)  # light gray
        thick = 2

        # Court outline
        bl = to_px(-5, -10)
        br = to_px(5, -10)
        fl = to_px(5, 10)
        fr = to_px(-5, 10)
        cv2.rectangle(img, bl, fl, line_color, thick)

        # Net (center line)
        net_l = to_px(-5, 0)
        net_r = to_px(5, 0)
        cv2.line(img, net_l, net_r, (0, 200, 255), thick + 1)  # yellow-orange

        # Service lines
        sl_l = to_px(-5, 7)
        sl_r = to_px(5, 7)
        cv2.line(img, sl_l, sl_r, line_color, 1)

        st_l = to_px(-5, -7)
        st_r = to_px(5, -7)
        cv2.line(img, st_l, st_r, line_color, 1)

        # Center service lines (vertical, from service line to back wall)
        cs_front_top = to_px(0, 7)
        cs_front_bot = to_px(0, 10)
        cv2.line(img, cs_front_top, cs_front_bot, line_color, 1)

        cs_back_top = to_px(0, -10)
        cs_back_bot = to_px(0, -7)
        cv2.line(img, cs_back_top, cs_back_bot, line_color, 1)

        # Labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, "NET", (net_l[0] + 5, net_l[1] - 8),
                    font, 0.4, (0, 200, 255), 1, cv2.LINE_AA)

        return img

    def _empty_result(self) -> Dict[str, Any]:
        """Return an empty heatmap result when generation is not possible."""
        return {
            "per_player": {},
            "per_team": {},
            "global": np.zeros((self.grid_h, self.grid_w)),
            "court_positions": {},
            "grid_bounds": {
                "x_min": self.x_min,
                "x_max": self.x_max,
                "y_min": self.y_min,
                "y_max": self.y_max,
            },
            "homography_matrix": None,
        }
