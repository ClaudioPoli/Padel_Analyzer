"""
Field detection using YOLO-pose keypoints with geometric post-processing.

Uses a fine-tuned YOLO11-pose model to detect 10 field keypoints and applies
geometric constraints to interpolate missing/low-confidence points.

Keypoint mapping (10 keypoints):
    KP0: BL  - Back-Left corner (fondo campo lontano, sinistro)
    KP1: BR  - Back-Right corner (fondo campo lontano, destro)
    KP2: FL  - Front-Left (label dataset), ma geometricamente sul lato DESTRO
              *Collineare con BR e NBR* — spesso fuori inquadratura
    KP3: FR  - Front-Right (label dataset), ma geometricamente sul lato SINISTRO
              *Collineare con BL e NTL* — spesso fuori inquadratura
    KP4: NTL - Net-Top-Left (palo rete, alto sinistro)
    KP5: NTR - Net-Top-Right (palo rete, basso sinistro)
    KP6: NBL - Net-Bottom-Left (palo rete, alto destro)
    KP7: NBR - Net-Bottom-Right (palo rete, basso destro)
    KP8: SL  - Service-Line intersection (linea servizio lato campo/front)
    KP9: ST  - Service-Line intersection (linea servizio lato rete/back)

Geometric alignment (verified on dataset):
    Left sideline:  BL(0) → NTL(4) → FR(3)  — collinearity error ~4px
    Right sideline: BR(1) → NBR(7) → FL(2)  — collinearity error ~9px

Standard padel court dimensions (meters):
    Total length: 20m (10m per half)
    Total width: 10m
    Service line: 3m from back wall (each side)
    Net at center (10m from each back wall)
"""

from typing import Dict, Any, List, Tuple, Optional
import logging
import numpy as np
import cv2
from pathlib import Path

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

NUM_KEYPOINTS = 10

# Keypoint indices
KP_BL = 0   # Back-Left
KP_BR = 1   # Back-Right
KP_FL = 2   # Front-Left (dataset label) — actually on RIGHT sideline (collinear BR→NBR→FL)
KP_FR = 3   # Front-Right (dataset label) — actually on LEFT sideline (collinear BL→NTL→FR)
KP_NTL = 4  # Net-Top-Left (left sideline, upper net attachment)
KP_NTR = 5  # Net-Top-Right (left sideline, lower net attachment)
KP_NBL = 6  # Net-Bottom-Left (right sideline, upper net attachment)
KP_NBR = 7  # Net-Bottom-Right (right sideline, lower net attachment)
KP_SL = 8   # Service-Line (front/camera side)
KP_ST = 9   # Service-Line (back/far side)

KEYPOINT_NAMES = [
    "BL", "BR", "FL", "FR",
    "NTL", "NTR", "NBL", "NBR",
    "SL", "ST"
]

# Real-world padel court coordinates (in meters, origin at center of court)
# Used for homography computation
# Layout: 20m long × 10m wide, net at center
COURT_REAL_COORDS = {
    KP_BL:  (-5.0, -10.0),  # Back-Left
    KP_BR:  (5.0, -10.0),   # Back-Right
    KP_FL:  (5.0, 10.0),    # FL is on RIGHT sideline (collinear BR→NBR→FL)
    KP_FR:  (-5.0, 10.0),   # FR is on LEFT sideline (collinear BL→NTL→FR)
    KP_NTL: (-5.0, 0.0),    # Net, left side, upper attachment
    KP_NTR: (-5.0, 0.0),    # Net, left side, lower attachment (same x as NTL)
    KP_NBL: (5.0, 0.0),     # Net, right side, upper attachment
    KP_NBR: (5.0, 0.0),     # Net, right side, lower attachment (same x as NBL)
    KP_SL:  (0.0, 7.0),     # Service-Line (front side, 3m from front wall)
    KP_ST:  (0.0, -7.0),    # Service-Line (back side, 3m from back wall)
}

# Skeleton connections for visualization
# Left sideline:  BL(0) → NTL(4) ... NTR(5) → FR(3)
# Right sideline: BR(1) → NBL(6) ... NBR(7) → FL(2)
FIELD_SKELETON = [
    # Left sideline (back half + front half)
    (KP_BL, KP_NTL), (KP_NTR, KP_FR),
    # Right sideline (back half + front half)
    (KP_BR, KP_NBL), (KP_NBR, KP_FL),
    # Back line (fondo lontano)
    (KP_BL, KP_BR),
    # Front line (fondo vicino alla camera) — FL and FR connect
    (KP_FL, KP_FR),
    # Net (top and bottom bands)
    (KP_NTL, KP_NBL),
    (KP_NTR, KP_NBR),
    # Service lines
    (KP_SL, KP_FL), (KP_SL, KP_FR),
    (KP_ST, KP_BL), (KP_ST, KP_BR),
]

# Keypoint colors for visualization (BGR format)
KEYPOINT_COLORS = {
    KP_BL:  (0, 0, 255),     # Red
    KP_BR:  (0, 127, 255),   # Orange
    KP_FL:  (0, 255, 0),     # Green
    KP_FR:  (127, 255, 0),   # Lime
    KP_NTL: (255, 0, 0),     # Blue
    KP_NTR: (255, 0, 127),   # Purple
    KP_NBL: (255, 0, 255),   # Magenta
    KP_NBR: (255, 127, 255), # Pink
    KP_SL:  (0, 255, 255),   # Yellow
    KP_ST:  (255, 255, 0),   # Cyan
}


class KeypointFieldDetector:
    """
    Detects padel court geometry using a YOLO-pose model trained on field keypoints.
    
    Pipeline:
    1. Run YOLO-pose inference to get 10 keypoints + confidences
    2. Validate keypoints against geometric constraints
    3. Interpolate missing/low-confidence keypoints (especially FL/FR)
       using known court geometry and homography
    4. Return structured field info with quality flags
    
    The detector handles both single-frame and video-based detection,
    with optional temporal smoothing for video use.
    """
    
    def __init__(self, config: Any):
        """
        Initialize the KeypointFieldDetector.
        
        Args:
            config: Configuration object with field_keypoints settings
        """
        self.config = config
        self._model = None
        self._model_loaded = False
        
    def _load_model(self):
        """Lazy-load the YOLO model on first use."""
        if self._model_loaded:
            return
            
        from ultralytics import YOLO
        from ..utils.device import get_device
        
        model_path = Path(self.config.field_keypoints.model_path)
        
        if not model_path.exists():
            # Try relative to project root
            project_root = Path(__file__).parent.parent.parent
            model_path = project_root / self.config.field_keypoints.model_path
        
        if not model_path.exists():
            raise FileNotFoundError(
                f"Field keypoints model not found: {self.config.field_keypoints.model_path}. "
                f"Searched: {model_path}"
            )
        
        device = get_device(self.config.model.device)
        logger.info(f"Loading field keypoints model from {model_path} on {device}")
        
        self._model = YOLO(str(model_path))
        self._model.to(device)
        self._model_loaded = True
        
        logger.info("Field keypoints model loaded successfully")
    
    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────
    
    def detect(self, video_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect field keypoints from video data.
        
        Samples multiple frames, detects keypoints in each, and selects
        the best detection (or averages across frames).
        
        Args:
            video_data: Video data dict from VideoLoader with 'capture' and 'metadata'
            
        Returns:
            Field info dictionary (see _build_field_info for structure)
        """
        self._load_model()
        
        capture = video_data.get("capture")
        if capture is None:
            raise ValueError("Video not properly loaded - no capture object")
        
        metadata = video_data.get("metadata", {})
        frame_count = metadata.get("frame_count", 0)
        
        if frame_count == 0:
            logger.warning("No frames in video")
            return self._empty_field_info()
        
        # Sample frames from the video
        num_samples = min(self.config.field_keypoints.num_sample_frames, frame_count)
        sample_indices = np.linspace(
            frame_count // 4, 3 * frame_count // 4,
            num_samples, dtype=int
        )
        
        all_detections = []
        
        for idx in sample_indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = capture.read()
            if not ret:
                continue
            
            detection = self.detect_in_frame(frame)
            if detection is not None:
                all_detections.append(detection)
        
        if not all_detections:
            logger.warning("Could not detect field keypoints in any sampled frame")
            return self._empty_field_info()
        
        # Select best detection (highest overall confidence)
        best = max(all_detections, key=lambda d: d["detection_confidence"])
        
        logger.info(
            f"Field detected: confidence={best['detection_confidence']:.3f}, "
            f"reliable_kpts={best['num_reliable']}/10, "
            f"interpolated={best['num_interpolated']}"
        )
        
        return best
    
    def detect_in_frame(self, frame: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Detect field keypoints in a single frame.
        
        Args:
            frame: BGR image (numpy array)
            
        Returns:
            Field info dictionary or None if no field detected
        """
        self._load_model()
        
        conf_threshold = self.config.field_keypoints.min_detection_confidence
        
        # Run inference
        results = self._model(
            frame,
            conf=conf_threshold,
            verbose=False
        )
        
        if not results or len(results) == 0:
            return None
        
        result = results[0]
        
        # Check if we have keypoints
        if result.keypoints is None or len(result.keypoints) == 0:
            return None
        
        # Get the best detection (highest box confidence)
        if result.boxes is None or len(result.boxes) == 0:
            return None
        
        best_idx = result.boxes.conf.argmax().item()
        box_conf = result.boxes.conf[best_idx].item()
        
        # Extract keypoints: shape (num_keypoints, 3) → [x, y, confidence]
        # keypoints.data is (num_detections, num_keypoints, 3),
        # index with best_idx on the data tensor directly to get (10, 3)
        kpts_data = result.keypoints.data[best_idx].cpu().numpy()  # (10, 3)
        
        if kpts_data.shape[0] != NUM_KEYPOINTS:
            logger.warning(
                f"Expected {NUM_KEYPOINTS} keypoints, got {kpts_data.shape[0]}"
            )
            return None
        
        # Extract bbox
        box_xyxy = result.boxes.xyxy[best_idx].cpu().numpy()
        
        # Build structured field info
        return self._build_field_info(kpts_data, box_conf, box_xyxy, frame.shape)
    
    def detect_in_video_stream(
        self, 
        frame: np.ndarray, 
        previous_field_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Detect field keypoints in a single frame with temporal smoothing.
        
        Useful for real-time/streaming processing where you want smooth
        keypoint positions across frames.
        
        Args:
            frame: BGR image (numpy array)
            previous_field_info: Field info from previous frame for smoothing
            
        Returns:
            Field info dictionary
        """
        current = self.detect_in_frame(frame)
        
        if current is None:
            if previous_field_info is not None:
                return previous_field_info  # Reuse previous detection
            return self._empty_field_info()
        
        if (previous_field_info is None or 
            not self.config.field_keypoints.temporal_smoothing):
            return current
        
        # Apply exponential moving average smoothing
        alpha = 2.0 / (self.config.field_keypoints.smoothing_window + 1)
        
        prev_kpts = previous_field_info.get("keypoints_xy")
        curr_kpts = current.get("keypoints_xy")
        
        if prev_kpts is not None and curr_kpts is not None:
            # Smooth only reliable keypoints
            smoothed = curr_kpts.copy()
            for i in range(NUM_KEYPOINTS):
                if (current["keypoints_status"][i] in ("reliable", "interpolated") and
                    previous_field_info["keypoints_status"][i] in ("reliable", "interpolated")):
                    smoothed[i] = alpha * curr_kpts[i] + (1 - alpha) * prev_kpts[i]
            
            current["keypoints_xy"] = smoothed
            # Rebuild derived data
            current = self._rebuild_derived_data(current, frame.shape)
        
        return current
    
    # ──────────────────────────────────────────────────────────────────────
    # Core Processing
    # ──────────────────────────────────────────────────────────────────────
    
    def _build_field_info(
        self,
        kpts_data: np.ndarray,
        box_conf: float,
        box_xyxy: np.ndarray,
        frame_shape: tuple
    ) -> Dict[str, Any]:
        """
        Build structured field info from raw keypoint predictions.
        
        Applies confidence filtering, geometric interpolation for missing
        keypoints, and computes homography.
        
        Args:
            kpts_data: Raw keypoints array (10, 3) with [x, y, conf]
            box_conf: Detection box confidence
            box_xyxy: Detection bounding box [x1, y1, x2, y2]
            frame_shape: Frame shape (H, W, C)
            
        Returns:
            Comprehensive field info dictionary
        """
        h, w = frame_shape[:2]
        min_conf = self.config.field_keypoints.min_keypoint_confidence
        
        # Separate coordinates and confidences
        keypoints_xy = kpts_data[:, :2].copy()  # (10, 2) pixel coordinates
        keypoints_conf = kpts_data[:, 2].copy()  # (10,)
        
        # Classify each keypoint status
        keypoints_status = []
        num_reliable = 0
        
        for i in range(NUM_KEYPOINTS):
            if keypoints_conf[i] >= min_conf:
                keypoints_status.append("reliable")
                num_reliable += 1
            else:
                keypoints_status.append("low_confidence")
        
        # Validate FL/FR geometry: if model predicts them as "reliable"
        # but they are NOT collinear with their sideline, downgrade to
        # "low_confidence" so they get re-interpolated via homography.
        # This catches cases where the model hallucinates a front corner
        # on the wrong side (e.g., when the real corner is off-screen).
        keypoints_status = self._validate_front_corners(
            keypoints_xy, keypoints_status
        )
        
        # Attempt geometric interpolation for low-confidence keypoints
        num_interpolated = 0
        if self.config.field_keypoints.interpolate_missing:
            keypoints_xy, keypoints_status, num_interpolated = (
                self._interpolate_missing_keypoints(
                    keypoints_xy, keypoints_conf, keypoints_status, h, w
                )
            )
        
        # Compute homography from reliable + interpolated keypoints
        homography_matrix = self._compute_homography(
            keypoints_xy, keypoints_status
        )
        
        # If we have homography but still have unreliable FL/FR, project them
        if homography_matrix is not None:
            keypoints_xy, keypoints_status, extra_interp = (
                self._project_missing_via_homography(
                    keypoints_xy, keypoints_status, homography_matrix, h, w
                )
            )
            num_interpolated += extra_interp
        
        # Create court mask
        court_mask = self._create_court_mask(keypoints_xy, keypoints_status, h, w)
        
        # Compute overall detection confidence
        reliable_confs = [
            keypoints_conf[i] for i in range(NUM_KEYPOINTS)
            if keypoints_status[i] == "reliable"
        ]
        avg_reliable_conf = np.mean(reliable_confs) if reliable_confs else 0.0
        detection_confidence = box_conf * 0.3 + avg_reliable_conf * 0.7
        
        # Build the final result
        field_info = {
            # ── Core keypoint data ──
            "keypoints_xy": keypoints_xy,              # (10, 2) pixel coordinates
            "keypoints_conf": keypoints_conf,           # (10,) raw model confidences
            "keypoints_status": keypoints_status,       # list of "reliable"/"interpolated"/"low_confidence"
            "keypoints_names": KEYPOINT_NAMES,          # ["BL", "BR", "FL", ...]
            
            # ── Aggregated quality metrics ──
            "detection_confidence": detection_confidence,
            "num_reliable": num_reliable,
            "num_interpolated": num_interpolated,
            "num_low_confidence": sum(
                1 for s in keypoints_status if s == "low_confidence"
            ),
            
            # ── Court geometry ──
            "court_perimeter": self._get_court_perimeter(
                keypoints_xy, keypoints_status
            ),
            "net_line": self._get_net_line(keypoints_xy, keypoints_status),
            "half_courts": self._get_half_courts(keypoints_xy, keypoints_status),
            "service_lines": self._get_service_lines(
                keypoints_xy, keypoints_status
            ),
            
            # ── Transformation ──
            "homography_matrix": homography_matrix,     # Image → Real-world
            
            # ── Legacy compatible fields ──
            "boundaries": self._get_boundaries(keypoints_xy, keypoints_status),
            "corners": self._get_corners_list(keypoints_xy, keypoints_status),
            "court_mask": court_mask,
            "confidence": detection_confidence,
            "lines": [],  # Legacy: not used in keypoint-based detection
            
            # ── Bounding box ──
            "bbox_xyxy": box_xyxy.tolist(),
        }
        
        return field_info
    
    def _rebuild_derived_data(
        self, field_info: Dict[str, Any], frame_shape: tuple
    ) -> Dict[str, Any]:
        """Rebuild geometry data after keypoint smoothing."""
        h, w = frame_shape[:2]
        kpts = field_info["keypoints_xy"]
        status = field_info["keypoints_status"]
        
        field_info["court_perimeter"] = self._get_court_perimeter(kpts, status)
        field_info["net_line"] = self._get_net_line(kpts, status)
        field_info["half_courts"] = self._get_half_courts(kpts, status)
        field_info["service_lines"] = self._get_service_lines(kpts, status)
        field_info["boundaries"] = self._get_boundaries(kpts, status)
        field_info["corners"] = self._get_corners_list(kpts, status)
        field_info["court_mask"] = self._create_court_mask(kpts, status, h, w)
        
        return field_info
    
    # ──────────────────────────────────────────────────────────────────────
    # Geometric Validation & Interpolation
    # ──────────────────────────────────────────────────────────────────────
    
    def _validate_front_corners(
        self,
        keypoints_xy: np.ndarray,
        keypoints_status: List[str],
        max_collinearity_error: float = 100.0
    ) -> List[str]:
        """
        Validate FL and FR positions using sideline collinearity.
        
        If FL or FR is marked "reliable" but is NOT collinear with its
        sideline (within tolerance), downgrade it to "low_confidence".
        This catches cases where the model hallucinates a front corner
        on the wrong side when the real corner is off-screen.
        
        Sideline geometry (verified on dataset):
            FL (KP2) must be collinear with BR(1) → NBR(7)  [right sideline]
            FR (KP3) must be collinear with BL(0) → NTL(4)  [left sideline]
        
        The max_collinearity_error threshold is in pixels. For a 1920x1080
        video, typical sideline collinearity for correct detections is <30px.
        The threshold should be generous (e.g. 100px) to avoid rejecting
        correct but slightly noisy detections.
        
        Args:
            keypoints_xy: Keypoint positions (10, 2)
            keypoints_status: Status list
            max_collinearity_error: Maximum allowed collinearity error in pixels
            
        Returns:
            Updated status list
        """
        status = list(keypoints_status)
        usable = lambda i: status[i] in ("reliable", "interpolated")
        
        # ── Validate FL (KP2) — must be collinear with BR → NBR ──
        if status[KP_FL] == "reliable" and usable(KP_BR) and usable(KP_NBR):
            col_err = self._collinearity_error(
                keypoints_xy[KP_BR], keypoints_xy[KP_NBR], keypoints_xy[KP_FL]
            )
            if col_err > max_collinearity_error:
                status[KP_FL] = "low_confidence"
                logger.info(
                    f"FL rejected: collinearity error {col_err:.0f}px "
                    f"> {max_collinearity_error:.0f}px threshold "
                    f"(not on BR→NBR sideline)"
                )
        
        # ── Validate FR (KP3) — must be collinear with BL → NTL ──
        if status[KP_FR] == "reliable" and usable(KP_BL) and usable(KP_NTL):
            col_err = self._collinearity_error(
                keypoints_xy[KP_BL], keypoints_xy[KP_NTL], keypoints_xy[KP_FR]
            )
            if col_err > max_collinearity_error:
                status[KP_FR] = "low_confidence"
                logger.info(
                    f"FR rejected: collinearity error {col_err:.0f}px "
                    f"> {max_collinearity_error:.0f}px threshold "
                    f"(not on BL→NTL sideline)"
                )
        
        return status
    
    @staticmethod
    def _collinearity_error(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """
        Compute the perpendicular distance from p3 to the line through p1 and p2.
        
        Uses the cross-product formula: |v × w| / |v|
        where v = p2 - p1 and w = p3 - p1.
        """
        v = p2 - p1
        w = p3 - p1
        cross = abs(v[0] * w[1] - v[1] * w[0])
        line_len = np.linalg.norm(v)
        return cross / line_len if line_len > 1e-6 else float("inf")
    
    def _interpolate_missing_keypoints(
        self,
        keypoints_xy: np.ndarray,
        keypoints_conf: np.ndarray,
        keypoints_status: List[str],
        img_h: int,
        img_w: int
    ) -> Tuple[np.ndarray, List[str], int]:
        """
        Interpolate low-confidence keypoints using perspective-aware homography.
        
        Primary method: compute a ground-plane homography from the 6 reliable
        ground-level points (BL, BR, NTL, NBR, SL, ST), then project FL/FR
        from real-world coordinates to image coordinates via inverse homography.
        This correctly accounts for perspective foreshortening.
        
        Fallback: simple line extension (if insufficient points for homography).
        
        Ground-plane points (verified collinear with sidelines on dataset):
            Left sideline:  BL(0) → NTL(4) → FR(3)  — collinearity ~4px
            Right sideline: BR(1) → NBR(7) → FL(2)  — collinearity ~9px
            
        Non-ground points (top of net, NOT used for ground homography):
            NTR(5), NBL(6) — ~110px collinearity error with sidelines
        
        Args:
            keypoints_xy: Current keypoint positions (10, 2)
            keypoints_conf: Keypoint confidences (10,)
            keypoints_status: Status list
            img_h, img_w: Image dimensions
            
        Returns:
            Updated (keypoints_xy, keypoints_status, num_interpolated)
        """
        num_interpolated = 0
        kpts = keypoints_xy.copy()
        status = list(keypoints_status)
        
        # Check which front corners need interpolation
        need_fl = status[KP_FL] == "low_confidence"
        need_fr = status[KP_FR] == "low_confidence"
        
        if not need_fl and not need_fr:
            return kpts, status, 0
        
        # ── Primary method: homography-based projection ──
        H = self._compute_ground_plane_homography(kpts, status)
        
        if H is not None:
            try:
                H_inv = np.linalg.inv(H)
            except np.linalg.LinAlgError:
                H_inv = None
        else:
            H_inv = None
        
        if H_inv is not None:
            # Front corners are often near or beyond the frame edge.
            # Use a generous margin: the largest frame dimension.
            # This accepts off-screen projections up to 1x frame size
            # beyond the edge (clipped to bounds), while rejecting
            # wildly unstable projections from degenerate homographies.
            front_margin = max(img_w, img_h)
            
            # Project FL (KP2) via homography: real-world (5, 10) → image
            if need_fl:
                fl_img = self._project_point_via_homography(
                    H_inv, COURT_REAL_COORDS[KP_FL], img_h, img_w,
                    margin=front_margin
                )
                if fl_img is not None:
                    kpts[KP_FL] = fl_img
                    status[KP_FL] = "interpolated"
                    num_interpolated += 1
                    logger.debug(
                        f"FL interpolated via homography at "
                        f"({fl_img[0]:.0f}, {fl_img[1]:.0f})"
                    )
            
            # Project FR (KP3) via homography: real-world (-5, 10) → image
            if need_fr:
                fr_img = self._project_point_via_homography(
                    H_inv, COURT_REAL_COORDS[KP_FR], img_h, img_w,
                    margin=front_margin
                )
                if fr_img is not None:
                    kpts[KP_FR] = fr_img
                    status[KP_FR] = "interpolated"
                    num_interpolated += 1
                    logger.debug(
                        f"FR interpolated via homography at "
                        f"({fr_img[0]:.0f}, {fr_img[1]:.0f})"
                    )
        else:
            # ── Fallback: simple line extension (ignores perspective) ──
            logger.warning(
                "Ground-plane homography unavailable, "
                "falling back to simple line extension (less accurate)"
            )
            
            if need_fl:
                fl_est = self._extrapolate_front_corner_simple(
                    kpts, status, back_idx=KP_BR, net_idx=KP_NBR,
                    net_fallback_idx=KP_NBL
                )
                if fl_est is not None:
                    kpts[KP_FL] = np.clip(fl_est, [0, 0], [img_w - 1, img_h - 1])
                    status[KP_FL] = "interpolated"
                    num_interpolated += 1
                    logger.debug(
                        f"FL interpolated via line extension at "
                        f"({kpts[KP_FL][0]:.0f}, {kpts[KP_FL][1]:.0f})"
                    )
            
            if need_fr:
                fr_est = self._extrapolate_front_corner_simple(
                    kpts, status, back_idx=KP_BL, net_idx=KP_NTL,
                    net_fallback_idx=KP_NTR
                )
                if fr_est is not None:
                    kpts[KP_FR] = np.clip(fr_est, [0, 0], [img_w - 1, img_h - 1])
                    status[KP_FR] = "interpolated"
                    num_interpolated += 1
                    logger.debug(
                        f"FR interpolated via line extension at "
                        f"({kpts[KP_FR][0]:.0f}, {kpts[KP_FR][1]:.0f})"
                    )
        
        return kpts, status, num_interpolated
    
    def _compute_ground_plane_homography(
        self,
        kpts: np.ndarray,
        status: List[str]
    ) -> Optional[np.ndarray]:
        """
        Compute homography from ground-plane points only.
        
        Uses the 6 points that are always reliable and lie on the court surface:
            BL  (-5, -10) — back-left corner
            BR  ( 5, -10) — back-right corner
            NTL (-5,   0) — net at ground level, left sideline
            NBR ( 5,   0) — net at ground level, right sideline
            SL  ( 0,   7) — front service line center
            ST  ( 0,  -7) — back service line center
        
        NTR and NBL are excluded because they are at the TOP of the net
        (not on the ground plane), verified by ~110px collinearity error
        with the sidelines vs ~5px for NTL and NBR.
        
        Returns:
            3×3 homography matrix (image → real-world) or None
        """
        GROUND_PLANE_POINTS = {
            KP_BL:  (-5.0, -10.0),
            KP_BR:  (5.0, -10.0),
            KP_NTL: (-5.0, 0.0),
            KP_NBR: (5.0, 0.0),
            KP_SL:  (0.0, 7.0),
            KP_ST:  (0.0, -7.0),
        }
        
        src_points = []  # image coordinates
        dst_points = []  # real-world coordinates
        
        for kp_idx, real_coord in GROUND_PLANE_POINTS.items():
            if status[kp_idx] in ("reliable", "interpolated"):
                src_points.append(kpts[kp_idx])
                dst_points.append(real_coord)
        
        if len(src_points) < 4:
            logger.debug(
                f"Not enough ground-plane points for homography: "
                f"{len(src_points)} < 4"
            )
            return None
        
        src = np.array(src_points, dtype=np.float32)
        dst = np.array(dst_points, dtype=np.float32)
        
        try:
            H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
            if H is None:
                return None
            
            det = np.linalg.det(H[:2, :2])
            if abs(det) < 1e-6:
                logger.warning("Degenerate ground-plane homography")
                return None
            
            inliers = int(mask.sum()) if mask is not None else 0
            logger.debug(
                f"Ground-plane homography computed from "
                f"{len(src_points)} points ({inliers} inliers)"
            )
            return H
        except Exception as e:
            logger.warning(f"Failed to compute ground-plane homography: {e}")
            return None
    
    def _project_point_via_homography(
        self,
        H_inv: np.ndarray,
        real_world_coord: Tuple[float, float],
        img_h: int,
        img_w: int,
        margin: int = 100
    ) -> Optional[np.ndarray]:
        """
        Project a real-world point to image coordinates via inverse homography.
        
        Front corners (FL/FR) are often near or beyond the image edge.
        The margin parameter controls how far outside the image bounds a
        projected point is still accepted (and clipped to the edge).
        
        Args:
            H_inv: Inverse homography (real-world → image)
            real_world_coord: (x, y) in real-world meters
            img_h, img_w: Image dimensions
            margin: Allowed margin outside image bounds (pixels).
                    Use a large value for front corners that may be off-screen.
            
        Returns:
            Image coordinates as np.ndarray [x, y] (clipped to bounds) or None
        """
        real_pt = np.array(
            [real_world_coord[0], real_world_coord[1], 1.0], dtype=np.float64
        )
        img_pt = H_inv @ real_pt
        
        if abs(img_pt[2]) < 1e-10:
            return None
        
        img_pt = img_pt[:2] / img_pt[2]
        
        # Validate: must be within image bounds (with margin)
        if (-margin <= img_pt[0] <= img_w + margin and
                -margin <= img_pt[1] <= img_h + margin):
            return np.clip(img_pt, [0, 0], [img_w - 1, img_h - 1])
        
        return None
    
    def _extrapolate_front_corner_simple(
        self,
        kpts: np.ndarray,
        status: List[str],
        back_idx: int,
        net_idx: int,
        net_fallback_idx: int,
    ) -> Optional[np.ndarray]:
        """
        Fallback: simple line extension for front corner extrapolation.
        
        WARNING: This method ignores perspective foreshortening and produces
        inaccurate results (typically 200+ pixel error). Use only when
        homography-based projection is unavailable.
        
        Args:
            kpts: Keypoints array (10, 2)
            status: Status list
            back_idx: Index of the back corner on this sideline
            net_idx: Index of the primary net point on this sideline
            net_fallback_idx: Fallback net point (same side, other height)
            
        Returns:
            Estimated (x, y) or None
        """
        usable = lambda i: status[i] in ("reliable", "interpolated")
        
        if usable(back_idx) and usable(net_idx):
            back_pt = kpts[back_idx]
            net_pt = kpts[net_idx]
            direction = net_pt - back_pt
            return (net_pt + direction).copy()
        
        elif usable(back_idx) and usable(net_fallback_idx):
            back_pt = kpts[back_idx]
            net_pt = kpts[net_fallback_idx]
            direction = net_pt - back_pt
            return (net_pt + direction).copy()
        
        return None
    
    def _project_missing_via_homography(
        self,
        keypoints_xy: np.ndarray,
        keypoints_status: List[str],
        H: np.ndarray,
        img_h: int,
        img_w: int
    ) -> Tuple[np.ndarray, List[str], int]:
        """
        Project remaining low-confidence keypoints using homography.
        
        If we have a valid homography (computed from 4+ reliable points),
        we can project any real-world court point to image coordinates.
        
        Args:
            keypoints_xy: Keypoints array (10, 2)
            keypoints_status: Status list
            H: Homography matrix (image → real-world)
            img_h, img_w: Image dimensions
            
        Returns:
            Updated (keypoints_xy, keypoints_status, num_new_interpolated)
        """
        kpts = keypoints_xy.copy()
        status = list(keypoints_status)
        num_interpolated = 0
        
        # Invert homography: real-world → image
        try:
            H_inv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            return kpts, status, 0
        
        for i in range(NUM_KEYPOINTS):
            if status[i] != "low_confidence":
                continue
            
            if i not in COURT_REAL_COORDS:
                continue
            
            # Project real-world coordinate to image
            real_pt = np.array(
                [COURT_REAL_COORDS[i][0], COURT_REAL_COORDS[i][1], 1.0],
                dtype=np.float64
            )
            img_pt = H_inv @ real_pt
            
            if abs(img_pt[2]) < 1e-10:
                continue
            
            img_pt = img_pt[:2] / img_pt[2]
            
            # Validate projected point is within image (with margin)
            margin = 50  # Allow some pixels outside for near-edge points
            if (-margin <= img_pt[0] <= img_w + margin and
                -margin <= img_pt[1] <= img_h + margin):
                kpts[i] = np.clip(img_pt, [0, 0], [img_w - 1, img_h - 1])
                status[i] = "interpolated"
                num_interpolated += 1
                logger.debug(
                    f"KP{i} ({KEYPOINT_NAMES[i]}) projected via homography "
                    f"at ({kpts[i][0]:.0f}, {kpts[i][1]:.0f})"
                )
        
        return kpts, status, num_interpolated
    
    # ──────────────────────────────────────────────────────────────────────
    # Homography
    # ──────────────────────────────────────────────────────────────────────
    
    def _compute_homography(
        self,
        keypoints_xy: np.ndarray,
        keypoints_status: List[str]
    ) -> Optional[np.ndarray]:
        """
        Compute homography from image to real-world court coordinates.
        
        Needs at least 4 reliable/interpolated keypoints with known
        real-world positions.
        
        Args:
            keypoints_xy: Keypoints in pixel coordinates (10, 2)
            keypoints_status: Status list
            
        Returns:
            3×3 homography matrix or None
        """
        src_points = []  # Image points
        dst_points = []  # Real-world points
        
        for i in range(NUM_KEYPOINTS):
            if keypoints_status[i] not in ("reliable", "interpolated"):
                continue
            if i not in COURT_REAL_COORDS:
                continue
            
            # Skip net points that are NOT on the ground plane.
            # NTL and NBR are ground-level sideline points (collinear with
            # sidelines at ~5px error). NTR and NBL are at the TOP of the
            # net (~110px collinearity error) and should be skipped when
            # ground-level alternatives are available.
            if i == KP_NTR and keypoints_status[KP_NTL] in ("reliable", "interpolated"):
                continue  # Skip NTR (top of net, left) if NTL (ground) available
            if i == KP_NBL and keypoints_status[KP_NBR] in ("reliable", "interpolated"):
                continue  # Skip NBL (top of net, right) if NBR (ground) available
            
            src_points.append(keypoints_xy[i])
            dst_points.append(COURT_REAL_COORDS[i])
        
        if len(src_points) < 4:
            logger.debug(
                f"Not enough points for homography: {len(src_points)} < 4"
            )
            return None
        
        src = np.array(src_points, dtype=np.float32)
        dst = np.array(dst_points, dtype=np.float32)
        
        try:
            H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
            if H is None:
                return None
            
            # Validate homography: check it's not degenerate
            det = np.linalg.det(H[:2, :2])
            if abs(det) < 1e-6:
                logger.warning("Degenerate homography detected")
                return None
            
            return H
        except Exception as e:
            logger.warning(f"Failed to compute homography: {e}")
            return None
    
    # ──────────────────────────────────────────────────────────────────────
    # Court Geometry Extraction
    # ──────────────────────────────────────────────────────────────────────
    
    def _is_usable(self, status: List[str], idx: int) -> bool:
        """Check if a keypoint is reliable or interpolated."""
        return status[idx] in ("reliable", "interpolated")
    
    def _get_court_perimeter(
        self, kpts: np.ndarray, status: List[str]
    ) -> Optional[Dict[str, Any]]:
        """
        Get the full court perimeter as a polygon.
        
        Returns:
            Dict with 'points' (ordered polygon vertices), 
            'complete' flag, and 'area_px'
        """
        # Full perimeter: BL → BR → FL → FR (→ back to BL)
        # In image space: back-left → back-right → front-right → front-left
        # (FL is geometrically on the right side, FR on the left)
        indices = [KP_BL, KP_BR, KP_FL, KP_FR]
        
        points = []
        complete = True
        for idx in indices:
            if self._is_usable(status, idx):
                points.append(kpts[idx].tolist())
            else:
                complete = False
                points.append(None)
        
        # Calculate area if complete
        area = None
        if complete:
            pts = np.array(points, dtype=np.float32)
            area = float(cv2.contourArea(pts))
        
        return {
            "points": points,
            "complete": complete,
            "area_px": area,
            "vertex_names": ["BL", "BR", "FL", "FR"]
        }
    
    def _get_net_line(
        self, kpts: np.ndarray, status: List[str]
    ) -> Optional[Dict[str, Any]]:
        """
        Get the net line (divides court into two halves).
        
        The net connects left-side net points to right-side net points.
        Uses the midpoints of (NTL, NTR) and (NBL, NBR) for the top
        and bottom of the visible net.
        """
        # Left side of net: midpoint of NTL and NTR
        left_ok = self._is_usable(status, KP_NTL) and self._is_usable(status, KP_NTR)
        right_ok = self._is_usable(status, KP_NBL) and self._is_usable(status, KP_NBR)
        
        if not (left_ok and right_ok):
            # Try with just the upper or lower points
            if self._is_usable(status, KP_NTL) and self._is_usable(status, KP_NBL):
                return {
                    "start": kpts[KP_NTL].tolist(),
                    "end": kpts[KP_NBL].tolist(),
                    "complete": True
                }
            return None
        
        # Full net with 4 points (trapezoid due to perspective)
        return {
            "start": kpts[KP_NTL].tolist(),
            "end": kpts[KP_NBL].tolist(),
            "start_bottom": kpts[KP_NTR].tolist(),
            "end_bottom": kpts[KP_NBR].tolist(),
            "complete": True,
            "left_points": [kpts[KP_NTL].tolist(), kpts[KP_NTR].tolist()],
            "right_points": [kpts[KP_NBL].tolist(), kpts[KP_NBR].tolist()],
        }
    
    def _get_half_courts(
        self, kpts: np.ndarray, status: List[str]
    ) -> Optional[Dict[str, Any]]:
        """
        Get the two half-courts divided by the net.
        
        Returns:
            Dict with 'back_half' and 'front_half' polygons
        """
        # Back half: BL → BR → NBL → NTL
        back_indices = [KP_BL, KP_BR, KP_NBL, KP_NTL]
        back_points = []
        back_complete = True
        for idx in back_indices:
            if self._is_usable(status, idx):
                back_points.append(kpts[idx].tolist())
            else:
                back_complete = False
                back_points.append(None)
        
        # Front half: NTR → NBR → FL → FR
        # (NTR=left side near front, NBR=right side near front, FL=right front, FR=left front)
        front_indices = [KP_NTR, KP_NBR, KP_FL, KP_FR]
        front_points = []
        front_complete = True
        for idx in front_indices:
            if self._is_usable(status, idx):
                front_points.append(kpts[idx].tolist())
            else:
                front_complete = False
                front_points.append(None)
        
        return {
            "back_half": {
                "points": back_points,
                "complete": back_complete,
                "vertex_names": ["BL", "BR", "NBL", "NTL"]
            },
            "front_half": {
                "points": front_points,
                "complete": front_complete,
                "vertex_names": ["NTR", "NBR", "FL", "FR"]
            }
        }
    
    def _get_service_lines(
        self, kpts: np.ndarray, status: List[str]
    ) -> Optional[Dict[str, Any]]:
        """Get service line positions."""
        result = {}
        
        if self._is_usable(status, KP_SL):
            result["front_service"] = kpts[KP_SL].tolist()
        if self._is_usable(status, KP_ST):
            result["back_service"] = kpts[KP_ST].tolist()
        
        return result if result else None
    
    def _get_boundaries(
        self, kpts: np.ndarray, status: List[str]
    ) -> Optional[List[Tuple[int, int]]]:
        """Legacy: Get 4 corner boundaries for backward compatibility."""
        corners = []
        for idx in [KP_BL, KP_BR, KP_FL, KP_FR]:
            if self._is_usable(status, idx):
                corners.append(tuple(kpts[idx].astype(int).tolist()))
            else:
                return None  # Can't provide complete boundaries
        return corners
    
    def _get_corners_list(
        self, kpts: np.ndarray, status: List[str]
    ) -> List[Tuple[int, int]]:
        """Legacy: Get all usable corners as a flat list."""
        corners = []
        for idx in [KP_BL, KP_BR, KP_FL, KP_FR]:
            if self._is_usable(status, idx):
                corners.append(tuple(kpts[idx].astype(int).tolist()))
        return corners
    
    # ──────────────────────────────────────────────────────────────────────
    # Court Mask
    # ──────────────────────────────────────────────────────────────────────
    
    def _create_court_mask(
        self,
        kpts: np.ndarray,
        status: List[str],
        img_h: int,
        img_w: int
    ) -> Optional[np.ndarray]:
        """
        Create binary mask of the court area.
        
        Uses the 4 corner keypoints (with expansion) to create a mask.
        Falls back to available points if not all corners are usable.
        """
        corner_indices = [KP_BL, KP_BR, KP_FR, KP_FL]
        available = [
            i for i in corner_indices if self._is_usable(status, i)
        ]
        
        if len(available) < 3:
            return None
        
        pts = np.array([kpts[i] for i in available], dtype=np.float32)
        
        # Expand polygon by 10% for player tracking (players near edges)
        center = pts.mean(axis=0)
        expanded = []
        for pt in pts:
            direction = pt - center
            exp_pt = pt + direction * 0.1
            exp_pt[0] = np.clip(exp_pt[0], 0, img_w - 1)
            exp_pt[1] = np.clip(exp_pt[1], 0, img_h - 1)
            expanded.append(exp_pt)
        
        mask = np.zeros((img_h, img_w), dtype=np.uint8)
        hull = cv2.convexHull(np.array(expanded, dtype=np.int32))
        cv2.fillPoly(mask, [hull], 255)
        
        return mask
    
    # ──────────────────────────────────────────────────────────────────────
    # Visualization
    # ──────────────────────────────────────────────────────────────────────
    
    def draw_field_info(
        self,
        frame: np.ndarray,
        field_info: Dict[str, Any],
        draw_keypoints: bool = True,
        draw_skeleton: bool = True,
        draw_labels: bool = True,
        draw_perimeter: bool = True,
    ) -> np.ndarray:
        """
        Draw field detection results on a frame.
        
        Args:
            frame: BGR image to draw on (will be modified in-place)
            field_info: Field info from detect_in_frame
            draw_keypoints: Draw keypoint circles
            draw_skeleton: Draw court skeleton lines
            draw_labels: Draw keypoint name labels
            draw_perimeter: Draw court perimeter
            
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        kpts = field_info.get("keypoints_xy")
        status = field_info.get("keypoints_status")
        
        if kpts is None or status is None:
            return annotated
        
        # Draw skeleton connections
        if draw_skeleton:
            for i, j in FIELD_SKELETON:
                if self._is_usable(status, i) and self._is_usable(status, j):
                    pt1 = tuple(kpts[i].astype(int))
                    pt2 = tuple(kpts[j].astype(int))
                    # Color based on whether points are interpolated
                    if status[i] == "interpolated" or status[j] == "interpolated":
                        color = (0, 165, 255)  # Orange for interpolated lines
                        thickness = 1
                    else:
                        color = (0, 255, 0)  # Green for reliable lines
                        thickness = 2
                    cv2.line(annotated, pt1, pt2, color, thickness)
        
        # Draw court perimeter
        if draw_perimeter:
            perimeter = field_info.get("court_perimeter")
            if perimeter and perimeter.get("complete"):
                pts = np.array(perimeter["points"], dtype=np.int32)
                cv2.polylines(annotated, [pts], True, (255, 255, 0), 2)
        
        # Draw keypoints
        if draw_keypoints:
            for i in range(NUM_KEYPOINTS):
                if not self._is_usable(status, i):
                    continue
                
                pt = tuple(kpts[i].astype(int))
                color = KEYPOINT_COLORS.get(i, (255, 255, 255))
                
                if status[i] == "interpolated":
                    # Dashed circle effect for interpolated
                    cv2.circle(annotated, pt, 10, color, 2)
                    cv2.circle(annotated, pt, 6, (0, 165, 255), -1)
                else:
                    # Solid circle for reliable
                    cv2.circle(annotated, pt, 8, color, -1)
                    cv2.circle(annotated, pt, 10, (255, 255, 255), 2)
                
                if draw_labels:
                    label = KEYPOINT_NAMES[i]
                    conf = field_info["keypoints_conf"][i]
                    label_text = f"{label} {conf:.2f}"
                    
                    # Background for text
                    (tw, th), _ = cv2.getTextSize(
                        label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
                    )
                    cv2.rectangle(
                        annotated,
                        (pt[0] + 12, pt[1] - th - 4),
                        (pt[0] + 14 + tw, pt[1] + 4),
                        (0, 0, 0), -1
                    )
                    cv2.putText(
                        annotated, label_text,
                        (pt[0] + 13, pt[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1
                    )
        
        # Draw info box
        info_lines = [
            f"Conf: {field_info.get('detection_confidence', 0):.3f}",
            f"Reliable: {field_info.get('num_reliable', 0)}/10",
            f"Interpolated: {field_info.get('num_interpolated', 0)}",
        ]
        y_offset = 30
        for line in info_lines:
            cv2.putText(
                annotated, line, (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
            )
            y_offset += 25
        
        return annotated
    
    # ──────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────
    
    def _empty_field_info(self) -> Dict[str, Any]:
        """Return empty field info structure."""
        return {
            "keypoints_xy": None,
            "keypoints_conf": None,
            "keypoints_status": ["low_confidence"] * NUM_KEYPOINTS,
            "keypoints_names": KEYPOINT_NAMES,
            "detection_confidence": 0.0,
            "num_reliable": 0,
            "num_interpolated": 0,
            "num_low_confidence": NUM_KEYPOINTS,
            "court_perimeter": None,
            "net_line": None,
            "half_courts": None,
            "service_lines": None,
            "homography_matrix": None,
            "boundaries": None,
            "corners": [],
            "court_mask": None,
            "confidence": 0.0,
            "lines": [],
            "bbox_xyxy": None,
        }
