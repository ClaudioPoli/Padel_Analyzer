"""
Player tracking module for detecting and tracking players in video frames.

Includes a PlayerIdentityManager that maintains stable player identities (1-4)
across frames, preventing ID swaps when players are near each other. Uses:
- Kalman filter for robust position prediction
- Appearance gallery with exponential moving average (stable + adaptive templates)
- Occlusion detection to freeze IDs when players overlap
- Keypoint-based features (estimated height, body proportions) for discrimination
- Anti-swap geometric constraints
- Re-identification buffer for temporarily lost players
"""

from typing import Dict, Any, List, Optional, Tuple
from itertools import combinations, permutations
import logging
import numpy as np
import cv2
from collections import defaultdict

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Kalman Filter for 2D position + velocity tracking
# ---------------------------------------------------------------------------

class PlayerKalmanFilter:
    """
    Per-player Kalman filter tracking position (x, y) and velocity (vx, vy).
    
    State vector: [x, y, vx, vy]
    Measurement:  [x, y]
    
    Provides much smoother and more accurate position predictions than
    simple velocity averaging, especially during direction changes and
    when measurements are noisy.
    """
    
    def __init__(self, initial_pos: Tuple[float, float],
                 process_noise: float = 5.0, measurement_noise: float = 10.0):
        self.kf = cv2.KalmanFilter(4, 2)  # 4 state dims, 2 measurement dims
        
        # Transition matrix (constant velocity model)
        # x' = x + vx*dt, y' = y + vy*dt, vx' = vx, vy' = vy
        dt = 1.0  # 1 frame
        self.kf.transitionMatrix = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1],
        ], dtype=np.float32)
        
        # Measurement matrix: we observe x and y
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float32)
        
        # Process noise covariance
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * process_noise
        # Velocity has higher process noise (players accelerate/decelerate)
        self.kf.processNoiseCov[2, 2] = process_noise * 2
        self.kf.processNoiseCov[3, 3] = process_noise * 2
        
        # Measurement noise covariance
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * measurement_noise
        
        # Initial state
        self.kf.statePost = np.array(
            [initial_pos[0], initial_pos[1], 0, 0], dtype=np.float32
        ).reshape(4, 1)
        
        # Initial error covariance
        self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 100
        
        self.frames_without_measurement = 0
    
    def predict(self) -> Tuple[float, float]:
        """Predict next position. Returns (x, y)."""
        pred = self.kf.predict()
        return (float(pred[0, 0]), float(pred[1, 0]))
    
    def update(self, measurement: Tuple[float, float]):
        """Correct state with a new measurement (x, y)."""
        meas = np.array([measurement[0], measurement[1]], dtype=np.float32).reshape(2, 1)
        self.kf.correct(meas)
        self.frames_without_measurement = 0
    
    def predict_without_update(self) -> Tuple[float, float]:
        """
        Predict without correcting (player not detected this frame).
        Increases internal uncertainty.
        """
        pred = self.kf.predict()
        self.frames_without_measurement += 1
        return (float(pred[0, 0]), float(pred[1, 0]))
    
    @property
    def velocity(self) -> Tuple[float, float]:
        """Current estimated velocity (vx, vy)."""
        return (float(self.kf.statePost[2, 0]), float(self.kf.statePost[3, 0]))
    
    @property
    def position(self) -> Tuple[float, float]:
        """Current estimated position (x, y)."""
        return (float(self.kf.statePost[0, 0]), float(self.kf.statePost[1, 0]))


# ---------------------------------------------------------------------------
# Appearance Gallery — stable template + adaptive EMA
# ---------------------------------------------------------------------------

class AppearanceGallery:
    """
    Maintains a robust appearance model for a single player using:
    - A *stable template*: averaged over the first N good observations,
      resistant to drift during occlusions.
    - An *adaptive template*: exponential moving average updated each frame,
      tracks gradual appearance changes (lighting, pose).
    
    Matching uses a weighted combination of both templates.
    """
    
    STABLE_TEMPLATE_FRAMES = 15   # frames used to build the stable template
    EMA_ALPHA = 0.15              # update rate for adaptive template
    STABLE_WEIGHT = 0.6           # weight of stable template in final score
    
    def __init__(self, initial_hist: Optional[np.ndarray] = None):
        self.stable_template: Optional[np.ndarray] = None
        self.adaptive_template: Optional[np.ndarray] = None
        self._stable_accum: List[np.ndarray] = []
        self._stable_locked = False
        
        if initial_hist is not None:
            self._update_internal(initial_hist)
    
    def update(self, hist: Optional[np.ndarray]):
        """Add a new observation to the gallery."""
        if hist is None:
            return
        self._update_internal(hist)
    
    def _update_internal(self, hist: np.ndarray):
        # Build stable template from first N observations
        if not self._stable_locked:
            self._stable_accum.append(hist.copy())
            if len(self._stable_accum) >= self.STABLE_TEMPLATE_FRAMES:
                self.stable_template = np.mean(self._stable_accum, axis=0).astype(np.float32)
                cv2.normalize(self.stable_template, self.stable_template)
                self._stable_locked = True
                self._stable_accum = []  # free memory
            else:
                # Temporary stable = mean so far
                self.stable_template = np.mean(self._stable_accum, axis=0).astype(np.float32)
                cv2.normalize(self.stable_template, self.stable_template)
        
        # Update adaptive template via EMA
        if self.adaptive_template is None:
            self.adaptive_template = hist.copy().astype(np.float32)
        else:
            self.adaptive_template = (
                (1 - self.EMA_ALPHA) * self.adaptive_template
                + self.EMA_ALPHA * hist.astype(np.float32)
            )
            cv2.normalize(self.adaptive_template, self.adaptive_template)
    
    def compare(self, hist: Optional[np.ndarray]) -> float:
        """
        Compare an observation against this gallery.
        Returns similarity in [0, 1] (higher = more similar).
        """
        if hist is None:
            return 0.0
        
        stable_sim = 0.0
        adaptive_sim = 0.0
        
        if self.stable_template is not None:
            stable_sim = self._hist_correl(self.stable_template, hist)
        if self.adaptive_template is not None:
            adaptive_sim = self._hist_correl(self.adaptive_template, hist)
        
        if self.stable_template is not None and self.adaptive_template is not None:
            return self.STABLE_WEIGHT * stable_sim + (1 - self.STABLE_WEIGHT) * adaptive_sim
        elif self.stable_template is not None:
            return stable_sim
        elif self.adaptive_template is not None:
            return adaptive_sim
        return 0.0
    
    @staticmethod
    def _hist_correl(h1: np.ndarray, h2: np.ndarray) -> float:
        return float(cv2.compareHist(
            h1.reshape(-1, 1).astype(np.float32),
            h2.reshape(-1, 1).astype(np.float32),
            cv2.HISTCMP_CORREL,
        ))


class PlayerIdentityManager:
    """
    Maintains stable player identities (1-4) independently of ByteTrack IDs.
    
    Uses a combination of:
    - **Kalman filter** for position prediction (handles acceleration/deceleration)
    - **Appearance gallery** with stable + adaptive templates (resists drift)
    - **Occlusion detection**: when bboxes overlap, freezes IDs and relies on prediction
    - **Keypoint features**: estimated height and proportions for extra discrimination
    - **Anti-swap constraints**: penalizes assignments that would swap two nearby players
    - **Re-ID buffer**: keeps state for lost players and re-matches them later
    - ByteTrack ID consistency bonus
    - Bounding box size consistency
    """
    
    # --- Tunable thresholds ---
    INIT_FRAMES_NEEDED = 5
    MAX_COST_THRESHOLD = 500
    OCCLUSION_IOU_THRESHOLD = 0.10   # bbox IoU above which players are "occluding"
    LOST_PLAYER_MAX_FRAMES = 60      # keep predicting for this many frames after loss
    APPEARANCE_COST_WEIGHT = 250     # scale appearance dissimilarity to pixel-equivalent
    SIZE_COST_WEIGHT = 50
    BYTETRACK_BONUS = 0.6           # multiplier when ByteTrack ID agrees
    SWAP_PENALTY = 150              # extra cost for assignments that swap nearby players
    KEYPOINT_HEIGHT_WEIGHT = 40     # cost contribution from height difference
    
    def __init__(self, max_players: int = 4, history_len: int = 30, config=None):
        self.max_players = max_players
        self.history_len = history_len
        self.players: Dict[int, Dict[str, Any]] = {}   # stable_id -> state
        self.bt_to_stable: Dict[int, int] = {}          # bytetrack_id -> stable_id
        self.next_id = 1
        self._initialized = False
        self._init_frames: List[List[Dict[str, Any]]] = []
        self._init_frame_data: List[np.ndarray] = []
        
        # Apply config overrides if available
        if config is not None:
            self.history_len = getattr(config, 'identity_history_len', history_len)
            self.INIT_FRAMES_NEEDED = getattr(config, 'identity_init_frames', self.INIT_FRAMES_NEEDED)
            self.OCCLUSION_IOU_THRESHOLD = getattr(config, 'occlusion_iou_threshold', self.OCCLUSION_IOU_THRESHOLD)
            self.LOST_PLAYER_MAX_FRAMES = getattr(config, 'lost_player_max_frames', self.LOST_PLAYER_MAX_FRAMES)
            self.SWAP_PENALTY = getattr(config, 'swap_penalty', self.SWAP_PENALTY)
            self.APPEARANCE_COST_WEIGHT = getattr(config, 'appearance_cost_weight', self.APPEARANCE_COST_WEIGHT)
            self._kalman_process_noise = getattr(config, 'kalman_process_noise', 5.0)
            self._kalman_measurement_noise = getattr(config, 'kalman_measurement_noise', 10.0)
        else:
            self._kalman_process_noise = 5.0
            self._kalman_measurement_noise = 10.0
        
        # Previous-frame assignment (for anti-swap detection)
        self._prev_assignment: Dict[int, int] = {}  # stable_id -> det index
        self._prev_det_centers: List[Tuple[int, int]] = []
        
        # Frame counter
        self._frame_count = 0
    
    def update(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process detections for a new frame. Assigns a 'stable_player_id' to
        each detection, maintaining consistent identity across frames.
        """
        if not detections:
            # Even with no detections, advance Kalman filters for lost players
            if self._initialized:
                self._advance_lost_players()
            self._frame_count += 1
            return detections
        
        # Extract appearance features for each detection
        for det in detections:
            det['_appearance'] = self._extract_appearance(frame, det['bbox'])
            det['_height_est'] = self._estimate_height_from_keypoints(det)
        
        # Initialization phase: collect a few frames to pick the best 4 players
        if not self._initialized:
            self._init_frames.append(detections)
            self._init_frame_data.append(frame)
            if len(self._init_frames) >= self.INIT_FRAMES_NEEDED:
                self._initialize_players()
            else:
                for det in detections:
                    det['stable_player_id'] = det.get('track_id')
                self._frame_count += 1
                return detections
        
        # Detect occlusions between current detections
        occlusion_pairs = self._detect_occlusions(detections)
        
        # Match detections to known stable players
        self._match_detections(detections, occlusion_pairs)
        
        # Advance Kalman filters for players not matched this frame
        self._advance_lost_players()
        
        self._frame_count += 1
        return detections
    
    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    
    def _initialize_players(self):
        """
        After collecting INIT_FRAMES_NEEDED frames, identify the 4 main players
        and assign stable IDs 1-4, sorted by Y position so that top-of-court
        players get lower IDs (Team A) and bottom-of-court get higher IDs (Team B).
        """
        bt_counts: Dict[int, int] = defaultdict(int)
        bt_positions: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
        bt_appearances: Dict[int, List[np.ndarray]] = defaultdict(list)
        bt_bboxes: Dict[int, List[List[int]]] = defaultdict(list)
        bt_heights: Dict[int, List[float]] = defaultdict(list)
        
        for frame_dets in self._init_frames:
            for det in frame_dets:
                bt_id = det.get('track_id')
                if bt_id is None:
                    continue
                bt_counts[bt_id] += 1
                bt_positions[bt_id].append(det['center'])
                bt_bboxes[bt_id].append(det['bbox'])
                if det.get('_appearance') is not None:
                    bt_appearances[bt_id].append(det['_appearance'])
                h_est = det.get('_height_est', 0.0)
                if h_est > 0:
                    bt_heights[bt_id].append(h_est)
        
        if not bt_counts:
            self._initialized = True
            return
        
        top_ids = sorted(bt_counts.keys(), key=lambda k: bt_counts[k], reverse=True)
        top_ids = top_ids[:self.max_players]
        
        def median_y(bt_id):
            return np.median([p[1] for p in bt_positions[bt_id]])
        
        top_ids.sort(key=median_y)
        
        for stable_id, bt_id in enumerate(top_ids, start=1):
            # Build appearance gallery from init observations
            gallery = AppearanceGallery()
            for hist in bt_appearances.get(bt_id, []):
                gallery.update(hist)
            
            # Initialize Kalman filter from median position
            positions = bt_positions[bt_id]
            med_pos = (
                float(np.median([p[0] for p in positions])),
                float(np.median([p[1] for p in positions]))
            )
            kf = PlayerKalmanFilter(
                med_pos,
                process_noise=self._kalman_process_noise,
                measurement_noise=self._kalman_measurement_noise,
            )
            # Feed all init positions to warm up the filter
            for pos in positions:
                kf.predict()
                kf.update(pos)
            
            avg_height = float(np.median(bt_heights[bt_id])) if bt_heights.get(bt_id) else 0.0
            
            self.players[stable_id] = {
                'positions': list(bt_positions[bt_id][-self.history_len:]),
                'bboxes': list(bt_bboxes[bt_id][-self.history_len:]),
                'gallery': gallery,
                'kalman': kf,
                'frames_lost': 0,
                'height_estimate': avg_height,
            }
            self.bt_to_stable[bt_id] = stable_id
        
        self.next_id = len(self.players) + 1
        self._initialized = True
        
        for frame_dets in self._init_frames:
            for det in frame_dets:
                bt_id = det.get('track_id')
                det['stable_player_id'] = self.bt_to_stable.get(bt_id)
        
        logger.info(
            f"PlayerIdentityManager initialized with {len(self.players)} players "
            f"(ByteTrack→Stable mapping: {self.bt_to_stable})"
        )
    
    # ------------------------------------------------------------------
    # Occlusion detection
    # ------------------------------------------------------------------
    
    def _detect_occlusions(self, detections: List[Dict[str, Any]]) -> set:
        """
        Detect pairs of detections that are overlapping (potential occlusion).
        Returns a set of frozensets of detection indices that are occluding.
        """
        occlusion_pairs = set()
        for i in range(len(detections)):
            for j in range(i + 1, len(detections)):
                iou = self._calculate_iou(detections[i]['bbox'], detections[j]['bbox'])
                if iou > self.OCCLUSION_IOU_THRESHOLD:
                    occlusion_pairs.add(frozenset((i, j)))
        return occlusion_pairs
    
    @staticmethod
    def _calculate_iou(bbox1: List[int], bbox2: List[int]) -> float:
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        if x2 < x1 or y2 < y1:
            return 0.0
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        return intersection / union if union > 0 else 0.0
    
    # ------------------------------------------------------------------
    # Keypoint-based features
    # ------------------------------------------------------------------
    
    @staticmethod
    def _estimate_height_from_keypoints(det: Dict[str, Any]) -> float:
        """
        Estimate a player's approximate pixel height from keypoints
        (head to ankle distance). Returns 0 if keypoints unavailable.
        """
        kpts = det.get('keypoints')
        kpts_conf = det.get('keypoints_conf')
        if kpts is None or kpts_conf is None:
            return 0.0
        
        # COCO keypoint indices: 0=nose, 5=left_shoulder, 6=right_shoulder,
        # 15=left_ankle, 16=right_ankle
        HEAD_IDX = 0
        L_ANKLE, R_ANKLE = 15, 16
        
        conf_threshold = 0.3
        head = None
        if kpts_conf[HEAD_IDX] > conf_threshold:
            head = kpts[HEAD_IDX]
        
        ankle = None
        if kpts_conf[L_ANKLE] > conf_threshold and kpts_conf[R_ANKLE] > conf_threshold:
            ankle = (kpts[L_ANKLE] + kpts[R_ANKLE]) / 2.0
        elif kpts_conf[L_ANKLE] > conf_threshold:
            ankle = kpts[L_ANKLE]
        elif kpts_conf[R_ANKLE] > conf_threshold:
            ankle = kpts[R_ANKLE]
        
        if head is not None and ankle is not None:
            return float(np.linalg.norm(ankle - head))
        return 0.0
    
    # ------------------------------------------------------------------
    # Matching
    # ------------------------------------------------------------------
    
    def _match_detections(self, detections: List[Dict[str, Any]],
                          occlusion_pairs: set):
        """
        Match current frame detections to stable player identities.
        
        Uses Kalman-predicted positions, appearance gallery comparison,
        size consistency, keypoint height, ByteTrack bonus, and anti-swap
        penalty to build a cost matrix, then solves optimal assignment.
        
        During occlusions, relies more heavily on Kalman prediction and
        appearance gallery (especially the stable template).
        """
        stable_ids = list(self.players.keys())
        n_players = len(stable_ids)
        n_dets = len(detections)
        
        if n_players == 0 or n_dets == 0:
            for det in detections:
                det['stable_player_id'] = None
            return
        
        # Indices of detections involved in any occlusion
        occluded_det_indices = set()
        for pair in occlusion_pairs:
            occluded_det_indices.update(pair)
        
        # Build cost matrix (n_players x n_dets)
        cost_matrix = np.full((n_players, n_dets), 1e6, dtype=np.float64)
        
        for i, sid in enumerate(stable_ids):
            player = self.players[sid]
            kf: PlayerKalmanFilter = player['kalman']
            pred_pos = kf.predict()
            gallery: AppearanceGallery = player['gallery']
            last_bbox = player['bboxes'][-1] if player['bboxes'] else None
            player_height = player.get('height_estimate', 0.0)
            
            for j, det in enumerate(detections):
                is_occluded = j in occluded_det_indices
                
                # 1. Position cost (Kalman-predicted)
                dx = det['center'][0] - pred_pos[0]
                dy = det['center'][1] - pred_pos[1]
                pos_cost = np.sqrt(dx * dx + dy * dy)
                
                # 2. Appearance cost (gallery-based)
                app_sim = gallery.compare(det.get('_appearance'))
                app_cost = (1.0 - max(0.0, app_sim)) * self.APPEARANCE_COST_WEIGHT
                # During occlusion, increase appearance weight (it's the most
                # reliable discriminator when positions overlap)
                if is_occluded:
                    app_cost *= 1.5
                
                # 3. Size consistency cost
                size_cost = 0.0
                if last_bbox is not None:
                    last_area = (last_bbox[2] - last_bbox[0]) * (last_bbox[3] - last_bbox[1])
                    curr_area = (det['bbox'][2] - det['bbox'][0]) * (det['bbox'][3] - det['bbox'][1])
                    if last_area > 0 and curr_area > 0:
                        ratio = curr_area / last_area
                        size_cost = abs(np.log(max(ratio, 0.1))) * self.SIZE_COST_WEIGHT
                
                # 4. Keypoint height consistency
                height_cost = 0.0
                det_height = det.get('_height_est', 0.0)
                if player_height > 0 and det_height > 0:
                    height_ratio = det_height / player_height
                    height_cost = abs(np.log(max(height_ratio, 0.3))) * self.KEYPOINT_HEIGHT_WEIGHT
                
                # Combined cost
                cost = pos_cost + app_cost + size_cost + height_cost
                
                # Bonus: ByteTrack ID consistency reduces cost
                bt_id = det.get('track_id')
                if bt_id is not None and self.bt_to_stable.get(bt_id) == sid:
                    cost *= self.BYTETRACK_BONUS
                
                cost_matrix[i][j] = cost
        
        # Solve optimal assignment
        row_ind, col_ind = self._solve_assignment(cost_matrix)
        
        # --- Anti-swap check ---
        # If two players swapped assignments compared to previous frame,
        # and they were close to each other, add penalty and re-solve.
        if self._prev_det_centers and len(row_ind) >= 2:
            needs_resolve = False
            proposed = {}  # sid -> det_index
            for r, c in zip(row_ind, col_ind):
                proposed[stable_ids[r]] = c
            
            for sid_a in proposed:
                for sid_b in proposed:
                    if sid_a >= sid_b:
                        continue
                    prev_a = self._prev_assignment.get(sid_a)
                    prev_b = self._prev_assignment.get(sid_b)
                    curr_a = proposed[sid_a]
                    curr_b = proposed[sid_b]
                    
                    # Check if they swapped and are involved in occlusion
                    if (prev_a is not None and prev_b is not None
                            and frozenset((curr_a, curr_b)) in occlusion_pairs):
                        # Players are overlapping — add swap penalty
                        idx_a = stable_ids.index(sid_a)
                        idx_b = stable_ids.index(sid_b)
                        cost_matrix[idx_a, curr_a] += self.SWAP_PENALTY
                        cost_matrix[idx_b, curr_b] += self.SWAP_PENALTY
                        needs_resolve = True
            
            if needs_resolve:
                row_ind, col_ind = self._solve_assignment(cost_matrix)
        
        # Apply assignments
        assigned_dets = set()
        new_assignment: Dict[int, int] = {}
        
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] > self.MAX_COST_THRESHOLD:
                continue
            
            sid = stable_ids[r]
            det = detections[c]
            det['stable_player_id'] = sid
            assigned_dets.add(c)
            new_assignment[sid] = c
            
            player = self.players[sid]
            
            # Kalman update
            player['kalman'].update(det['center'])
            
            # Update state
            player['positions'].append(det['center'])
            player['bboxes'].append(det['bbox'])
            player['frames_lost'] = 0
            
            # Update appearance gallery (skip during heavy occlusion to avoid drift)
            is_occluded = c in occluded_det_indices
            if not is_occluded and det.get('_appearance') is not None:
                player['gallery'].update(det['_appearance'])
            
            # Update height estimate (EMA)
            det_height = det.get('_height_est', 0.0)
            if det_height > 0:
                old_h = player.get('height_estimate', 0.0)
                if old_h > 0:
                    player['height_estimate'] = 0.9 * old_h + 0.1 * det_height
                else:
                    player['height_estimate'] = det_height
            
            # Update ByteTrack mapping
            bt_id = det.get('track_id')
            if bt_id is not None:
                self.bt_to_stable[bt_id] = sid
            
            # Trim history
            if len(player['positions']) > self.history_len:
                player['positions'] = player['positions'][-self.history_len:]
                player['bboxes'] = player['bboxes'][-self.history_len:]
        
        # Save assignment for anti-swap check next frame
        self._prev_assignment = new_assignment
        self._prev_det_centers = [d['center'] for d in detections]
        
        # Unmatched detections
        for j, det in enumerate(detections):
            if j not in assigned_dets:
                det['stable_player_id'] = None
    
    def _advance_lost_players(self):
        """
        For players not matched this frame, advance their Kalman filter
        (predict-only) so that re-identification is possible in future frames.
        Removes players lost for too many consecutive frames.
        """
        to_remove = []
        for sid, player in self.players.items():
            if player['frames_lost'] > 0 or sid not in [
                det_sid for det_sid in self._prev_assignment
            ]:
                player['frames_lost'] = player.get('frames_lost', 0) + 1
                if player['frames_lost'] <= self.LOST_PLAYER_MAX_FRAMES:
                    # Keep predicting position via Kalman
                    player['kalman'].predict_without_update()
                else:
                    to_remove.append(sid)
        
        for sid in to_remove:
            logger.warning(f"Player {sid} lost for >{self.LOST_PLAYER_MAX_FRAMES} frames, removing.")
            del self.players[sid]
    
    # ------------------------------------------------------------------
    # Assignment solver (exact, brute-force for small N)
    # ------------------------------------------------------------------
    
    @staticmethod
    def _solve_assignment(cost_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Optimal assignment for small matrices. For 4 players × N detections
        the search space is tiny (C(N,4)*4! ≤ a few thousand), so brute force
        gives the globally optimal solution.
        """
        n_rows, n_cols = cost_matrix.shape
        if n_rows == 0 or n_cols == 0:
            return np.array([], dtype=int), np.array([], dtype=int)
        
        k = min(n_rows, n_cols)
        best_cost = float('inf')
        best_row_perm = None
        best_col_combo = None
        
        for col_combo in combinations(range(n_cols), k):
            for row_perm in permutations(range(n_rows), k):
                total = sum(cost_matrix[row_perm[idx], col_combo[idx]] for idx in range(k))
                if total < best_cost:
                    best_cost = total
                    best_row_perm = row_perm
                    best_col_combo = col_combo
        
        if best_row_perm is None:
            return np.array([], dtype=int), np.array([], dtype=int)
        
        return np.array(best_row_perm, dtype=int), np.array(best_col_combo, dtype=int)
    
    # ------------------------------------------------------------------
    # Appearance features
    # ------------------------------------------------------------------
    
    @staticmethod
    def _extract_appearance(frame: np.ndarray, bbox: List[int]) -> Optional[np.ndarray]:
        """
        Extract an HSV color histogram from the upper body (torso) region
        of the detection, which is the most discriminative area for
        distinguishing players by shirt color.
        
        Uses a multi-region approach: torso (main) + lower body (secondary)
        for richer appearance representation.
        """
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        
        crop_h = crop.shape[0]
        
        # Upper third ≈ torso/shirt (primary feature)
        torso_end = max(1, crop_h // 3)
        torso = crop[:torso_end, :]
        
        # Middle third ≈ waist/shorts (secondary feature)
        mid_start = torso_end
        mid_end = max(mid_start + 1, 2 * crop_h // 3)
        mid_section = crop[mid_start:mid_end, :]
        
        if torso.size == 0:
            torso = crop
        
        hsv_torso = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
        hist_torso = cv2.calcHist([hsv_torso], [0, 1], None, [12, 8], [0, 180, 0, 256])
        cv2.normalize(hist_torso, hist_torso)
        
        if mid_section.size > 0:
            hsv_mid = cv2.cvtColor(mid_section, cv2.COLOR_BGR2HSV)
            hist_mid = cv2.calcHist([hsv_mid], [0, 1], None, [12, 8], [0, 180, 0, 256])
            cv2.normalize(hist_mid, hist_mid)
            # Concatenate: torso (weighted more) + mid-body
            combined = np.concatenate([hist_torso.flatten() * 0.7, hist_mid.flatten() * 0.3])
        else:
            combined = hist_torso.flatten()
        
        return combined
    
    @staticmethod
    def _compare_appearance(hist1: Optional[np.ndarray], hist2: Optional[np.ndarray]) -> float:
        """Correlation-based similarity between two HSV histograms (0..1)."""
        if hist1 is None or hist2 is None:
            return 0.0
        # Ensure same length
        min_len = min(len(hist1), len(hist2))
        h1 = hist1[:min_len].reshape(-1, 1).astype(np.float32)
        h2 = hist2[:min_len].reshape(-1, 1).astype(np.float32)
        return float(cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL))
    
    # ------------------------------------------------------------------
    # Motion prediction (legacy fallback, Kalman is primary)
    # ------------------------------------------------------------------
    
    def _predict_position(self, stable_id: int) -> Tuple[float, float]:
        """
        Predict the next position. Uses Kalman filter if available,
        falls back to simple velocity averaging.
        """
        player = self.players.get(stable_id)
        if player is None:
            return (0.0, 0.0)
        
        kf = player.get('kalman')
        if kf is not None:
            return kf.predict()
        
        # Fallback: simple velocity
        positions = player['positions']
        if not positions:
            return (0.0, 0.0)
        if len(positions) < 2:
            return positions[-1]
        
        n = min(5, len(positions))
        recent = positions[-n:]
        vx = (recent[-1][0] - recent[0][0]) / (n - 1)
        vy = (recent[-1][1] - recent[0][1]) / (n - 1)
        return (positions[-1][0] + vx, positions[-1][1] + vy)


class PlayerTracker:
    """
    Tracks player movements throughout a padel match video.
    
    Uses YOLO for person detection and tracking across frames.
    """
    
    def __init__(self, config: Any, use_pose_estimation: bool = False):
        """
        Initialize the PlayerTracker.
        
        Args:
            config: Configuration object containing tracking settings
            use_pose_estimation: If True, integrate pose estimation with player tracking
        """
        self.config = config
        self.detection_model = None
        self.use_pose_estimation = use_pose_estimation
        self.pose_estimator = None
        self._load_models()
        
    def _load_models(self):
        """Load YOLO model for player detection and optionally pose estimator."""
        try:
            from ultralytics import YOLO
            from src.utils.device import get_device
            
            # Get device
            device = get_device(self.config.model.device)
            
            # If pose estimation is enabled, use YOLOv8-pose directly for efficiency
            # This does detection + tracking + pose in one pass (much faster!)
            if self.use_pose_estimation:
                # Use MEDIUM model for better detection of small/distant players
                pose_model_name = getattr(self.config.pose, 'pose_model', 'yolov8m-pose.pt')
                logger.info(f"Loading YOLOv8-pose for integrated tracking + pose: {pose_model_name}")
                self.detection_model = YOLO(pose_model_name)
                self.detection_model.to(device)
                self.pose_estimator = None  # Not needed, pose model does everything
                logger.info(f"Hybrid tracker initialized (pose mode) with device: {device}")
            else:
                # Load standard detection model
                detection_model_name = self.config.model.player_model
                logger.info(f"Loading player detection model: {detection_model_name}")
                self.detection_model = YOLO(detection_model_name)
                self.detection_model.to(device)
                self.pose_estimator = None
                logger.info(f"Player tracker initialized with device: {device}")
            
        except ImportError:
            logger.warning(
                "Ultralytics YOLO not installed. Install with: pip install ultralytics"
            )
            self.detection_model = None
        except Exception as e:
            logger.error(f"Failed to load player tracking model: {e}")
            self.detection_model = None
    
    def track(self, video_data: Dict[str, Any], field_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Track players throughout the video.
        
        Uses PlayerIdentityManager to maintain stable IDs (1-4) across frames,
        preventing identity swaps when players are near each other.
        
        Args:
            video_data: Video data from VideoLoader
            field_info: Field detection information (used to filter detections)
            
        Returns:
            List of player tracking data:
            - player_id: Stable identifier (1-4)
            - positions: List of (x, y) tuples per frame
            - bounding_boxes: List of bounding boxes for each frame
            - team: "A" (players 1-2) or "B" (players 3-4)
            - confidence_scores: Detection confidence per frame
        """
        logger.info("Starting player tracking with identity stabilization")
        
        if self.detection_model is None:
            logger.warning("Player tracking model not loaded, returning empty tracks")
            return []
        
        capture = video_data.get("capture")
        if capture is None:
            raise ValueError("Video not properly loaded")
        
        metadata = video_data.get("metadata", {})
        frame_count = metadata.get("frame_count", 0)
        
        # Reset video to beginning
        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # Identity manager keeps stable player IDs across frames
        tracking_cfg = getattr(self.config, 'tracking', None)
        identity_mgr = PlayerIdentityManager(max_players=4, config=tracking_cfg)
        
        # Track players frame by frame
        all_detections = []
        
        logger.info(f"Processing {frame_count} frames for player tracking...")
        
        frame_idx = 0
        while True:
            ret, frame = capture.read()
            if not ret:
                break
            
            # Detect players in frame (ByteTrack assigns volatile track_id)
            detections = self.detect_players_in_frame(
                frame, 
                frame_idx, 
                field_mask=field_info.get("court_mask")
            )
            
            # Stabilize identities: assigns 'stable_player_id' to each detection
            detections = identity_mgr.update(frame, detections)
            
            all_detections.extend(detections)
            
            frame_idx += 1
            
            # Log progress
            if frame_idx % 100 == 0:
                logger.info(f"Processed {frame_idx}/{frame_count} frames")
        
        logger.info(f"Total detections: {len(all_detections)}")
        
        # Associate detections into tracks using stable IDs
        player_tracks = self._associate_tracks(all_detections, field_info)
        
        logger.info(f"Created {len(player_tracks)} player tracks")
        
        return player_tracks
    
    def detect_players_in_frame(
        self, 
        frame: np.ndarray, 
        frame_number: int,
        field_mask: Optional[np.ndarray] = None
    ) -> List[Dict[str, Any]]:
        """
        Detect players in a single frame.
        If pose estimation is enabled, uses YOLOv8-pose for integrated detection+tracking+pose.
        Otherwise, uses standard YOLO detection.
        
        Args:
            frame: Video frame to process
            frame_number: Frame index in video
            field_mask: Optional court mask to filter detections
            
        Returns:
            List of detected players with bounding boxes, keypoints (if pose mode), and confidence scores
        """
        if self.detection_model is None:
            return []
        
        try:
            # Run YOLO detection/tracking with very low confidence for better recall
            # Important: Players far from camera (top of court) have lower confidence
            # Use higher input resolution to better detect small/distant objects
            min_conf = 0.05  # Extremely permissive to catch distant players
            
            results = self.detection_model.track(
                frame, 
                persist=True,
                classes=[0],  # 0 = person in COCO dataset
                conf=min_conf,
                verbose=False,
                iou=0.3,  # Lower IOU for better tracking of small/distant players
                max_det=30,  # Increased to ensure we catch all players
                imgsz=1280,  # Higher resolution input for better small object detection
                tracker="bytetrack.yaml"
            )
            
            detections = []
            
            for result in results:
                boxes = result.boxes
                
                if boxes is None or len(boxes) == 0:
                    continue
                
                # Check if pose keypoints are available (YOLOv8-pose model)
                has_keypoints = hasattr(result, 'keypoints') and result.keypoints is not None
                
                for i, box in enumerate(boxes):
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0].cpu().numpy())
                    
                    # Get track ID if available
                    track_id = int(box.id[0].cpu().numpy()) if box.id is not None else None
                    
                    # Calculate center point
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    
                    # Calculate bottom center (feet position) which is more reliable for court filtering
                    bottom_center_x = int((x1 + x2) / 2)
                    bottom_center_y = int(y2)  # Bottom of bounding box
                    
                    # Very lenient filtering for field mask
                    # Only filter obvious non-players (e.g., spectators far from court)
                    # For padel, be extremely permissive to avoid losing distant players
                    if field_mask is not None:
                        h, w = field_mask.shape
                        # Very generous margin - 100px for distant/small players
                        margin = 100
                        
                        # Check if feet position is within expanded court area
                        if 0 <= bottom_center_y < h and 0 <= bottom_center_x < w:
                            # Check surrounding area before filtering
                            y_min = max(0, bottom_center_y - margin)
                            y_max = min(h, bottom_center_y + margin)
                            x_min = max(0, bottom_center_x - margin)
                            x_max = min(w, bottom_center_x + margin)
                            
                            # Only filter if NO part of the margin area is on court
                            # AND confidence is very low (likely not a player)
                            if np.sum(field_mask[y_min:y_max, x_min:x_max]) == 0:
                                # Keep even low confidence if bbox is reasonably sized
                                bbox_area = (x2 - x1) * (y2 - y1)
                                if confidence < 0.3 and bbox_area < 500:  # Very small and low conf
                                    continue
                    
                    detection = {
                        "frame_number": frame_number,
                        "bbox": [int(x1), int(y1), int(x2), int(y2)],
                        "center": (center_x, center_y),
                        "confidence": confidence,
                        "track_id": track_id
                    }
                    
                    # Extract keypoints if available (YOLOv8-pose)
                    if has_keypoints and i < len(result.keypoints):
                        kpts = result.keypoints[i]
                        
                        # Extract xy coordinates and confidence
                        if hasattr(kpts, 'xy'):
                            kpts_xy = kpts.xy[0].cpu().numpy()  # Shape: (17, 2)
                        else:
                            kpts_xy = kpts.data[0, :, :2].cpu().numpy()
                        
                        if hasattr(kpts, 'conf'):
                            kpts_conf = kpts.conf[0].cpu().numpy()  # Shape: (17,)
                        else:
                            kpts_conf = kpts.data[0, :, 2].cpu().numpy()
                        
                        detection["keypoints"] = kpts_xy
                        detection["keypoints_conf"] = kpts_conf
                    else:
                        detection["keypoints"] = None
                        detection["keypoints_conf"] = None
                    
                    detections.append(detection)
            
            return detections
            
        except Exception as e:
            logger.warning(f"Error detecting players in frame {frame_number}: {e}")
            return []
    
    def _calculate_iou(self, bbox1: List[int], bbox2: List[int]) -> float:
        """
        Calculate Intersection over Union (IoU) between two bounding boxes.
        
        Args:
            bbox1: First bounding box [x1, y1, x2, y2]
            bbox2: Second bounding box [x1, y1, x2, y2]
            
        Returns:
            IoU score (0.0 to 1.0)
        """
        # Calculate intersection
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        if x2 < x1 or y2 < y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        
        # Calculate union
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def _associate_tracks(
        self, 
        all_detections: List[Dict[str, Any]], 
        field_info: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Associate detections into continuous player tracks.
        
        Uses 'stable_player_id' assigned by PlayerIdentityManager when available,
        falling back to ByteTrack 'track_id' for backward compatibility (e.g.
        when called directly from test scripts without the identity manager).
        
        Args:
            all_detections: All detections from all frames
            field_info: Field information for context
            
        Returns:
            List of player tracks with player_id 1-4, team A/B
        """
        # Decide which ID field to group by
        has_stable = any(d.get("stable_player_id") is not None for d in all_detections)
        id_field = "stable_player_id" if has_stable else "track_id"
        
        if has_stable:
            logger.info("Associating tracks using stable player IDs (identity-stabilized)")
        else:
            logger.info("Associating tracks using ByteTrack IDs (no identity manager)")
        
        # Group detections by the chosen ID
        tracks_dict = defaultdict(lambda: {
            "positions": [],
            "bounding_boxes": [],
            "confidence_scores": [],
            "frame_numbers": [],
            "keypoints_sequence": [],
            "keypoints_conf_sequence": []
        })
        
        for detection in all_detections:
            pid = detection.get(id_field)
            if pid is None:
                continue
            
            tracks_dict[pid]["positions"].append(detection["center"])
            tracks_dict[pid]["bounding_boxes"].append(detection["bbox"])
            tracks_dict[pid]["confidence_scores"].append(detection["confidence"])
            tracks_dict[pid]["frame_numbers"].append(detection["frame_number"])
            
            if "keypoints" in detection and detection["keypoints"] is not None:
                tracks_dict[pid]["keypoints_sequence"].append(detection["keypoints"])
                tracks_dict[pid]["keypoints_conf_sequence"].append(detection["keypoints_conf"])
            else:
                tracks_dict[pid]["keypoints_sequence"].append(None)
                tracks_dict[pid]["keypoints_conf_sequence"].append(None)
        
        # Build raw track list
        all_tracks_raw = []
        for pid, track_data in tracks_dict.items():
            all_tracks_raw.append({
                "player_id": pid,
                "positions": track_data["positions"],
                "bounding_boxes": track_data["bounding_boxes"],
                "confidence_scores": track_data["confidence_scores"],
                "frame_numbers": track_data["frame_numbers"],
                "keypoints_sequence": track_data["keypoints_sequence"],
                "keypoints_conf_sequence": track_data["keypoints_conf_sequence"],
                "team": None,
                "track_length": len(track_data["positions"])
            })
        
        # ----- Filtering / selection -----
        if has_stable:
            # With stable IDs we already have at most 4 tracks → keep all
            player_tracks = all_tracks_raw
        else:
            # Legacy path: adaptive filtering like before
            player_tracks = self._filter_tracks_legacy(all_tracks_raw, len(all_detections))
        
        # Assign teams (Player 1-2 → A, Player 3-4 → B)
        if len(player_tracks) >= 2:
            player_tracks = self.assign_teams(player_tracks, field_info)
        
        if len(player_tracks) != 4:
            logger.warning(
                f"Expected 4 players but found {len(player_tracks)} tracks. "
                "Consider adjusting confidence threshold or tracking parameters."
            )
        
        return player_tracks
    
    def _filter_tracks_legacy(
        self, all_tracks_raw: List[Dict[str, Any]], total_detections: int
    ) -> List[Dict[str, Any]]:
        """
        Legacy track filtering for when PlayerIdentityManager is not used
        (e.g. direct calls from test scripts).
        """
        if len(all_tracks_raw) <= 4:
            all_tracks = all_tracks_raw
            logger.info(f"Found {len(all_tracks)} tracks - keeping all (expecting 4 players)")
        elif len(all_tracks_raw) <= 6:
            min_len = max(3, total_detections // 100)
            all_tracks = [t for t in all_tracks_raw if t["track_length"] >= min_len]
            logger.info(f"Filtered {len(all_tracks_raw)} to {len(all_tracks)} (min_length={min_len})")
        else:
            min_len = max(10, total_detections // 50)
            all_tracks = [t for t in all_tracks_raw if t["track_length"] >= min_len]
            logger.info(f"Filtered {len(all_tracks_raw)} to {len(all_tracks)} (min_length={min_len})")
        
        if len(all_tracks) > 4:
            scored = []
            for t in all_tracks:
                cov = len(t["positions"])
                avg_c = np.mean(t["confidence_scores"]) if t["confidence_scores"] else 0
                scored.append((t, cov * 0.8 + avg_c * cov * 0.2))
            scored.sort(key=lambda x: x[1], reverse=True)
            all_tracks = [t for t, _ in scored[:4]]
            logger.info(f"Kept top 4 player tracks out of {len(scored)}")
        
        return all_tracks
    
    def assign_teams(
        self, 
        player_tracks: List[Dict[str, Any]], 
        field_info: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Assign players to teams.
        
        When stable IDs from PlayerIdentityManager are used (IDs 1-4), the
        assignment is deterministic:
            - Player 1, Player 2 → Team A  (top of court / closer to camera far side)
            - Player 3, Player 4 → Team B  (bottom of court / closer to camera)
        
        This is because PlayerIdentityManager initializes IDs sorted by Y-position
        (ascending), so players with lower Y (top of image = far side of court)
        get IDs 1-2, and players with higher Y (bottom = near side) get IDs 3-4.
        
        Falls back to position-based assignment for legacy/non-stabilized tracks.
        """
        # Check if we have stable IDs (integer 1-4)
        stable_ids = {t["player_id"] for t in player_tracks}
        ids_are_stable = stable_ids.issubset({1, 2, 3, 4}) and len(stable_ids) >= 2
        
        if ids_are_stable:
            # Deterministic assignment: 1-2 → A, 3-4 → B
            for track in player_tracks:
                pid = track["player_id"]
                track["team"] = "A" if pid <= 2 else "B"
            
            team_a = [t["player_id"] for t in player_tracks if t["team"] == "A"]
            team_b = [t["player_id"] for t in player_tracks if t["team"] == "B"]
            logger.info(f"Team A (stable): Players {sorted(team_a)}, "
                       f"Team B (stable): Players {sorted(team_b)}")
            return player_tracks
        
        # Try color-based assignment
        if self.config.tracking.use_keypoints_for_team:
            if self._assign_teams_by_color(player_tracks):
                logger.info("Team assignment: Using shirt color from keypoints")
                return player_tracks
        
        # Fallback: Position-based team assignment
        logger.info("Team assignment: Using court position (fallback)")
        return self._assign_teams_by_position(player_tracks, field_info)
    
    def _assign_teams_by_position(
        self, 
        player_tracks: List[Dict[str, Any]], 
        field_info: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Assign teams based on Y-position (net divides court horizontally).
        
        Args:
            player_tracks: List of player tracking data
            field_info: Field information for court context
            
        Returns:
            Updated player tracks with team assignments
        """
        # Calculate average y-position for each player to determine which side of net
        player_avg_positions = []
        for track in player_tracks:
            positions = track["positions"]
            if len(positions) == 0:
                continue
            
            # Use median y-position (more robust to outliers)
            avg_y = np.median([pos[1] for pos in positions])
            avg_x = np.median([pos[0] for pos in positions])
            player_avg_positions.append((track["player_id"], avg_y, avg_x))
        
        if len(player_avg_positions) < 2:
            return player_tracks
        
        # Sort by y-position (vertical position on court)
        player_avg_positions.sort(key=lambda x: x[1])
        
        # For padel: net divides court horizontally
        # Split players into two groups based on y-position (top/bottom of court)
        # Each group is a team (2 players per team for standard padel doubles)
        
        if len(player_avg_positions) == 2:
            # Only 2 players detected - assign to different teams
            team_a_ids = [player_avg_positions[0][0]]
            team_b_ids = [player_avg_positions[1][0]]
        elif len(player_avg_positions) == 3:
            # 3 players - assign based on gaps in y-position
            # Find the largest gap to determine net position
            gaps = []
            for i in range(len(player_avg_positions) - 1):
                gap = player_avg_positions[i+1][1] - player_avg_positions[i][1]
                gaps.append((i, gap))
            
            # Split at largest gap (likely the net)
            max_gap_idx = max(gaps, key=lambda x: x[1])[0]
            team_a_ids = [pid for pid, _, _ in player_avg_positions[:max_gap_idx+1]]
            team_b_ids = [pid for pid, _, _ in player_avg_positions[max_gap_idx+1:]]
        else:
            # 4 or more players
            # For 4 players: split in half (2 per team)
            # For more: try to identify the 4 main players and assign others
            mid_point = len(player_avg_positions) // 2
            
            # Calculate y-position gaps to find natural split (the net)
            if len(player_avg_positions) >= 4:
                # Find largest gap in y-positions - this should be the net
                gaps = []
                for i in range(len(player_avg_positions) - 1):
                    gap = player_avg_positions[i+1][1] - player_avg_positions[i][1]
                    gaps.append((i, gap))
                
                # If there's a clear gap, use it to split teams
                max_gap = max(gaps, key=lambda x: x[1])
                if max_gap[1] > 50:  # Significant gap (likely the net)
                    mid_point = max_gap[0] + 1
            
            team_a_ids = [pid for pid, _, _ in player_avg_positions[:mid_point]]
            team_b_ids = [pid for pid, _, _ in player_avg_positions[mid_point:]]
        
        # Update tracks with team assignments
        for track in player_tracks:
            if track["player_id"] in team_a_ids:
                track["team"] = "A"
            elif track["player_id"] in team_b_ids:
                track["team"] = "B"
        
        logger.info(f"Team A: {len(team_a_ids)} players {team_a_ids}, Team B: {len(team_b_ids)} players {team_b_ids}")
        
        return player_tracks
    
    def _assign_teams_by_color(self, player_tracks: List[Dict[str, Any]]) -> bool:
        """
        Assign teams based on shirt color extracted from torso keypoints.
        Uses k-means clustering to identify two dominant colors.
        
        Args:
            player_tracks: List of player tracking data with keypoints
            
        Returns:
            True if successful, False if not enough data
        """
        # This is a placeholder for future implementation
        # Requires:
        # 1. Extract torso region using shoulder/hip keypoints
        # 2. Sample dominant color from torso
        # 3. Cluster players into 2 groups by color
        # 4. Assign teams based on clusters
        
        logger.debug("Color-based team assignment not yet implemented")
        return False


# Import cv2 here to avoid circular imports at module level
try:
    import cv2
except ImportError:
    logger.warning("OpenCV not installed")
