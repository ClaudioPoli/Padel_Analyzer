"""
Field detection module for identifying and mapping the padel court.
"""

from typing import Dict, Any, List, Tuple, Optional
import logging
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class FieldDetector:
    """
    Detects and identifies the padel field in video frames.
    
    Automatically identifies court boundaries, lines, and key areas using
    computer vision techniques (line detection, edge detection, etc.).
    """
    
    def __init__(self, config: Any):
        """
        Initialize the FieldDetector.
        
        Args:
            config: Configuration object containing detection settings
        """
        self.config = config
        
    def detect(self, video_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect the padel field in the video.
        
        Args:
            video_data: Video data from VideoLoader
            
        Returns:
            Dictionary containing field information:
            - boundaries: Court boundary coordinates
            - lines: Detected court lines (service line, center line, etc.)
            - corners: Corner points of the court
            - homography_matrix: Transformation matrix for top-down view
            - court_mask: Binary mask of court area
        """
        logger.info("Starting field detection")
        
        capture = video_data.get("capture")
        if capture is None:
            raise ValueError("Video not properly loaded")
        
        # Get a representative frame (middle of video)
        metadata = video_data.get("metadata", {})
        frame_count = metadata.get("frame_count", 0)
        
        if frame_count == 0:
            logger.warning("No frames in video")
            return self._empty_field_info()
        
        # Sample multiple frames and use the best one
        sample_frames = self._get_sample_frames(capture, frame_count)
        
        best_field_info = None
        best_confidence = 0.0
        
        for frame in sample_frames:
            field_info = self._detect_in_frame(frame)
            if field_info["confidence"] > best_confidence:
                best_confidence = field_info["confidence"]
                best_field_info = field_info
        
        if best_field_info is None:
            logger.warning("Could not detect field in any frame")
            return self._empty_field_info()
        
        logger.info(f"Field detected with confidence: {best_field_info['confidence']:.2f}")
        return best_field_info
    
    def _get_sample_frames(self, capture: cv2.VideoCapture, frame_count: int, num_samples: int = 5) -> List[np.ndarray]:
        """
        Get sample frames from different parts of the video.
        
        Args:
            capture: OpenCV VideoCapture object
            frame_count: Total number of frames
            num_samples: Number of frames to sample
            
        Returns:
            List of sampled frames
        """
        frames = []
        # Sample evenly distributed frames
        indices = np.linspace(frame_count // 4, 3 * frame_count // 4, num_samples, dtype=int)
        
        for idx in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = capture.read()
            if ret:
                frames.append(frame)
        
        return frames
    
    def _detect_in_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Detect field in a single frame.
        
        Args:
            frame: Video frame to process
            
        Returns:
            Field information dictionary
        """
        # Detect lines
        lines = self.detect_court_lines(frame)
        
        if len(lines) < 4:
            # Not enough lines detected
            return self._empty_field_info()
        
        # Detect corners from lines
        corners = self.detect_court_corners(lines)
        
        # Create court mask
        court_mask = self.create_court_mask(frame.shape[:2], corners)
        
        # Calculate confidence based on detected features
        confidence = self._calculate_confidence(lines, corners)
        
        # Estimate homography if we have corners
        homography_matrix = None
        if len(corners) >= 4 and self.config.field_detection.use_homography:
            homography_matrix = self.estimate_homography(corners)
        
        return {
            "boundaries": corners[:4] if len(corners) >= 4 else None,
            "lines": lines,
            "corners": corners,
            "homography_matrix": homography_matrix,
            "court_mask": court_mask,
            "confidence": confidence
        }
    
    def _empty_field_info(self) -> Dict[str, Any]:
        """Return empty field info structure."""
        return {
            "boundaries": None,
            "lines": [],
            "corners": [],
            "homography_matrix": None,
            "court_mask": None,
            "confidence": 0.0
        }
    
    def _calculate_confidence(self, lines: List, corners: List) -> float:
        """
        Calculate confidence score for field detection.
        
        Args:
            lines: Detected lines
            corners: Detected corners
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        # Simple heuristic: more lines and corners = higher confidence
        line_score = min(len(lines) / 10.0, 0.5)  # Max 0.5 from lines
        corner_score = min(len(corners) / 4.0, 0.5)  # Max 0.5 from corners
        
        return line_score + corner_score
    
    def detect_court_lines(self, frame: np.ndarray) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        Detect court lines in a single frame using Hough Line Transform.
        
        Args:
            frame: Video frame to process
            
        Returns:
            List of line segments as ((x1, y1), (x2, y2)) tuples
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection with adjusted thresholds for better line detection
        edges = cv2.Canny(blurred, 30, 100, apertureSize=3)
        
        # Detect lines using probabilistic Hough transform with relaxed parameters
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=max(50, self.config.field_detection.line_detection_threshold // 2),
            minLineLength=50,  # Reduced from 100 for more lines
            maxLineGap=20  # Increased from 10 for better connectivity
        )
        
        if lines is None:
            return []
        
        # Convert to expected format and filter
        line_segments = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Calculate line length
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            
            # Only keep reasonably long lines
            if length < 30:
                continue
            
            line_segments.append(((int(x1), int(y1)), (int(x2), int(y2))))
        
        return line_segments
    
    def detect_court_corners(self, lines: List[Tuple[Tuple[int, int], Tuple[int, int]]]) -> List[Tuple[int, int]]:
        """
        Detect court corners from detected lines.
        
        Args:
            lines: List of detected line segments
            
        Returns:
            List of corner coordinates (x, y)
        """
        if len(lines) < 2:
            return []
        
        corners = []
        tolerance = self.config.field_detection.corner_detection_tolerance
        
        # Find intersections between lines
        for i, line1 in enumerate(lines):
            for line2 in lines[i + 1:]:
                intersection = self._line_intersection(line1, line2)
                if intersection is not None:
                    x, y = intersection
                    # Check if this corner is far enough from existing corners
                    is_new = True
                    for cx, cy in corners:
                        if abs(x - cx) < tolerance and abs(y - cy) < tolerance:
                            is_new = False
                            break
                    
                    if is_new:
                        corners.append((int(x), int(y)))
        
        # If we have enough corners, try to find the largest quadrilateral
        # that represents the court
        if len(corners) >= 4:
            # Find the 4 corners that form the largest quadrilateral
            corners = self._find_court_quadrilateral(corners)
            return corners[:4]  # Return best 4
        
        return corners
    
    def _find_court_quadrilateral(self, corners: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Find the 4 corners that best represent the court boundary.
        
        Args:
            corners: List of detected corner points (will be converted to float32 for processing)
            
        Returns:
            4 corners representing the court
        """
        if len(corners) < 4:
            return corners
        
        # Convert to numpy array
        pts = np.array(corners, dtype=np.float32)
        
        # Find the convex hull - this often gives us the court boundary
        hull = cv2.convexHull(pts)
        hull_points = [tuple(pt[0]) for pt in hull]
        
        # If hull has exactly 4 points, use them
        if len(hull_points) == 4:
            return self._sort_corners(hull_points)
        
        # Otherwise, find 4 extreme points (top-left, top-right, bottom-right, bottom-left)
        # Sort by y-coordinate
        sorted_pts = pts[pts[:, 1].argsort()]
        
        # Top points (lowest y)
        top_candidates = sorted_pts[:len(sorted_pts)//2]
        top_left = tuple(top_candidates[top_candidates[:, 0].argmin()])
        top_right = tuple(top_candidates[top_candidates[:, 0].argmax()])
        
        # Bottom points (highest y)
        bottom_candidates = sorted_pts[len(sorted_pts)//2:]
        bottom_left = tuple(bottom_candidates[bottom_candidates[:, 0].argmin()])
        bottom_right = tuple(bottom_candidates[bottom_candidates[:, 0].argmax()])
        
        return self._sort_corners([top_left, top_right, bottom_right, bottom_left])
    
    def _line_intersection(
        self, 
        line1: Tuple[Tuple[int, int], Tuple[int, int]], 
        line2: Tuple[Tuple[int, int], Tuple[int, int]]
    ) -> Optional[Tuple[float, float]]:
        """
        Find intersection point of two line segments.
        
        Args:
            line1: First line segment ((x1, y1), (x2, y2))
            line2: Second line segment ((x1, y1), (x2, y2))
            
        Returns:
            Intersection point (x, y) or None if lines don't intersect
        """
        (x1, y1), (x2, y2) = line1
        (x3, y3), (x4, y4) = line2
        
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        
        if abs(denom) < 1e-6:  # Lines are parallel
            return None
        
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
        
        # Check if intersection is within both line segments (with some tolerance)
        if -0.1 <= t <= 1.1 and -0.1 <= u <= 1.1:
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            return (x, y)
        
        return None
    
    def _sort_corners(self, corners: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Sort corners in order: top-left, top-right, bottom-right, bottom-left.
        
        Args:
            corners: List of corner points
            
        Returns:
            Sorted list of corners
        """
        if len(corners) < 4:
            return corners
        
        # Convert to numpy array for easier computation
        pts = np.array(corners, dtype=np.float32)
        
        # Sort by y-coordinate
        sorted_pts = pts[pts[:, 1].argsort()]
        
        # Top two points
        top_pts = sorted_pts[:2]
        top_pts = top_pts[top_pts[:, 0].argsort()]  # Sort by x
        
        # Bottom two points
        bottom_pts = sorted_pts[-2:]
        bottom_pts = bottom_pts[bottom_pts[:, 0].argsort()]  # Sort by x
        
        # Order: top-left, top-right, bottom-right, bottom-left
        ordered = [
            tuple(top_pts[0]),
            tuple(top_pts[1]),
            tuple(bottom_pts[1]),
            tuple(bottom_pts[0])
        ]
        
        return [(int(x), int(y)) for x, y in ordered]
    
    def estimate_homography(self, corners: List[Tuple[int, int]]) -> Optional[np.ndarray]:
        """
        Estimate homography matrix for perspective transformation.
        
        Args:
            corners: Detected court corners (must be 4 points)
            
        Returns:
            Homography matrix or None if estimation fails
        """
        if len(corners) < 4:
            return None
        
        # Standard padel court dimensions (in meters, used for reference)
        # We'll map to a normalized coordinate system
        src_points = np.array(corners[:4], dtype=np.float32)
        
        # Define destination points (standard court rectangle)
        # Using a 20x10 meter court as reference (padel court dimensions)
        dst_points = np.array([
            [0, 0],
            [1000, 0],
            [1000, 2000],
            [0, 2000]
        ], dtype=np.float32)
        
        try:
            # Compute homography
            H, _ = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
            return H
        except Exception as e:
            logger.warning(f"Failed to compute homography: {e}")
            return None
    
    def create_court_mask(self, frame_shape: Tuple[int, int], corners: List[Tuple[int, int]]) -> Optional[np.ndarray]:
        """
        Create a binary mask of the court area with expanded boundaries.
        
        Args:
            frame_shape: Shape of the video frame (height, width)
            corners: Court corner points
            
        Returns:
            Binary mask of court area (expanded to include nearby areas)
        """
        if len(corners) < 4:
            return None
        
        height, width = frame_shape
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Expand the court boundaries to avoid filtering players at edges
        # Calculate center of court
        pts = np.array(corners[:4], dtype=np.float32)
        center = np.mean(pts, axis=0)
        
        # Expand each corner away from center by 15%
        expanded_pts = []
        for pt in pts:
            direction = pt - center
            expanded_pt = pt + direction * 0.15
            # Clamp to frame boundaries
            expanded_pt[0] = np.clip(expanded_pt[0], 0, width - 1)
            expanded_pt[1] = np.clip(expanded_pt[1], 0, height - 1)
            expanded_pts.append(expanded_pt)
        
        # Draw filled polygon with expanded boundaries
        expanded_pts = np.array(expanded_pts, dtype=np.int32)
        cv2.fillPoly(mask, [expanded_pts], 255)
        
        return mask
