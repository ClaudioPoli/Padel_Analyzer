# Ball Detection and Field Perspective Improvements

## User Feedback

The user reported:
1. **Ball identification issues** - "Most times it is not identified"
2. **Field perspective issue** - Lower side correct, but upper side needs perspective correction (should be shorter)

## Changes Implemented

### 1. Enhanced Ball Detection

#### Problem
Small, fast-moving padel balls were being missed frequently.

#### Solutions Implemented

**A. Player Proximity Hints**
- Ball tracker now receives player tracking data
- Builds frame-by-frame player position lookup
- Boosts confidence by 20% for balls detected near players (within 300px)
- Rationale: User noted "ball can be near to the players most of times"

**B. Motion-Based Detection (NEW)**
- Added `_detect_ball_by_motion()` method
- Uses previous frame's ball position to predict location
- Searches in 100px radius around previous position
- Useful for fast-moving balls that might be blurred
- Confidence based on proximity to previous position

**C. More Sensitive Detection Parameters**
- YOLO confidence: 0.15 → **0.10** (more sensitive)
- Circle detection minRadius: 3 → **2** (catches smaller balls)
- Circle detection param2: 20 → **18** (more sensitive)
- Color detection area minimum: 20 → **10** pixels (smaller balls)
- Color detection circularity: 0.4 → **0.35** (more lenient)
- Morphological kernel: 3x3 → **2x2** (preserves small features)

**D. Broader Color Ranges**
- Yellow range: [20,80,80] to [45,255,255] → **[18,60,60] to [48,255,255]**
- Bright yellow: [15,100,150] to [35,255,255] → **[12,80,120] to [38,255,255]**
- White: [0,0,200] to [180,30,255] → **[0,0,180] to [180,40,255]**
- More lenient thresholds to catch balls in different lighting

**E. Temporal Tracking**
- `detect_ball_in_frame()` now accepts `prev_ball_pos` parameter
- Uses temporal continuity to improve tracking
- Helps with fast-moving balls across frames

### 2. Field Perspective Correction

#### Problem
Field detection didn't properly account for perspective - upper (far) side should be shorter than lower (near) side.

#### Solutions Implemented

**A. Perspective-Aware Corner Sorting**
- Added `_sort_corners_with_perspective()` method
- Verifies that upper width < lower width
- Logs warning if perspective seems reversed
- Auto-corrects if camera angle is unusual

**B. Better Quadrilateral Approximation**
- Tries multiple epsilon values (0.01, 0.02, 0.03, 0.04, 0.05)
- Finds best approximation to 4 corners
- More robust for different court shapes

**C. Perspective Validation**
```python
upper_width = top_pts[1][0] - top_pts[0][0]
lower_width = bottom_pts[1][0] - bottom_pts[0][0]

# Verify perspective is correct
if upper_width > lower_width * 1.2:
    logger.warning("Unusual perspective detected")
    # Swap if needed
```

## Technical Details

### Ball Detection Flow

```python
def detect_ball_in_frame(frame, frame_idx, field_mask, prev_ball_pos):
    detections = []
    player_positions = get_player_positions(frame_idx)
    
    # Try 4 detection methods
    1. YOLO model (conf=0.10)
    2. Hough circles (minRadius=2, param2=18)
    3. Color segmentation (broadened ranges)
    4. Motion-based (if prev_ball_pos available)
    
    # Boost confidence for balls near players
    for each detection:
        if distance_to_nearest_player < 300px:
            confidence *= 1.2
    
    return best_detection
```

### Motion-Based Detection

```python
def _detect_ball_by_motion(frame, prev_ball_pos, field_mask):
    # Define search region (100px radius)
    search_region = prev_position ± 100px
    
    # Look for yellow ball in ROI
    # Also look for bright objects (overexposure)
    
    # Find contour closest to previous position
    confidence = 1.0 - (distance / search_radius)
    
    return (x, y, confidence)
```

### Field Perspective Sorting

```python
def _sort_corners_with_perspective(corners):
    # Sort by y-coordinate
    top_pts = corners[:2]  # Far from camera
    bottom_pts = corners[-2:]  # Near camera
    
    # Calculate widths
    upper_width = top_right.x - top_left.x
    lower_width = bottom_right.x - bottom_left.x
    
    # Verify: upper_width should be < lower_width
    if upper_width > lower_width * 1.2:
        # Swap (unusual camera angle)
        swap(top_pts, bottom_pts)
    
    return [top_left, top_right, bottom_right, bottom_left]
```

## Expected Improvements

### Ball Detection
- **Better coverage**: Should detect balls in more frames
- **Better small ball detection**: Minimum size reduced from 20 to 10 pixels
- **Better fast ball tracking**: Motion-based detection helps with blur
- **Better player interaction detection**: Proximity boost helps during play

### Field Perspective
- **Correct perspective**: Upper side will be shorter than lower side
- **Better visualization**: Court overlay will properly show trapezoid shape
- **More accurate**: Respects camera viewing angle

## Testing

To test improvements:
```python
from padel_analyzer import PadelAnalyzer

analyzer = PadelAnalyzer()
results = analyzer.analyze_video("rally.mp4")

# Check field perspective
corners = results['field_info']['corners'][:4]
upper_width = corners[1][0] - corners[0][0]
lower_width = corners[2][0] - corners[3][0]
print(f"Perspective ratio: {upper_width/lower_width:.2f}")  # Should be < 1.0

# Check ball detection
detections = len(results['ball_tracks']['positions'])
frames = results['metadata']['frame_count']
print(f"Ball detection: {detections}/{frames} ({detections/frames*100:.1f}%)")
```

## Summary of Changes

**Files Modified:**
1. `padel_analyzer/tracking/ball_tracker.py`
   - Added player_positions_by_frame lookup
   - Enhanced detect_ball_in_frame with proximity boost
   - Lowered detection thresholds
   - Broadened color ranges
   - Added _detect_ball_by_motion method
   - Updated all detection methods to use player_positions

2. `padel_analyzer/detection/field_detector.py`
   - Added _sort_corners_with_perspective method
   - Enhanced quadrilateral approximation
   - Added perspective validation

3. `padel_analyzer/analyzer.py`
   - Pass player_tracks to ball_tracker.track()

**Key Improvements:**
- ✓ Ball detection more sensitive to small balls
- ✓ Ball detection uses player proximity hints
- ✓ Ball detection tracks motion across frames
- ✓ Field perspective properly accounts for camera angle
- ✓ Upper court side correctly shorter than lower side
