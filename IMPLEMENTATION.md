# Video Processing Implementation

## Model Selection

After evaluating different approaches for padel video analysis, we selected **YOLOv8** with **PyTorch** as the primary framework for the following reasons:

### Why YOLOv8?

1. **Cross-Platform Support**:
   - Works with CUDA on Windows (NVIDIA GPUs)
   - Works with MPS on macOS (Apple Silicon)
   - Falls back to CPU when GPU unavailable
   
2. **Zero-Shot Capability**:
   - Pre-trained on COCO dataset with person detection (class 0)
   - Pre-trained with sports ball detection (class 32)
   - No fine-tuning required for basic functionality

3. **Built-in Tracking**:
   - YOLOv8 includes object tracking via `track()` method
   - Maintains consistent IDs across frames
   - Reduces need for separate tracking algorithms

4. **Performance**:
   - Fast inference (real-time capable)
   - Good accuracy on person and object detection
   - Multiple model sizes (nano to extra-large)

### Architecture Overview

The implementation follows a modular pipeline:

```
Video Input → Video Loader → Field Detection → Player Tracking → Ball Tracking → Results
```

#### 1. Video Loader (`video/video_loader.py`)
- Uses OpenCV (`cv2.VideoCapture`) for video I/O
- Supports: MP4, MOV, AVI, MKV formats
- Extracts metadata: FPS, resolution, frame count, duration
- Provides frame iteration and random access

#### 2. Field Detector (`detection/field_detector.py`)
- **Traditional CV Methods**:
  - Canny edge detection
  - Hough Line Transform for court lines
  - Line intersection for corner detection
  - Homography estimation for perspective correction
- **Purpose**: Identify court boundaries to filter false detections
- **Flexibility**: Works on videos with partial court visibility

#### 3. Player Tracker (`tracking/player_tracker.py`)
- **Model**: YOLOv8n (nano) by default
- **Method**: Detect persons (COCO class 0)
- **Tracking**: Built-in YOLO tracking with persistent IDs
- **Team Assignment**: Based on court position (simple heuristic)
- **Field Context**: Uses court mask to filter out-of-bounds detections

#### 4. Ball Tracker (`tracking/ball_tracker.py`)
- **Primary Method**: YOLO sports ball detection (COCO class 32)
- **Fallback Method**: Circular Hough Transform (traditional CV)
- **Interpolation**: Scipy-based trajectory interpolation for missing frames
- **Velocity Calculation**: Frame-to-frame motion analysis
- **In-Play Detection**: Movement-based heuristic

### Device Detection (`utils/device.py`)

Automatic device selection:
```python
if torch.cuda.is_available():
    device = "cuda"  # NVIDIA GPU on Windows/Linux
elif torch.backends.mps.is_available():
    device = "mps"   # Apple Silicon on macOS
else:
    device = "cpu"   # Fallback
```

Configuration default is `"auto"` which auto-detects the best device.

## Handling Imperfect Videos

The implementation includes several strategies for robustness:

### 1. Field Detection
- **Multiple Frame Sampling**: Tests 5 frames from different video parts
- **Best Candidate Selection**: Uses highest confidence detection
- **Graceful Degradation**: Returns empty field info if detection fails
- **Partial Court Support**: Line detection works even with partial visibility

### 2. Player Tracking
- **Confidence Thresholds**: Configurable detection confidence (default 0.5)
- **Court Mask Filtering**: Eliminates detections outside court area
- **Minimum Track Length**: Filters spurious detections (min 10 frames)
- **Team Assignment**: Robust to individual player position variance

### 3. Ball Tracking
- **Dual Detection Methods**: 
  - YOLO model (primary)
  - Hough Circle detection (fallback)
- **Trajectory Interpolation**: Fills gaps from missed detections
- **Low Confidence Threshold**: More permissive (default 0.3) to catch small ball
- **Field Context**: Uses court mask to reduce false positives

### 4. Error Handling
- Graceful failures with logging
- Try-except blocks around model operations
- Resource cleanup (video capture release)
- Batch processing continues on individual failures

## Configuration

See `config.example.json`:

```json
{
  "model": {
    "player_model": "yolov8n",
    "ball_model": "custom_ball_detector",
    "device": "auto",
    "batch_size": 1
  },
  "tracking": {
    "player_detection_confidence": 0.5,
    "ball_detection_confidence": 0.3,
    "interpolate_missing": true
  },
  "field_detection": {
    "line_detection_threshold": 100,
    "use_homography": true,
    "auto_calibrate": true
  }
}
```

## Usage Example

```python
from padel_analyzer import PadelAnalyzer

# Auto-detects CUDA/MPS/CPU
analyzer = PadelAnalyzer()

# Analyze a video
results = analyzer.analyze_video("match.mp4")

print(f"Device used: {results['metadata']}")
print(f"Players tracked: {len(results['player_tracks'])}")
print(f"Ball positions: {len(results['ball_tracks']['positions'])}")
```

## Future Enhancements

1. **Fine-tuning**: Train on padel-specific dataset for better accuracy
2. **Advanced Models**: 
   - SAM (Segment Anything Model) for precise court segmentation
   - CLIP for semantic understanding
   - Specialized ball detectors (TrackNet)
3. **Tracking Improvements**:
   - DeepSORT for better player re-identification
   - Kalman filtering for smoother trajectories
   - Physics-based ball motion prediction
4. **Performance**:
   - Batch processing optimization
   - Frame skipping strategies
   - Model quantization for speed

## Dependencies

Core dependencies:
- `opencv-python>=4.8.0` - Video I/O and traditional CV
- `torch>=2.0.0` - Deep learning framework
- `ultralytics>=8.0.0` - YOLOv8 implementation
- `scipy>=1.10.0` - Trajectory interpolation
- `numpy>=1.24.0` - Numerical operations

Install with:
```bash
pip install -r requirements.txt
```
