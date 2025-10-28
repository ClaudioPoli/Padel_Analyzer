# Padel Analyzer

AI-based tool to analyze padel matches from video input. This tool uses computer vision and machine learning models to track players, detect ball positions, and automatically identify the padel court.

## Features

- 🎥 **Video Processing**: Support for multiple video formats (MP4, MOV, AVI, MKV, etc.)
- 🏃 **Player Tracking**: Detect and track player movements throughout the match
- 🎾 **Ball Tracking**: Track ball position and trajectory with high precision
- 🏟️ **Field Detection**: Automatically identify and map the padel court
- ⚙️ **Configurable**: Flexible configuration system for customizing analysis parameters
- 🔄 **Batch Processing**: Analyze multiple videos in one go

## Project Structure

```
Padel_Analyzer/
├── padel_analyzer/          # Main package directory
│   ├── __init__.py          # Package initialization
│   ├── analyzer.py          # Main analyzer orchestration
│   ├── video/               # Video processing modules
│   │   ├── __init__.py
│   │   └── video_loader.py  # Video loading and preprocessing
│   ├── tracking/            # Tracking modules
│   │   ├── __init__.py
│   │   ├── player_tracker.py  # Player detection and tracking
│   │   └── ball_tracker.py    # Ball detection and tracking
│   ├── detection/           # Detection modules
│   │   ├── __init__.py
│   │   └── field_detector.py  # Field/court detection
│   └── utils/               # Utility modules
│       ├── __init__.py
│       └── config.py        # Configuration management
├── examples/                # Example usage scripts
│   └── basic_usage.py
├── tests/                   # Unit tests
│   ├── __init__.py
│   ├── test_analyzer.py
│   ├── test_config.py
│   └── test_video_loader.py
├── config.example.json      # Example configuration file
├── requirements.txt         # Python dependencies
├── pyproject.toml          # Project configuration
├── setup.py                # Package setup script
└── README.md               # This file
```

## Installation

### From Source

1. Clone the repository:
```bash
git clone https://github.com/ClaudioPoli/Padel_Analyzer.git
cd Padel_Analyzer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install the package:
```bash
pip install -e .
```

### Optional Dependencies

For deep learning model support, install optional dependencies:

```bash
# For PyTorch-based models
pip install -e ".[pytorch]"

# For TensorFlow-based models
pip install -e ".[tensorflow]"

# For development
pip install -e ".[dev]"
```

## Quick Start

### Basic Usage

```python
from padel_analyzer import PadelAnalyzer

# Initialize analyzer with default configuration
analyzer = PadelAnalyzer()

# Analyze a video
results = analyzer.analyze_video("path/to/match.mp4")

# Access results
print(f"Field detected: {results['field_info']}")
print(f"Players tracked: {len(results['player_tracks'])}")
print(f"Ball positions: {len(results['ball_tracks']['positions'])}")
```

### Running the Demo

To test the implementation with a synthetic video:

```bash
python examples/demo_processing.py
```

Or with your own video:

```bash
python examples/demo_processing.py path/to/your/video.mp4
```

### Custom Configuration

```python
from padel_analyzer import PadelAnalyzer
from padel_analyzer.utils.config import Config

# Create custom configuration
config = Config()
config.model.device = "cuda"  # Use GPU
config.tracking.player_detection_confidence = 0.7
config.tracking.ball_detection_confidence = 0.5

# Initialize with custom config
analyzer = PadelAnalyzer(config)
results = analyzer.analyze_video("match.mp4")
```

### Batch Processing

```python
from padel_analyzer import PadelAnalyzer

analyzer = PadelAnalyzer()

# Process multiple videos
video_paths = ["match1.mp4", "match2.mp4", "match3.mov"]
results = analyzer.analyze_video_batch(video_paths)

for i, result in enumerate(results):
    if "error" not in result:
        print(f"Video {i+1} analyzed successfully")
```

## Configuration

Configuration can be managed through:

1. **Python API**:
```python
from padel_analyzer.utils.config import Config

config = Config()
config.model.device = "cuda"
config.tracking.player_detection_confidence = 0.8
```

2. **JSON Configuration File**:
```python
config = Config.from_file("my_config.json")
```

See `config.example.json` for a complete configuration example.

### Configuration Options

- **Video Settings**: Format support, FPS, resolution
- **Tracking Settings**: Detection confidence thresholds, interpolation
- **Field Detection**: Line detection, corner detection, homography
- **Model Settings**: Model selection, device (CPU/GPU), batch size

## Development Roadmap

### Current Status (v0.1.0)
✅ Base project structure  
✅ Core module architecture  
✅ Configuration system  
✅ Basic testing framework  
✅ **Video loading with OpenCV (MP4, MOV, AVI, MKV)**  
✅ **Field detection using Hough Transform**  
✅ **Player tracking with YOLOv8**  
✅ **Ball tracking with YOLO + traditional CV fallback**  
✅ **Cross-platform device detection (CUDA/MPS/CPU)**  

### Planned Features

#### Short-term
- [ ] Fine-tune models for padel-specific detection
- [ ] Improve player re-identification
- [ ] Enhanced trajectory prediction
- [ ] Better team assignment algorithms
- [ ] Performance optimization and caching

#### Medium-term
- [ ] Action recognition (serves, volleys, smashes)
- [ ] Game statistics extraction
- [ ] Point tracking and scoring
- [ ] Player heatmaps
- [ ] Shot type classification

#### Long-term
- [ ] Real-time video analysis
- [ ] Multi-camera support
- [ ] Web interface for visualization
- [ ] Integration with LLMs for match commentary
- [ ] Advanced analytics dashboard

## Model Selection

The project currently uses **YOLOv8** as the primary detection framework:

### Player Tracking
- **YOLOv8 (Ultralytics)**: Fast and accurate person detection with built-in tracking
- **Pre-trained on COCO dataset**: Works zero-shot without fine-tuning
- **Cross-platform**: Supports CUDA (Windows/Linux), MPS (Apple Silicon), and CPU

### Ball Tracking
- **Primary**: YOLOv8 sports ball detection (COCO class 32)
- **Fallback**: Hough Circle Transform for traditional CV-based detection
- **Trajectory interpolation**: Scipy-based smoothing for missed frames

### Field Detection
- **Hough Line Transform**: For court line detection
- **Canny Edge Detection**: Preprocessing for line detection
- **Homography Estimation**: For perspective correction and top-down view
- **Future**: Semantic segmentation models for more robust court detection

### Device Support
Automatic detection of best available device:
- **CUDA**: NVIDIA GPUs on Windows/Linux
- **MPS**: Apple Silicon GPUs on macOS
- **CPU**: Fallback for systems without GPU

See `IMPLEMENTATION.md` for detailed technical documentation.

## Testing

Run tests using pytest:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=padel_analyzer

# Run specific test file
pytest tests/test_analyzer.py
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License.

## Acknowledgments

- Inspired by sports analytics and computer vision research
- Built on top of OpenCV, PyTorch/TensorFlow ecosystems
- Thanks to the open-source community for various CV tools

## Contact

For questions or suggestions, please open an issue on GitHub.
