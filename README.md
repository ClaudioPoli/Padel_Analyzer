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

### Planned Features

#### Short-term
- [ ] Implement video loading with OpenCV
- [ ] Integrate object detection models (YOLO/Faster R-CNN)
- [ ] Implement basic player tracking
- [ ] Implement ball detection and tracking
- [ ] Add field line detection using Hough transform

#### Medium-term
- [ ] Fine-tune models for padel-specific detection
- [ ] Implement player re-identification
- [ ] Add trajectory prediction
- [ ] Team assignment based on position/appearance
- [ ] Performance metrics and analytics

#### Long-term
- [ ] Real-time video analysis
- [ ] Action recognition (serves, volleys, smashes)
- [ ] Game statistics extraction
- [ ] Multi-camera support
- [ ] Web interface for visualization
- [ ] Integration with LLMs for match commentary

## Model Selection

The project is designed to support multiple model backends:

### Player Tracking
- **YOLO (v5/v8)**: Fast and accurate object detection
- **Faster R-CNN**: High accuracy for player detection
- **DeepSORT/ByteTrack**: Robust multi-object tracking

### Ball Tracking
- **Custom CNN**: Trained specifically for ball detection
- **TrackNetV2**: Specialized for ball tracking in sports
- **Temporal models**: Using frame sequences for better accuracy

### Field Detection
- **Semantic Segmentation**: For court area identification
- **Hough Transform**: For line detection
- **Template Matching**: For known court layouts

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
