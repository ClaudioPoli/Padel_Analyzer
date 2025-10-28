# Padel Analyzer - AI Coding Agent Instructions

## Project Overview
Padel Analyzer is a computer vision pipeline for analyzing padel match videos. The system orchestrates four main components: video loading, field detection, player tracking, and ball tracking. Currently at v0.1.0 with core architecture in place, but most CV/ML implementations are placeholders marked with TODO comments.

**Goal**: Extract ball, player, and field information from padel match videos using the best CV/ML approach available (zero-shot or fine-tuned models).

## Architecture & Data Flow

**Pipeline orchestration** (`padel_analyzer/analyzer.py`):
1. `VideoLoader` loads and preprocesses video files (multiple formats)
2. `FieldDetector` identifies court boundaries/characteristics
3. `PlayerTracker` tracks players using field context
4. `BallTracker` tracks ball using field context
5. Results returned as dictionary with `field_info`, `player_tracks`, `ball_tracks`, `metadata`

**Critical design pattern**: All tracking components receive `field_info` from field detection step. This field context is essential for accurate tracking within court boundaries.

## Configuration System

Uses dataclass-based config hierarchy defined in `padel_analyzer/utils/config.py`:
- `Config` → `VideoConfig`, `TrackingConfig`, `FieldDetectionConfig`, `ModelConfig`
- Load from JSON: `Config.from_file("config.json")`
- Save to JSON: `config.to_file("config.json")`
- All components accept `config` in `__init__` and store as `self.config`

**Important**: When adding new config parameters, add them to the appropriate dataclass in `utils/config.py` and update `config.example.json`.

## Development Patterns

**Module structure convention**:
- Each module has `__init__.py` exporting main classes
- Main classes accept `config` parameter and implement primary method (`track()`, `detect()`, `load()`)
- Use logging: `logger = logging.getLogger(__name__)`
- Return structured dictionaries with documented keys

**Testing convention** (`tests/`):
- Use pytest with class-based organization: `class TestClassName:`
- Test method naming: `test_<functionality>_<scenario>`
- Tests validate error handling (e.g., `FileNotFoundError` for missing videos)
- Run with: `pytest` or `pytest --cov=padel_analyzer`

**Code quality tools** (configured in `pyproject.toml`):
- Black formatter: 100 char line length, targets Python 3.8-3.11
- Run: `black padel_analyzer/`
- Future: flake8, mypy (defined in dev extras but not yet enforced)

## Installation & Dependencies

**Core dependencies** (OpenCV-based, see `requirements.txt`):
- `opencv-python` + `opencv-contrib-python` for CV operations
- `numpy`, `Pillow` for image processing
- `imageio` + `imageio-ffmpeg` for video I/O
- NO deep learning framework by default

**Optional ML frameworks** (install via extras):
```bash
pip install -e ".[pytorch]"    # For PyTorch models
pip install -e ".[tensorflow]" # For TensorFlow models
pip install -e ".[dev]"        # For development tools
```

**Development setup**:
```bash
pip install -r requirements.txt
pip install -e .
```

**Cross-platform support**: 
- Must work on **Windows with CUDA** (GPU acceleration)
- Must work on **macOS with ARM processors** (Apple Silicon)
- Use PyTorch with MPS backend for macOS, CUDA for Windows
- Test device detection: check for `cuda`, `mps`, or fallback to `cpu`

## Implementation Priorities

**Phase 1: Video Format Support (HIGHEST PRIORITY)**
- Implement robust video loading in `video/video_loader.py`
- Support: `.mp4`, `.mov`, `.avi`, `.mkv` (see `VideoLoader.SUPPORTED_FORMATS`)
- Extract metadata: FPS, resolution, duration, frame count
- Handle different codecs and containers gracefully
- Test on both Windows and macOS

**Phase 2: CV/ML Model Selection & Integration**
- **Approach decision**: Evaluate zero-shot vs fine-tuning for padel-specific detection
  - Zero-shot: Use pre-trained models (YOLO, SAM, CLIP) directly
  - Fine-tuning: Collect padel dataset and train/fine-tune models
- **Model flexibility**: Architecture supports any detection/tracking approach
- **Cross-platform ML**: Ensure models run on CUDA (Windows) and MPS (macOS ARM)
  
**Detection targets**:
1. **Field detection** (`detection/field_detector.py`): Court lines, boundaries, key points
2. **Player tracking** (`tracking/player_tracker.py`): Detect and track 4 players, assign teams
3. **Ball tracking** (`tracking/ball_tracker.py`): Small object detection with trajectory

**Phase 3: Everything Else**
- Team assignment algorithms
- Trajectory prediction
- Analytics and statistics
- Performance optimization

**When implementing**: Follow existing structure returning dictionaries with documented keys, preserve field_info context passing pattern, add tests to `tests/`.

## Model Integration

**Model configuration pattern** (`config.model`):
- `player_model`: Model name or path (default: `"yolov8n"`)
- `ball_model`: Model name or path (default: `"custom_ball_detector"`)
- `device`: `"cpu"`, `"cuda"`, or `"mps"` (auto-detect recommended)
- `batch_size`: Processing batch size

**Device detection pattern**:
```python
import torch
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"  # Apple Silicon
else:
    device = "cpu"
```

**Model approach considerations**:
- Evaluate zero-shot models (SAM, YOLO-World, CLIP) before committing to fine-tuning
- Document model selection rationale in code comments
- Keep model loading flexible to swap approaches easily

## Examples & Usage

Reference `examples/basic_usage.py` for canonical usage patterns:
- Example 1: Default initialization
- Example 2: Custom config via Python API
- Example 3: Config save/load from JSON
- Example 4: Batch processing with error handling

**Error handling pattern**: Batch processing catches exceptions and returns `{"error": str(e), "video_path": path}` for failed videos rather than crashing.

## Git Workflow

**Branching strategy**:
- `main` branch: Stable, production-ready code only
- Feature branches: `feature/<feature-name>` for new development
- Work on feature branches, then PR → review → merge to `main`
- Never commit directly to `main`

**Typical workflow**:
```bash
git checkout -b feature/video-loader-implementation
# Make changes, commit
git push origin feature/video-loader-implementation
# Create PR on GitHub, review, merge to main
```

## Common Commands

```bash
# Testing
pytest                          # Run all tests
pytest --cov=padel_analyzer    # With coverage
pytest tests/test_analyzer.py  # Specific file

# Formatting
black padel_analyzer/          # Format code

# Install modes
pip install -e .                    # Editable install
pip install -e ".[dev,pytorch]"     # With extras
```

## Critical Notes

- **Python 3.8+ required**: Code uses typing features (e.g., `Optional`, `Dict`)
- **Supported video formats**: `.mp4`, `.mov`, `.avi`, `.mkv` (defined in `VideoLoader.SUPPORTED_FORMATS`)
- **Public API**: Only `PadelAnalyzer` exported from root `__init__.py`
- **Cross-platform requirement**: Must work on Windows (CUDA) and macOS (ARM/MPS)
- **Configuration philosophy**: Keep high-level; avoid exposing too many low-level parameters
- **Current focus**: Video format support first, then model selection/integration, then advanced features
