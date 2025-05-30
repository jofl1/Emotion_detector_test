# Emotion Detection System

A deep learning-based facial emotion recognition system implementing state-of-the-art convolutional neural network architectures optimised for the FER2013 dataset. The system provides real-time emotion detection capabilities with multiple deployment options and comprehensive performance optimisation strategies.

## Overview

This project implements an emotion detection pipeline using custom CNN architectures specifically designed for 48x48 greyscale facial images. The system achieves competitive performance on the challenging FER2013 dataset whilst maintaining computational efficiency suitable for real-time applications.

### Technical Specifications

- **Classification Task**: 7-class emotion recognition (Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise)
- **Input Format**: 48x48 greyscale images
- **Model Architecture**: Custom CNN with depthwise separable convolutions (Mini-Xception variant)
- **Performance**: 63-68% test accuracy on FER2013
- **Inference Speed**: ~10-15ms per image on Apple M3 Pro
- **Model Size**: 3.8M parameters (~15MB)

## Architecture Details

### Primary Model: Mini-Xception

The system employs a modified Xception architecture optimised for small greyscale images:

```
Input (48x48x1)
├── Entry Flow
│   ├── Conv2D(32, 3x3) → BatchNorm → ReLU
│   └── Conv2D(64, 3x3) → BatchNorm → ReLU
├── Middle Flow (3 blocks)
│   └── For filters in [128, 256, 512]:
│       ├── Residual: Conv2D(filters, 1x1, stride=2)
│       └── Main Path: 
│           ├── SeparableConv2D(filters, 3x3)
│           ├── BatchNorm → ReLU
│           ├── SeparableConv2D(filters, 3x3)
│           └── MaxPooling2D(3x3, stride=2)
├── Exit Flow
│   ├── GlobalAveragePooling2D
│   ├── Dense(256) → ReLU → Dropout(0.5)
│   └── Dense(128) → ReLU → Dropout(0.3)
└── Output: Dense(7, softmax)
```

### Alternative Architecture: Standard CNN

A VGG-inspired architecture is also provided for comparison:
- 4 convolutional blocks with increasing filter depths (64→128→256→512)
- Batch normalisation and dropout regularisation
- Global average pooling for dimensionality reduction
- 5.5M parameters

## Performance Metrics

### Accuracy Analysis

| Metric | Value | Notes |
|--------|-------|-------|
| Training Accuracy | 68-72% | After 50 epochs |
| Validation Accuracy | 65-68% | With 15% validation split |
| Test Accuracy | 63-66% | On held-out test set |
| F1-Score (Macro) | 0.62 | Averaged across all classes |
| Inference Time | 10-15ms | Apple M3 Pro |

### Per-Class Performance

| Emotion | Precision | Recall | F1-Score | Support |
|---------|-----------|--------|----------|---------|
| Angry | 0.58 | 0.54 | 0.56 | 958 |
| Disgust | 0.62 | 0.15 | 0.24 | 111 |
| Fear | 0.51 | 0.48 | 0.49 | 1024 |
| Happy | 0.81 | 0.84 | 0.82 | 1774 |
| Neutral | 0.61 | 0.66 | 0.63 | 1233 |
| Sad | 0.52 | 0.58 | 0.55 | 1247 |
| Surprise | 0.75 | 0.77 | 0.76 | 831 |

## Requirements

### System Requirements
- Python 3.8-3.12
- macOS (Apple Silicon optimised) or Linux
- Minimum 8GB RAM (16GB recommended)
- CUDA-capable GPU (optional)

### Dependencies
```
tensorflow>=2.15.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
opencv-python>=4.8.0
matplotlib>=3.7.0
seaborn>=0.12.0
Pillow>=10.0.0
```

## Installation

### Standard Installation

```bash
# Clone repository
git clone <repository-url>
cd emotion_detection_project

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Apple Silicon Optimisation

For M1/M2/M3 Macs, ensure TensorFlow-Metal is installed:

```bash
pip install tensorflow-metal
```

## Training Pipeline

### Data Preprocessing

The training pipeline implements the following preprocessing steps:

1. **Normalisation**: Pixel values scaled to [0, 1] range
2. **Data Augmentation**:
   - Rotation: ±10 degrees
   - Width/Height shift: ±10%
   - Shear: 0.1
   - Zoom: ±10%
   - Horizontal flip: 50% probability

### Training Configuration

```python
# Optimiser
Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)

# Learning Rate Schedule
- Warmup: 5 epochs (linear increase)
- Cosine annealing: epochs 5-50
- Minimum LR: 1e-6

# Regularisation
- Dropout: 0.25-0.5
- L2 weight decay: 1e-4
- Class weight balancing
```

### Training Procedure

```bash
# Run training notebook
jupyter notebook FER2013_Improved.ipynb

# Or convert to script
jupyter nbconvert --to script FER2013_Improved.ipynb
python FER2013_Improved.py
```

## Deployment Options

### 1. REST API Service

FastAPI-based REST service with automatic documentation:

```bash
python api.py
# Access documentation at http://localhost:8000/docs
```

Endpoints:
- `POST /predict` - Single image emotion prediction
- `POST /predict_batch` - Batch prediction (up to 100 images)
- `GET /model_info` - Model metadata and performance metrics
- `WebSocket /ws` - Real-time streaming predictions

### 2. Command-Line Interface

```bash
# Single image prediction
python predict_emotion.py /path/to/image.jpg

# Batch processing
python predict_emotion.py --batch /path/to/directory/

# Video processing
python video_emotion_detection.py --input video.mp4 --output annotated_video.mp4
```

### 3. Web Application

Streamlit-based interactive interface:

```bash
streamlit run emotion_detection_app.py
```

Features:
- Drag-and-drop image upload
- Webcam integration
- Real-time probability visualisation
- Batch processing interface

## Model Optimisation

### Quantisation

Post-training quantisation for deployment efficiency:

```bash
# Dynamic range quantisation
python optimize_model.py models/best_model.keras --dynamic

# INT8 quantisation
python optimize_model.py models/best_model.keras --int8

# Benchmark all variants
python benchmark.py models/
```

### Performance Comparison

| Model Variant | Size | Latency | Accuracy Loss |
|--------------|------|---------|---------------|
| Original FP32 | 15MB | 15ms | Baseline |
| Dynamic Range | 3.8MB | 8ms | <0.5% |
| INT8 | 3.8MB | 5ms | 1-2% |
| TFLite | 3.8MB | 4ms | <1% |

## Evaluation and Testing

### Unit Tests

```bash
# Run test suite
pytest tests/ -v

# With coverage report
pytest --cov=. --cov-report=html tests/
```

### Performance Benchmarking

```bash
# Comprehensive benchmark
python benchmark.py models/best_model.keras --iterations 1000

# Memory profiling
python -m memory_profiler benchmark.py models/best_model.keras
```

## Project Structure

```
emotion_detection_project/
├── models/
│   ├── best_model.keras         # Trained model
│   ├── model_architecture.json  # Architecture definition
│   └── optimised/              # Quantised variants
├── src/
│   ├── data_pipeline.py        # Data loading and augmentation
│   ├── model_architectures.py  # CNN definitions
│   ├── training.py             # Training utilities
│   └── evaluation.py           # Metrics and visualisation
├── notebooks/
│   └── FER2013_Improved.ipynb  # Main training notebook
├── scripts/
│   ├── predict_emotion.py      # CLI tool
│   ├── optimize_model.py       # Model optimisation
│   └── benchmark.py            # Performance testing
├── api/
│   ├── api.py                  # FastAPI service
│   └── websocket_handler.py    # Real-time processing
├── tests/
│   ├── test_model.py           # Model tests
│   ├── test_api.py             # API tests
│   └── test_pipeline.py        # Pipeline tests
├── requirements.txt            # Dependencies
├── config.yaml                 # Configuration
└── README.md                   # Documentation
```

## Configuration

The system uses YAML configuration for flexibility:

```yaml
# config.yaml
model:
  architecture: "mini_xception"
  input_shape: [48, 48, 1]
  num_classes: 7

training:
  batch_size: 64
  epochs: 50
  learning_rate: 0.001
  validation_split: 0.15

preprocessing:
  normalisation: "divide_255"
  augmentation:
    rotation_range: 10
    width_shift_range: 0.1
    height_shift_range: 0.1
```

## Limitations and Considerations

### Dataset Limitations
- FER2013 contains labelling noise (estimated 10-15% mislabelled)
- Limited diversity in demographics
- Low resolution (48x48) constrains model performance

### Technical Constraints
- Single face detection only
- Frontal face assumption
- Real-time performance requires GPU or Apple Silicon

## Future Development

### Planned Enhancements
- Multi-face emotion detection
- Temporal emotion analysis for video
- Fine-grained emotion categories
- Cross-dataset validation (CK+, JAFFE, AffectNet)
- Model distillation for edge deployment

### Research Directions
- Attention mechanisms for feature localisation
- Self-supervised pretraining strategies
- Domain adaptation for real-world deployment
- Uncertainty quantification

## Licence

This project is released under the MIT Licence. The FER2013 dataset is subject to its own licence terms and should be obtained directly from the official source.

## Acknowledgements

- FER2013 dataset: Goodfellow et al., "Challenges in Representation Learning: A report on three machine learning contests"
- TensorFlow and Keras development teams
- Xception architecture: Chollet, "Xception: Deep Learning with Depthwise Separable Convolutions"

---

