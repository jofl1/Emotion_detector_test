# Emotion Detection System

Professional real-time emotion detection using deep learning and Mac camera.

## Features

- **High Accuracy**: Advanced CNN architecture optimized for emotion detection
- **Real-time Performance**: Smooth 30+ FPS on Mac with M-series chips
- **7 Emotions**: Detects Angry, Disgust, Fear, Happy, Neutral, Sad, and Surprise
- **Mac Optimized**: Built specifically for macOS camera integration
- **Clean Interface**: Professional visualization with emotion probability bars

## Quick Start

### 1. Install Requirements

```bash
pip install -r requirements_minimal.txt
```

### 2. Run Emotion Detection

```bash
python emotion_detector.py
```

This will start the camera and begin real-time emotion detection.

### Controls

- **Q**: Quit the application
- **S**: Save screenshot
- **R**: Reset emotion smoothing

## Training Your Own Model

If you want to train a new model with the FER2013 dataset:

1. Download the FER2013 dataset
2. Update the `BASE_DIR` in `src/train_improved_model.py` to point to your dataset
3. Run training:

```bash
python emotion_detector.py --train
```

## Project Structure

```
emotion_detector_test/
├── emotion_detector.py        # Main launcher script
├── src/                       # Core source code
│   ├── main.py               # Application entry point
│   ├── emotion_detector.py   # Core detection logic
│   ├── train_improved_model.py  # Model training
│   ├── predict_emotion.py    # CLI prediction tool
│   └── inference_engine.py   # Inference utilities
├── models/                    # Trained models
│   └── best_emotion_model.keras
├── configs/                   # Configuration files
│   ├── config.yaml          # Basic configuration
│   └── config_pro.yaml      # Advanced configuration
├── examples/                  # Examples and tutorials
│   └── FER2013_Improved.ipynb
├── scripts/                   # Utility scripts
│   ├── setup.sh             # Setup script
│   └── pro/                 # Advanced pro version
│       ├── main_pro.py      # Pro system entry
│       ├── emotion_detector_pro.py  # Pro detector
│       └── train_advanced.py  # Advanced training
├── docs/                      # Documentation
│   └── README_PRO.md         # Pro version docs
├── tests/                     # Test files
├── requirements_minimal.txt   # Basic dependencies
├── requirements.txt          # Full dependencies
└── requirements_pro.txt      # Pro version dependencies
```

## Model Architecture

The system uses a deep CNN architecture with:
- Multiple convolutional blocks with batch normalization
- Dropout layers for regularization  
- Global average pooling for efficiency
- Optimized for 48x48 grayscale facial images

Expected accuracy: 65-70% on FER2013 test set

## Requirements

- Python 3.8+
- macOS (tested on M-series Macs)
- Camera access permissions

## Troubleshooting

**Camera not working?**
- Grant camera permissions to Terminal in System Preferences
- Make sure no other application is using the camera

**Model not found?**
- Train a model first using `python main.py --train`
- Or download a pre-trained model

**Low FPS?**
- Close other applications
- Reduce camera resolution in `emotion_detector.py`

## License

MIT License - feel free to use for your projects!

