# 🎭 Advanced Emotion Detection System

A state-of-the-art emotion detection system using deep learning, built with TensorFlow/Keras and deployed with modern web technologies. This project features multiple deployment options, real-time video processing, and various model optimization techniques.

## 🌟 Features

### Core Capabilities
- **7 Emotion Classes**: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise
- **Multiple Model Architectures**: EfficientNet, ResNet50V2, MobileNetV2
- **Advanced Training Techniques**: 
  - Two-phase training with fine-tuning
  - Cosine annealing learning rate schedule
  - Class weight balancing
  - Advanced data augmentation

### Deployment Options
- **Streamlit Web App**: Interactive UI with webcam support
- **FastAPI REST API**: High-performance API with WebSocket support
- **Command-Line Tool**: Batch processing capabilities
- **Docker Support**: Containerized deployment with docker-compose
- **Real-time Video Processing**: Process video streams with emotion tracking

### Optimization Features
- **Model Quantization**: Dynamic, INT8, and Float16 quantization
- **TensorFlow Lite**: Mobile and edge device deployment
- **Model Pruning**: Reduce model size while maintaining accuracy
- **Edge TPU Support**: Optimized for Google Coral devices
- **Ensemble Methods**: Combine multiple models for better accuracy

## 📋 Requirements

- Python 3.8 or higher
- TensorFlow 2.10+
- CUDA-capable GPU (optional but recommended)
- 8GB+ RAM
- FER2013 dataset

## 🚀 Quick Start

### 1. Clone and Setup

```bash
# Clone the repository
git clone <repository-url>
cd emotion_detection_project

# Run setup script
chmod +x setup.sh
./setup.sh

# Or manually:
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Train the Model

Open and run the Jupyter notebook:
```bash
jupyter notebook FER2013_Improved.ipynb
```

The notebook includes:
- Comprehensive data exploration
- Advanced model architecture (EfficientNet)
- Two-phase training with fine-tuning
- Detailed evaluation metrics

### 3. Run the Application

#### Option A: Streamlit Web App
```bash
streamlit run emotion_detection_app.py
```
- Upload images or use webcam
- Real-time emotion detection
- Probability visualization

#### Option B: FastAPI REST API
```bash
python api.py
```
- Access API docs at http://localhost:8000/docs
- WebSocket support for real-time processing
- Batch prediction endpoints

#### Option C: Command-Line Tool
```bash
# Single image
python predict_emotion.py path/to/image.jpg

# Batch processing
python predict_emotion.py --batch path/to/images/

# Video processing
python video_emotion_detection.py --input video.mp4
```

## 🐳 Docker Deployment

```bash
# Build and run with docker-compose
docker-compose up --build

# Services will be available at:
# - API: http://localhost:8000
# - Web App: http://localhost:8501
# - Redis: localhost:6379
```

## 📊 Model Performance

### Accuracy Metrics
- **Baseline Model**: ~25% accuracy
- **Improved Model**: 65-75% accuracy
- **With Ensemble**: Up to 80% accuracy

### Optimization Results
| Model Type | Size | Inference Time | Accuracy Loss |
|------------|------|----------------|---------------|
| Original | 90MB | 50ms | - |
| TFLite Dynamic | 23MB | 15ms | <1% |
| TFLite INT8 | 23MB | 10ms | 2-3% |
| Pruned (50%) | 45MB | 45ms | 1-2% |

## 🛠️ Advanced Features

### 1. Real-time Video Processing
```bash
python video_emotion_detection.py
```
- Webcam support
- Video file processing
- Emotion timeline visualization
- Face tracking

### 2. Model Optimization
```bash
# Run all optimizations
python optimize_model.py models/best_model.keras --all

# Specific optimizations
python optimize_model.py models/best_model.keras --int8 --benchmark
```

### 3. Ensemble Models
```python
from ensemble_model import EnsembleEmotionModel

# Create ensemble
ensemble = EnsembleEmotionModel(ensemble_method='weighted')
ensemble.add_model('models/efficientnet_model.keras', weight=0.4)
ensemble.add_model('models/resnet_model.keras', weight=0.3)
ensemble.add_model('models/mobilenet_model.keras', weight=0.3)

# Build and save
ensemble_model = ensemble.build_ensemble()
ensemble.save_ensemble('models/ensemble_final.keras')
```

### 4. Performance Benchmarking
```bash
python benchmark.py models/best_model.keras
```
- Inference speed testing
- Memory usage profiling
- Optimization comparison
- Detailed reports

## 📁 Project Structure

```
emotion_detection_project/
├── FER2013_Improved.ipynb      # Main training notebook
├── emotion_detection_app.py    # Streamlit web application
├── api.py                      # FastAPI REST API
├── predict_emotion.py          # CLI tool
├── video_emotion_detection.py  # Video processing
├── ensemble_model.py           # Ensemble methods
├── optimize_model.py           # Model optimization
├── benchmark.py                # Performance testing
├── config.yaml                 # Configuration file
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Docker container
├── docker-compose.yml          # Multi-container setup
├── setup.sh                    # Setup script
├── models/                     # Saved models
├── logs/                       # Training logs
└── tests/                      # Unit tests
```

## 🔧 Configuration

Edit `config.yaml` to customize:
- Model architecture and hyperparameters
- Training settings
- Data augmentation parameters
- API configuration
- Performance settings

## 📈 Monitoring and Logging

- TensorBoard logs in `logs/` directory
- API logs with configurable levels
- Real-time performance metrics
- Redis caching for improved performance

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_model.py -v

# Run with coverage
pytest --cov=. tests/
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📝 API Documentation

### REST Endpoints

- `POST /predict` - Single image prediction
- `POST /predict_batch` - Batch prediction
- `POST /analyze_video` - Video analysis (async)
- `GET /emotions` - List available emotions
- `GET /model_info` - Model information
- `WebSocket /ws` - Real-time predictions

### Example Usage

```python
import requests

# Single prediction
with open('face.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/predict',
        files={'file': f}
    )
    print(response.json())
```

## ⚠️ Limitations

- Trained on FER2013 dataset (Western-centric)
- Best with frontal face images
- Single face detection (largest face in multi-face images)
- Emotions are simplified categories

## 🔮 Future Improvements

- [ ] Multi-face emotion detection
- [ ] Emotion intensity measurement
- [ ] Temporal emotion analysis
- [ ] Mobile app (React Native/Flutter)
- [ ] Custom emotion categories
- [ ] Real-time emotion dashboard
- [ ] Integration with video conferencing tools

## 📜 License

This project is for educational purposes. Please respect the FER2013 dataset license terms.

## 🙏 Acknowledgments

- FER2013 dataset creators
- TensorFlow/Keras team
- EfficientNet authors
- Open source community

---

For questions or issues, please open a GitHub issue or contact the maintainers.