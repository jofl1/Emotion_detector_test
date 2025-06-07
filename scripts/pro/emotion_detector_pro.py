#!/usr/bin/env python3
"""
Professional Emotion Detection System
High-performance implementation with advanced optimization techniques
"""

import cv2
import numpy as np
import tensorflow as tf
from typing import Tuple, List, Dict, Optional, Callable
from dataclasses import dataclass
from collections import deque
import threading
import queue
import time
from abc import ABC, abstractmethod
import logging
from contextlib import contextmanager
from functools import lru_cache
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enable GPU memory growth for TensorFlow
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        logger.error(f"GPU configuration error: {e}")


@dataclass(frozen=True)
class EmotionLabel:
    """Immutable emotion label with metadata"""
    id: int
    name: str
    color: Tuple[int, int, int]  # BGR format
    
    def __hash__(self):
        return hash((self.id, self.name))


class EmotionLabels:
    """Emotion labels registry with O(1) lookup"""
    ANGRY = EmotionLabel(0, "Angry", (0, 0, 255))
    DISGUST = EmotionLabel(1, "Disgust", (128, 0, 128))
    FEAR = EmotionLabel(2, "Fear", (255, 128, 0))
    HAPPY = EmotionLabel(3, "Happy", (0, 255, 255))
    NEUTRAL = EmotionLabel(4, "Neutral", (128, 128, 128))
    SAD = EmotionLabel(5, "Sad", (255, 0, 0))
    SURPRISE = EmotionLabel(6, "Surprise", (0, 255, 0))
    
    _ALL = [ANGRY, DISGUST, FEAR, HAPPY, NEUTRAL, SAD, SURPRISE]
    _BY_ID = {label.id: label for label in _ALL}
    _BY_NAME = {label.name: label for label in _ALL}
    
    @classmethod
    def get_by_id(cls, id: int) -> EmotionLabel:
        return cls._BY_ID[id]
    
    @classmethod
    def get_by_name(cls, name: str) -> EmotionLabel:
        return cls._BY_NAME[name]
    
    @classmethod
    def all(cls) -> List[EmotionLabel]:
        return cls._ALL.copy()


class FaceDetector(ABC):
    """Abstract base class for face detection strategies"""
    
    @abstractmethod
    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        pass


class HaarCascadeFaceDetector(FaceDetector):
    """OpenCV Haar Cascade face detector with caching"""
    
    def __init__(self):
        self._cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self._cache = {}
        self._cache_size = 10
        
    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        # Generate frame hash for caching
        frame_hash = hash(frame.tobytes())
        
        if frame_hash in self._cache:
            return self._cache[frame_hash]
        
        # Convert to grayscale for detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces with optimized parameters
        faces = self._cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(48, 48),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # Convert to list of tuples
        faces_list = [tuple(face) for face in faces]
        
        # Update cache with LRU eviction
        if len(self._cache) >= self._cache_size:
            self._cache.pop(next(iter(self._cache)))
        self._cache[frame_hash] = faces_list
        
        return faces_list


class DNNFaceDetector(FaceDetector):
    """DNN-based face detector for higher accuracy"""
    
    def __init__(self):
        # Load pre-trained model
        self._net = cv2.dnn.readNetFromCaffe(
            'deploy.prototxt',
            'res10_300x300_ssd_iter_140000.caffemodel'
        )
        self._confidence_threshold = 0.5
        
    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        h, w = frame.shape[:2]
        
        # Prepare input blob
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)),
            1.0,
            (300, 300),
            (104.0, 177.0, 123.0)
        )
        
        # Run inference
        self._net.setInput(blob)
        detections = self._net.forward()
        
        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self._confidence_threshold:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x, y, x2, y2 = box.astype(int)
                faces.append((x, y, x2 - x, y2 - y))
                
        return faces


class EmotionPredictor:
    """High-performance emotion prediction with TensorFlow optimization"""
    
    def __init__(self, model_path: str):
        self._model = self._load_optimized_model(model_path)
        self._input_shape = (48, 48)
        self._preprocessor = self._create_preprocessor()
        
    def _load_optimized_model(self, model_path: str) -> tf.keras.Model:
        """Load model with optimization"""
        try:
            # Load model
            model = tf.keras.models.load_model(model_path)
            
            # Optimize for inference
            model = tf.function(lambda x: model(x))
            
            logger.info(f"Model loaded from {model_path}")
            return model
            
        except Exception as e:
            # Try alternative paths
            alt_paths = [
                'best_emotion_model.keras',
                'emotion_model_final.keras',
                'best_emotion_model_improved.keras'
            ]
            
            for path in alt_paths:
                try:
                    model = tf.keras.models.load_model(path)
                    model = tf.function(lambda x: model(x))
                    logger.info(f"Model loaded from {path}")
                    return model
                except:
                    continue
                    
            raise RuntimeError(f"Failed to load model: {e}")
    
    def _create_preprocessor(self) -> Callable:
        """Create optimized preprocessing function"""
        @tf.function
        def preprocess(image: tf.Tensor) -> tf.Tensor:
            # Resize
            image = tf.image.resize(image, self._input_shape)
            
            # Normalize to [0, 1]
            image = tf.cast(image, tf.float32) / 255.0
            
            # Add batch dimension
            image = tf.expand_dims(image, 0)
            
            return image
            
        return preprocess
    
    @lru_cache(maxsize=128)
    def _preprocess_face_cached(self, face_bytes: bytes) -> np.ndarray:
        """Cached preprocessing for repeated faces"""
        face = np.frombuffer(face_bytes, dtype=np.uint8)
        face = face.reshape((48, 48))
        return self._preprocessor(face).numpy()
    
    def predict(self, face_img: np.ndarray) -> Tuple[EmotionLabel, float, np.ndarray]:
        """Predict emotion with optimized pipeline"""
        # Convert to grayscale if needed
        if len(face_img.shape) == 3:
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        
        # Resize for preprocessing
        face_resized = cv2.resize(face_img, self._input_shape)
        
        # Use cached preprocessing if possible
        face_bytes = face_resized.tobytes()
        processed = self._preprocess_face_cached(face_bytes)
        
        # Run prediction
        predictions = self._model(processed).numpy()[0]
        
        # Get top emotion
        emotion_id = np.argmax(predictions)
        confidence = predictions[emotion_id]
        emotion = EmotionLabels.get_by_id(emotion_id)
        
        return emotion, confidence, predictions


class KalmanFilter:
    """Kalman filter for smoothing emotion predictions"""
    
    def __init__(self, num_states: int):
        self.num_states = num_states
        
        # State transition matrix
        self.F = np.eye(num_states)
        
        # Measurement matrix
        self.H = np.eye(num_states)
        
        # Process noise covariance
        self.Q = np.eye(num_states) * 0.01
        
        # Measurement noise covariance
        self.R = np.eye(num_states) * 0.1
        
        # Initial state estimate
        self.x = np.ones(num_states) / num_states
        
        # Initial covariance estimate
        self.P = np.eye(num_states)
        
    def update(self, measurement: np.ndarray) -> np.ndarray:
        """Update filter with new measurement"""
        # Prediction step
        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q
        
        # Update step
        y = measurement - self.H @ x_pred
        S = self.H @ P_pred @ self.H.T + self.R
        K = P_pred @ self.H.T @ np.linalg.inv(S)
        
        self.x = x_pred + K @ y
        self.P = (np.eye(self.num_states) - K @ self.H) @ P_pred
        
        # Normalize to ensure probabilities sum to 1
        self.x = np.clip(self.x, 0, 1)
        self.x /= np.sum(self.x)
        
        return self.x


class FrameProcessor:
    """Multi-threaded frame processing pipeline"""
    
    def __init__(self, face_detector: FaceDetector, emotion_predictor: EmotionPredictor):
        self.face_detector = face_detector
        self.emotion_predictor = emotion_predictor
        self.kalman_filters = {}  # Per-face Kalman filters
        
        # Thread pool for parallel processing
        self.input_queue = queue.Queue(maxsize=10)
        self.output_queue = queue.Queue(maxsize=10)
        self.processing_thread = threading.Thread(target=self._process_frames, daemon=True)
        self.processing_thread.start()
        
    def _process_frames(self):
        """Background thread for frame processing"""
        while True:
            try:
                frame = self.input_queue.get(timeout=1)
                if frame is None:
                    break
                    
                # Detect faces
                faces = self.face_detector.detect_faces(frame)
                
                results = []
                for i, (x, y, w, h) in enumerate(faces):
                    # Extract face region
                    face_roi = frame[y:y+h, x:x+w]
                    
                    # Predict emotion
                    emotion, confidence, predictions = self.emotion_predictor.predict(face_roi)
                    
                    # Apply Kalman filtering
                    if i not in self.kalman_filters:
                        self.kalman_filters[i] = KalmanFilter(len(EmotionLabels.all()))
                    
                    smoothed_predictions = self.kalman_filters[i].update(predictions)
                    
                    # Update emotion based on smoothed predictions
                    smoothed_id = np.argmax(smoothed_predictions)
                    smoothed_emotion = EmotionLabels.get_by_id(smoothed_id)
                    smoothed_confidence = smoothed_predictions[smoothed_id]
                    
                    results.append({
                        'bbox': (x, y, w, h),
                        'emotion': smoothed_emotion,
                        'confidence': smoothed_confidence,
                        'predictions': smoothed_predictions,
                        'raw_predictions': predictions
                    })
                
                self.output_queue.put((frame, results))
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Frame processing error: {e}")
    
    def process(self, frame: np.ndarray) -> Optional[Tuple[np.ndarray, List[Dict]]]:
        """Process frame asynchronously"""
        try:
            # Add frame to processing queue
            self.input_queue.put(frame, block=False)
            
            # Get processed result if available
            if not self.output_queue.empty():
                return self.output_queue.get(block=False)
                
        except queue.Full:
            logger.warning("Processing queue full, dropping frame")
            
        return None


class Renderer:
    """Optimized rendering with hardware acceleration"""
    
    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.8
        self.thickness = 2
        
    def render_emotion_results(self, frame: np.ndarray, results: List[Dict]) -> np.ndarray:
        """Render emotion detection results on frame"""
        # Create overlay for transparency effects
        overlay = frame.copy()
        
        for result in results:
            x, y, w, h = result['bbox']
            emotion = result['emotion']
            confidence = result['confidence']
            predictions = result['predictions']
            
            # Draw face rectangle with emotion color
            cv2.rectangle(overlay, (x, y), (x + w, y + h), emotion.color, 3)
            
            # Draw emotion label with background
            label = f"{emotion.name}: {confidence:.1%}"
            label_size = cv2.getTextSize(label, self.font, self.font_scale, self.thickness)[0]
            
            # Semi-transparent background
            cv2.rectangle(overlay,
                         (x, y - label_size[1] - 10),
                         (x + label_size[0], y),
                         emotion.color, -1)
            
            # Blend overlay
            frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
            
            # Draw text
            cv2.putText(frame, label,
                       (x, y - 5),
                       self.font, self.font_scale, (255, 255, 255), self.thickness)
            
            # Draw probability bars
            self._render_probability_bars(frame, predictions, x + w + 10, y)
            
        return frame
    
    def _render_probability_bars(self, frame: np.ndarray, predictions: np.ndarray,
                                x: int, y: int) -> None:
        """Render probability bars with smooth animation"""
        bar_width = 200
        bar_height = 20
        spacing = 5
        
        # Check if bars fit in frame
        if x + bar_width > frame.shape[1]:
            x = 10
            
        for i, (emotion, prob) in enumerate(zip(EmotionLabels.all(), predictions)):
            y_pos = y + i * (bar_height + spacing)
            
            # Background
            cv2.rectangle(frame,
                         (x, y_pos),
                         (x + bar_width, y_pos + bar_height),
                         (40, 40, 40), -1)
            
            # Probability bar
            bar_length = int(prob * bar_width)
            cv2.rectangle(frame,
                         (x, y_pos),
                         (x + bar_length, y_pos + bar_height),
                         emotion.color, -1)
            
            # Text label
            label = f"{emotion.name}: {prob:.1%}"
            cv2.putText(frame, label,
                       (x + bar_width + 5, y_pos + bar_height - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def render_fps(self, frame: np.ndarray, fps: float) -> None:
        """Render FPS counter"""
        label = f"FPS: {fps:.1f}"
        cv2.putText(frame, label,
                   (10, 30),
                   self.font, 1.0, (0, 255, 0), 2)
    
    def render_info(self, frame: np.ndarray, info: Dict[str, str]) -> None:
        """Render additional information"""
        y = 60
        for key, value in info.items():
            label = f"{key}: {value}"
            cv2.putText(frame, label,
                       (10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            y += 25


class CameraCapture:
    """Optimized camera capture with buffering"""
    
    def __init__(self, camera_id: int = 0):
        self.camera_id = camera_id
        self.cap = None
        self.buffer = queue.Queue(maxsize=2)
        self.capture_thread = None
        self.running = False
        
    def start(self) -> bool:
        """Start camera capture"""
        self.cap = cv2.VideoCapture(self.camera_id)
        
        # Set camera properties for optimal performance
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not self.cap.isOpened():
            return False
            
        self.running = True
        self.capture_thread = threading.Thread(target=self._capture_frames, daemon=True)
        self.capture_thread.start()
        
        return True
    
    def _capture_frames(self):
        """Background thread for capturing frames"""
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                # Drop old frames
                if self.buffer.full():
                    try:
                        self.buffer.get_nowait()
                    except queue.Empty:
                        pass
                        
                self.buffer.put(frame)
    
    def read(self) -> Optional[np.ndarray]:
        """Read latest frame"""
        try:
            return self.buffer.get(timeout=0.1)
        except queue.Empty:
            return None
    
    def stop(self):
        """Stop camera capture"""
        self.running = False
        if self.capture_thread:
            self.capture_thread.join()
        if self.cap:
            self.cap.release()


class EmotionDetectorPro:
    """Professional emotion detection system with advanced optimization"""
    
    def __init__(self, model_path: str = 'best_emotion_model.keras',
                 use_dnn_face_detector: bool = False):
        logger.info("Initializing Professional Emotion Detector...")
        
        # Initialize components
        if use_dnn_face_detector:
            self.face_detector = DNNFaceDetector()
        else:
            self.face_detector = HaarCascadeFaceDetector()
            
        self.emotion_predictor = EmotionPredictor(model_path)
        self.frame_processor = FrameProcessor(self.face_detector, self.emotion_predictor)
        self.renderer = Renderer()
        self.camera = CameraCapture()
        
        # Performance tracking
        self.fps_buffer = deque(maxlen=30)
        self.frame_count = 0
        self.start_time = time.time()
        
        logger.info("Emotion Detector ready!")
    
    @contextmanager
    def performance_timer(self, name: str):
        """Context manager for performance timing"""
        start = time.time()
        yield
        elapsed = (time.time() - start) * 1000
        logger.debug(f"{name} took {elapsed:.2f}ms")
    
    def run(self):
        """Run emotion detection system"""
        if not self.camera.start():
            logger.error("Failed to start camera")
            return
            
        logger.info("Starting emotion detection...")
        logger.info("Controls: Q=Quit, S=Screenshot, R=Reset filters, D=Debug mode")
        
        debug_mode = False
        
        try:
            while True:
                # Get frame
                frame = self.camera.read()
                if frame is None:
                    continue
                
                # Process frame
                start_time = time.time()
                result = self.frame_processor.process(frame)
                
                if result:
                    frame, results = result
                    
                    # Render results
                    frame = self.renderer.render_emotion_results(frame, results)
                    
                    # Calculate FPS
                    fps = 1.0 / (time.time() - start_time)
                    self.fps_buffer.append(fps)
                    avg_fps = np.mean(self.fps_buffer)
                    
                    # Render FPS
                    self.renderer.render_fps(frame, avg_fps)
                    
                    # Debug info
                    if debug_mode:
                        info = {
                            'Faces': len(results),
                            'Frame': self.frame_count,
                            'Queue': self.frame_processor.input_queue.qsize()
                        }
                        self.renderer.render_info(frame, info)
                
                # Display frame
                cv2.imshow('Emotion Detector Pro', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    self._save_screenshot(frame)
                elif key == ord('r'):
                    self._reset_filters()
                elif key == ord('d'):
                    debug_mode = not debug_mode
                    logger.info(f"Debug mode: {debug_mode}")
                
                self.frame_count += 1
                
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self.cleanup()
    
    def _save_screenshot(self, frame: np.ndarray):
        """Save screenshot with timestamp"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"emotion_pro_{timestamp}.png"
        cv2.imwrite(filename, frame)
        logger.info(f"Screenshot saved: {filename}")
    
    def _reset_filters(self):
        """Reset Kalman filters"""
        self.frame_processor.kalman_filters.clear()
        logger.info("Filters reset")
    
    def cleanup(self):
        """Clean up resources"""
        logger.info("Cleaning up...")
        self.camera.stop()
        cv2.destroyAllWindows()
        
        # Print statistics
        total_time = time.time() - self.start_time
        avg_fps = self.frame_count / total_time
        logger.info(f"Total frames: {self.frame_count}")
        logger.info(f"Total time: {total_time:.2f}s")
        logger.info(f"Average FPS: {avg_fps:.2f}")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Professional Emotion Detection System'
    )
    parser.add_argument('--model', type=str, default='best_emotion_model.keras',
                        help='Path to emotion detection model')
    parser.add_argument('--dnn', action='store_true',
                        help='Use DNN face detector (more accurate but slower)')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging')
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Run detector
    detector = EmotionDetectorPro(args.model, args.dnn)
    detector.run()


if __name__ == "__main__":
    main()