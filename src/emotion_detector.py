#!/usr/bin/env python3
"""
Professional Real-time Emotion Detection for Mac Camera
Clean, efficient implementation with high accuracy
"""

import cv2
import numpy as np
import tensorflow as tf
import time
from collections import deque
import argparse
import os

# Emotion labels
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Colors for each emotion (BGR format for OpenCV)
EMOTION_COLORS = {
    'Angry': (0, 0, 255),      # Red
    'Disgust': (128, 0, 128),  # Purple
    'Fear': (255, 128, 0),     # Orange
    'Happy': (0, 255, 255),    # Yellow
    'Neutral': (128, 128, 128), # Gray
    'Sad': (255, 0, 0),        # Blue
    'Surprise': (0, 255, 0)    # Green
}

class EmotionDetector:
    def __init__(self, model_path='../models/best_emotion_model.keras'):
        """Initialize the emotion detector"""
        print("Initializing Emotion Detector...")
        
        # Load model
        self.model = self.load_model(model_path)
        
        # Load face detector
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Performance tracking
        self.fps_buffer = deque(maxlen=30)
        self.emotion_buffer = deque(maxlen=10)  # Smooth predictions
        
        print("Emotion Detector ready!")
    
    def load_model(self, model_path):
        """Load the emotion detection model"""
        if not os.path.exists(model_path):
            # Try alternative paths
            alt_paths = [
                '../models/emotion_model_final.keras',
                '../models/best_emotion_model_improved.keras',
                '../models/fer2013_emotion_detector_final.keras',
                'best_emotion_model.keras'  # fallback to current dir
            ]
            for path in alt_paths:
                if os.path.exists(path):
                    model_path = path
                    break
            else:
                raise FileNotFoundError(
                    "No emotion detection model found. "
                    "Please train a model first using train_improved_model.py"
                )
        
        print(f"Loading model from: {model_path}")
        return tf.keras.models.load_model(model_path)
    
    def preprocess_face(self, face_img):
        """Preprocess face for emotion detection"""
        # Resize to 48x48
        face_img = cv2.resize(face_img, (48, 48))
        
        # Convert to grayscale if needed
        if len(face_img.shape) == 3:
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        
        # Normalize
        face_img = face_img.astype('float32') / 255.0
        
        # Reshape for model
        face_img = np.expand_dims(face_img, axis=-1)  # Add channel dimension
        face_img = np.expand_dims(face_img, axis=0)   # Add batch dimension
        
        return face_img
    
    def detect_faces(self, frame):
        """Detect faces in frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(48, 48)
        )
        return faces
    
    def predict_emotion(self, face_img):
        """Predict emotion from face image"""
        processed = self.preprocess_face(face_img)
        predictions = self.model.predict(processed, verbose=0)[0]
        
        # Get top emotion
        emotion_idx = np.argmax(predictions)
        emotion = EMOTION_LABELS[emotion_idx]
        confidence = predictions[emotion_idx]
        
        return emotion, confidence, predictions
    
    def smooth_predictions(self, predictions):
        """Smooth predictions over time to reduce jitter"""
        self.emotion_buffer.append(predictions)
        if len(self.emotion_buffer) > 1:
            # Average recent predictions
            avg_predictions = np.mean(self.emotion_buffer, axis=0)
            return avg_predictions
        return predictions
    
    def draw_results(self, frame, face_coords, emotion, confidence, predictions):
        """Draw detection results on frame"""
        x, y, w, h = face_coords
        color = EMOTION_COLORS[emotion]
        
        # Draw face rectangle
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        
        # Draw emotion label with background
        label = f"{emotion}: {confidence:.1%}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        
        # Background rectangle for label
        cv2.rectangle(frame,
                     (x, y - label_size[1] - 10),
                     (x + label_size[0], y),
                     color, -1)
        
        # Text
        cv2.putText(frame, label,
                   (x, y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Draw emotion probabilities bar
        self.draw_emotion_bars(frame, predictions, x + w + 10, y)
        
        return frame
    
    def draw_emotion_bars(self, frame, predictions, x, y):
        """Draw emotion probability bars"""
        bar_width = 200
        bar_height = 20
        spacing = 5
        
        # Check if bars fit in frame
        if x + bar_width > frame.shape[1]:
            x = 10  # Move to left side
        
        for i, (emotion, prob) in enumerate(zip(EMOTION_LABELS, predictions)):
            # Bar background
            y_pos = y + i * (bar_height + spacing)
            cv2.rectangle(frame,
                         (x, y_pos),
                         (x + bar_width, y_pos + bar_height),
                         (50, 50, 50), -1)
            
            # Probability bar
            bar_length = int(prob * bar_width)
            color = EMOTION_COLORS[emotion]
            cv2.rectangle(frame,
                         (x, y_pos),
                         (x + bar_length, y_pos + bar_height),
                         color, -1)
            
            # Label
            label = f"{emotion}: {prob:.1%}"
            cv2.putText(frame, label,
                       (x + bar_width + 5, y_pos + bar_height - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def draw_fps(self, frame, fps):
        """Draw FPS counter"""
        cv2.putText(frame, f"FPS: {fps:.1f}",
                   (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    def run_camera(self):
        """Run emotion detection on camera feed"""
        # Open camera (0 for default camera on Mac)
        cap = cv2.VideoCapture(0)
        
        # Set camera properties for better quality
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("\nCamera started!")
        print("Press 'q' to quit")
        print("Press 's' to save screenshot")
        print("Press 'r' to reset emotion smoothing")
        
        while True:
            start_time = time.time()
            
            # Read frame
            ret, frame = cap.read()
            if not ret:
                print("Failed to read from camera")
                break
            
            # Detect faces
            faces = self.detect_faces(frame)
            
            # Process each face
            for (x, y, w, h) in faces:
                # Extract face region
                face_roi = frame[y:y+h, x:x+w]
                
                try:
                    # Predict emotion
                    emotion, confidence, predictions = self.predict_emotion(face_roi)
                    
                    # Smooth predictions
                    smoothed = self.smooth_predictions(predictions)
                    emotion_idx = np.argmax(smoothed)
                    emotion = EMOTION_LABELS[emotion_idx]
                    confidence = smoothed[emotion_idx]
                    
                    # Draw results
                    self.draw_results(frame, (x, y, w, h), emotion, confidence, smoothed)
                    
                except Exception as e:
                    print(f"Error processing face: {e}")
            
            # Calculate and display FPS
            fps = 1.0 / (time.time() - start_time)
            self.fps_buffer.append(fps)
            avg_fps = np.mean(self.fps_buffer)
            self.draw_fps(frame, avg_fps)
            
            # Show frame
            cv2.imshow('Emotion Detection - Mac Camera', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save screenshot
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"emotion_screenshot_{timestamp}.png"
                cv2.imwrite(filename, frame)
                print(f"Screenshot saved: {filename}")
            elif key == ord('r'):
                # Reset emotion buffer
                self.emotion_buffer.clear()
                print("Emotion smoothing reset")
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("\nCamera stopped")

def main():
    parser = argparse.ArgumentParser(description='Real-time Emotion Detection')
    parser.add_argument('--model', type=str, default='best_emotion_model.keras',
                        help='Path to the emotion detection model')
    parser.add_argument('--video', type=str, help='Process video file instead of camera')
    
    args = parser.parse_args()
    
    # Create detector
    detector = EmotionDetector(args.model)
    
    # Run detection
    if args.video:
        print(f"Processing video: {args.video}")
        # Video processing not implemented in this version
        print("Video processing not yet implemented. Using camera instead.")
        detector.run_camera()
    else:
        detector.run_camera()

if __name__ == "__main__":
    main()