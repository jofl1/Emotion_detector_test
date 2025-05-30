#!/usr/bin/env python3
"""
Real-time Video Emotion Detection
Detects emotions from webcam feed or video file in real-time
"""

import cv2
import numpy as np
import tensorflow as tf
import argparse
import time
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import yaml

# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Emotion labels and colors
EMOTION_LABELS = config['emotions']
EMOTION_COLORS = [
    (255, 107, 107),  # Angry - Red
    (132, 94, 194),   # Disgust - Purple
    (78, 131, 151),   # Fear - Blue-gray
    (255, 199, 95),   # Happy - Yellow
    (147, 147, 147),  # Neutral - Gray
    (77, 128, 118),   # Sad - Teal
    (249, 248, 113),  # Surprise - Light yellow
]

class VideoEmotionDetector:
    def __init__(self, model_path=None):
        """Initialize the video emotion detector"""
        self.model = self.load_model(model_path)
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.emotion_history = deque(maxlen=30)  # Store last 30 frames
        self.fps_history = deque(maxlen=30)
        
    def load_model(self, model_path=None):
        """Load the emotion detection model"""
        if model_path:
            return tf.keras.models.load_model(model_path)
        
        # Try default paths
        model_paths = [
            'models/fer2013_emotion_detector_final.keras',
            'fer2013_emotion_detector_final.keras',
            'best_emotion_model.keras',
            'fine_tuned_best_model.keras'
        ]
        
        for path in model_paths:
            try:
                print(f"Loading model from: {path}")
                return tf.keras.models.load_model(path)
            except:
                continue
                
        raise FileNotFoundError("No trained model found!")
    
    def preprocess_face(self, face_img):
        """Preprocess face image for model"""
        # Resize to model input size
        face_img = cv2.resize(face_img, (48, 48))
        
        # Convert to RGB if grayscale
        if len(face_img.shape) == 2:
            face_img = cv2.cvtColor(face_img, cv2.COLOR_GRAY2RGB)
        
        # Normalize
        face_img = (face_img - 127.5) / 127.5
        
        # Add batch dimension
        return np.expand_dims(face_img, axis=0)
    
    def detect_faces(self, frame):
        """Detect faces in frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=config['inference']['face_detection_scale'],
            minNeighbors=config['inference']['face_detection_neighbors'],
            minSize=(30, 30)
        )
        return faces
    
    def predict_emotion(self, face_img):
        """Predict emotion from face image"""
        processed = self.preprocess_face(face_img)
        predictions = self.model.predict(processed, verbose=0)
        return predictions[0]
    
    def draw_emotion_bar(self, frame, predictions, x, y, w, h):
        """Draw emotion probability bars on frame"""
        bar_width = 150
        bar_height = 15
        start_x = x + w + 10
        start_y = y
        
        # Ensure bars fit in frame
        if start_x + bar_width > frame.shape[1]:
            start_x = x - bar_width - 10
        
        for i, (emotion, prob) in enumerate(zip(EMOTION_LABELS, predictions)):
            # Background bar
            cv2.rectangle(frame, 
                         (start_x, start_y + i * (bar_height + 5)),
                         (start_x + bar_width, start_y + (i + 1) * bar_height + i * 5),
                         (50, 50, 50), -1)
            
            # Probability bar
            bar_length = int(prob * bar_width)
            color = EMOTION_COLORS[i]
            cv2.rectangle(frame,
                         (start_x, start_y + i * (bar_height + 5)),
                         (start_x + bar_length, start_y + (i + 1) * bar_height + i * 5),
                         color, -1)
            
            # Label
            label = f"{emotion}: {prob:.1%}"
            cv2.putText(frame, label,
                       (start_x + bar_width + 5, start_y + (i + 1) * bar_height + i * 5 - 3),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def create_emotion_graph(self):
        """Create emotion history graph"""
        if len(self.emotion_history) < 2:
            return None
            
        fig, ax = plt.subplots(figsize=(6, 4), facecolor='black')
        ax.set_facecolor('black')
        
        # Plot emotion trends
        emotions_over_time = np.array(list(self.emotion_history))
        x = range(len(emotions_over_time))
        
        for i, emotion in enumerate(EMOTION_LABELS):
            color = np.array(EMOTION_COLORS[i]) / 255.0
            ax.plot(x, emotions_over_time[:, i], 
                   label=emotion, color=color, linewidth=2)
        
        ax.set_ylim(0, 1)
        ax.set_xlim(0, 29)
        ax.set_xlabel('Frame', color='white')
        ax.set_ylabel('Probability', color='white')
        ax.set_title('Emotion Trends (Last 30 Frames)', color='white')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Style
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_edgecolor('white')
        
        # Convert to image
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        graph_img = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
        graph_img = graph_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        
        return cv2.cvtColor(graph_img, cv2.COLOR_RGB2BGR)
    
    def process_frame(self, frame):
        """Process a single frame"""
        # Detect faces
        faces = self.detect_faces(frame)
        
        for (x, y, w, h) in faces:
            # Extract face
            face_img = frame[y:y+h, x:x+w]
            
            # Predict emotion
            predictions = self.predict_emotion(face_img)
            emotion_idx = np.argmax(predictions)
            emotion = EMOTION_LABELS[emotion_idx]
            confidence = predictions[emotion_idx]
            
            # Store in history
            self.emotion_history.append(predictions)
            
            # Draw bounding box
            color = EMOTION_COLORS[emotion_idx]
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            # Draw emotion label
            label = f"{emotion}: {confidence:.1%}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame, 
                         (x, y - label_size[1] - 10),
                         (x + label_size[0], y),
                         color, -1)
            cv2.putText(frame, label,
                       (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw probability bars
            self.draw_emotion_bar(frame, predictions, x, y, w, h)
        
        return frame
    
    def run_webcam(self, show_graph=True):
        """Run emotion detection on webcam"""
        cap = cv2.VideoCapture(0)
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print("Press 'q' to quit, 's' to save screenshot, 'g' to toggle graph")
        
        while True:
            start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            processed_frame = self.process_frame(frame)
            
            # Calculate FPS
            fps = 1 / (time.time() - start_time)
            self.fps_history.append(fps)
            avg_fps = np.mean(self.fps_history)
            
            # Draw FPS
            cv2.putText(processed_frame, f"FPS: {avg_fps:.1f}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Add emotion graph if enabled
            if show_graph and len(self.emotion_history) > 1:
                graph = self.create_emotion_graph()
                if graph is not None:
                    # Resize graph to fit
                    graph = cv2.resize(graph, (400, 300))
                    # Place graph in corner
                    processed_frame[10:310, -410:-10] = graph
            
            # Display
            cv2.imshow('Emotion Detection', processed_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f"emotion_screenshot_{time.strftime('%Y%m%d_%H%M%S')}.png"
                cv2.imwrite(filename, processed_frame)
                print(f"Screenshot saved: {filename}")
            elif key == ord('g'):
                show_graph = not show_graph
        
        cap.release()
        cv2.destroyAllWindows()
    
    def process_video(self, video_path, output_path=None, show_graph=True):
        """Process video file"""
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Setup video writer if output path specified
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        print(f"Processing video: {video_path}")
        print(f"Total frames: {total_frames}")
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            processed_frame = self.process_frame(frame)
            
            # Add progress
            progress = (frame_count / total_frames) * 100
            cv2.putText(processed_frame, f"Progress: {progress:.1f}%",
                       (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Add graph if enabled
            if show_graph and len(self.emotion_history) > 1:
                graph = self.create_emotion_graph()
                if graph is not None:
                    graph = cv2.resize(graph, (400, 300))
                    processed_frame[10:310, -410:-10] = graph
            
            # Write frame
            if output_path:
                out.write(processed_frame)
            
            # Display (optional)
            cv2.imshow('Processing Video', processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"Processed {frame_count}/{total_frames} frames")
        
        cap.release()
        if output_path:
            out.release()
            print(f"Output saved: {output_path}")
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='Real-time video emotion detection')
    parser.add_argument('--input', '-i', help='Video file path (omit for webcam)')
    parser.add_argument('--output', '-o', help='Output video path')
    parser.add_argument('--model', '-m', help='Model file path')
    parser.add_argument('--no-graph', action='store_true', help='Disable emotion graph')
    
    args = parser.parse_args()
    
    # Initialize detector
    try:
        detector = VideoEmotionDetector(args.model)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Run detection
    if args.input:
        detector.process_video(args.input, args.output, not args.no_graph)
    else:
        detector.run_webcam(not args.no_graph)

if __name__ == "__main__":
    main()