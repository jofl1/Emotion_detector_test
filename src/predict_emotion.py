#!/usr/bin/env python3
"""
Command-line emotion detection script
Usage: python predict_emotion.py <image_path>
"""

import sys
import os
import tensorflow as tf
import numpy as np
from PIL import Image
import argparse

# Emotion labels
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

def load_model(model_path=None):
    """Load the trained emotion detection model"""
    if model_path and os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    
    # Try different model paths
    model_paths = [
        'fer2013_emotion_detector_final.keras',
        'best_emotion_model.keras',
        'fine_tuned_best_model.keras',
        'initial_best_model.keras'
    ]
    
    for path in model_paths:
        if os.path.exists(path):
            print(f"Loading model from: {path}")
            return tf.keras.models.load_model(path)
    
    raise FileNotFoundError("No trained model found!")

def preprocess_image(image_path, target_size=(48, 48)):
    """Load and preprocess image for prediction"""
    # Load image as grayscale
    img = Image.open(image_path).convert('L')
    
    # Resize
    img = img.resize(target_size, Image.Resampling.LANCZOS)
    
    # Convert to array and normalize
    img_array = np.array(img)
    img_array = img_array.astype('float32') / 255.0
    
    # Add dimensions
    img_array = np.expand_dims(img_array, axis=-1)  # Channel dimension
    img_array = np.expand_dims(img_array, axis=0)   # Batch dimension
    
    return img_array

def predict_emotion(model, image_path, verbose=True):
    """Predict emotion from image file"""
    # Preprocess image
    processed_image = preprocess_image(image_path)
    
    # Make prediction
    predictions = model.predict(processed_image, verbose=0)
    emotion_idx = np.argmax(predictions[0])
    emotion = EMOTION_LABELS[emotion_idx]
    confidence = predictions[0][emotion_idx]
    
    if verbose:
        print("\n" + "="*50)
        print(f"Image: {os.path.basename(image_path)}")
        print("="*50)
        print(f"\nPredicted Emotion: {emotion}")
        print(f"Confidence: {confidence:.1%}")
        print("\nAll Probabilities:")
        print("-"*30)
        for i, (label, prob) in enumerate(zip(EMOTION_LABELS, predictions[0])):
            bar = "█" * int(prob * 20)
            print(f"{label:10} {prob:6.1%} {bar}")
        print("="*50 + "\n")
    
    return emotion, confidence, predictions[0]

def batch_predict(model, image_dir, output_file=None):
    """Predict emotions for all images in a directory"""
    results = []
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    image_files = [f for f in os.listdir(image_dir) 
                   if os.path.splitext(f)[1].lower() in image_extensions]
    
    print(f"Processing {len(image_files)} images...")
    
    for img_file in image_files:
        img_path = os.path.join(image_dir, img_file)
        try:
            emotion, confidence, probs = predict_emotion(model, img_path, verbose=False)
            results.append({
                'file': img_file,
                'emotion': emotion,
                'confidence': confidence
            })
            print(f"✓ {img_file}: {emotion} ({confidence:.1%})")
        except Exception as e:
            print(f"✗ {img_file}: Error - {str(e)}")
            results.append({
                'file': img_file,
                'emotion': 'Error',
                'confidence': 0.0
            })
    
    # Save results if output file specified
    if output_file:
        import csv
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['file', 'emotion', 'confidence'])
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults saved to: {output_file}")
    
    return results

def main():
    parser = argparse.ArgumentParser(
        description='Predict emotions from facial images using deep learning'
    )
    parser.add_argument('input', help='Path to image file or directory')
    parser.add_argument('--model', '-m', help='Path to model file')
    parser.add_argument('--batch', '-b', action='store_true', 
                       help='Process all images in directory')
    parser.add_argument('--output', '-o', help='Output CSV file for batch results')
    parser.add_argument('--quiet', '-q', action='store_true', 
                       help='Minimal output')
    
    args = parser.parse_args()
    
    # Load model
    try:
        model = load_model(args.model)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nPlease ensure you have trained the model first.")
        sys.exit(1)
    
    # Process input
    if args.batch or os.path.isdir(args.input):
        if not os.path.isdir(args.input):
            print(f"Error: {args.input} is not a directory")
            sys.exit(1)
        batch_predict(model, args.input, args.output)
    else:
        if not os.path.isfile(args.input):
            print(f"Error: {args.input} is not a file")
            sys.exit(1)
        emotion, confidence, _ = predict_emotion(
            model, args.input, verbose=not args.quiet
        )
        if args.quiet:
            print(f"{emotion} ({confidence:.1%})")

if __name__ == "__main__":
    main()