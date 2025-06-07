#!/usr/bin/env python3
"""
Emotion Detection Application
Professional real-time emotion detection using Mac camera
"""

import os
import sys
import argparse
from emotion_detector import EmotionDetector

def print_banner():
    """Print application banner"""
    print("\n" + "="*50)
    print("      EMOTION DETECTION SYSTEM")
    print("        Real-time Analysis")
    print("="*50 + "\n")

def main():
    # Print banner
    print_banner()
    
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Professional Emotion Detection System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Use default model with camera
  python main.py --model my_model.keras  # Use custom model
  python main.py --help             # Show this help
        """
    )
    
    parser.add_argument(
        '--model', 
        type=str, 
        default='../models/best_emotion_model.keras',
        help='Path to emotion detection model (default: best_emotion_model.keras)'
    )
    
    parser.add_argument(
        '--train',
        action='store_true',
        help='Train a new model instead of running detection'
    )
    
    args = parser.parse_args()
    
    if args.train:
        # Train new model
        print("Starting model training...")
        print("This will train a new emotion detection model.")
        response = input("Continue? (y/n): ")
        
        if response.lower() == 'y':
            try:
                from train_improved_model import train_model
                train_model()
            except Exception as e:
                print(f"Error during training: {e}")
                print("\nMake sure you have the FER2013 dataset downloaded.")
                print("Update the BASE_DIR in train_improved_model.py to point to your dataset.")
                sys.exit(1)
        else:
            print("Training cancelled.")
            sys.exit(0)
    else:
        # Run emotion detection
        try:
            detector = EmotionDetector(args.model)
            print("\nStarting emotion detection...")
            print("Make sure your camera is connected and permissions are granted.\n")
            detector.run_camera()
        except FileNotFoundError as e:
            print(f"\nError: {e}")
            print("\nTo train a new model, run: python main.py --train")
            sys.exit(1)
        except Exception as e:
            print(f"\nError: {e}")
            print("\nTroubleshooting:")
            print("1. Make sure your camera is connected")
            print("2. Grant camera permissions to Terminal/Python")
            print("3. Check that all requirements are installed")
            sys.exit(1)

if __name__ == "__main__":
    main()