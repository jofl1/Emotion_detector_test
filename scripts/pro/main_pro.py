#!/usr/bin/env python3
"""
Professional Emotion Detection System - Main Entry Point
Enterprise-grade implementation with advanced features
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional
import yaml

# Import our professional modules
from emotion_detector_pro import EmotionDetectorPro
from train_advanced import AdvancedTrainer, TrainingConfig
from inference_engine import InferenceEngine, ModelOptimizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EmotionDetectionSystem:
    """Main system orchestrator"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self._setup_directories()
        
    def _load_config(self, config_path: Optional[str]) -> dict:
        """Load system configuration"""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            # Default configuration
            return {
                'model': {
                    'path': 'best_emotion_model.keras',
                    'backend': 'auto',
                    'use_gpu': True
                },
                'camera': {
                    'device_id': 0,
                    'resolution': [1280, 720],
                    'fps': 30
                },
                'inference': {
                    'batch_size': 1,
                    'num_threads': 4,
                    'use_optimization': True
                },
                'training': {
                    'dataset_path': '~/Python/archive',
                    'batch_size': 64,
                    'epochs': 100
                }
            }
    
    def _setup_directories(self):
        """Create necessary directories"""
        dirs = ['models', 'logs', 'exports', 'checkpoints']
        for dir_name in dirs:
            Path(dir_name).mkdir(exist_ok=True)
    
    def run_detection(self, model_path: Optional[str] = None, use_dnn: bool = False):
        """Run real-time emotion detection"""
        if not model_path:
            model_path = self._find_best_model()
        
        logger.info(f"Starting emotion detection with model: {model_path}")
        
        # Check if we should use optimized inference
        if self.config['inference']['use_optimization'] and model_path.endswith('.keras'):
            # Convert to optimized format
            logger.info("Converting model for optimized inference...")
            optimized_path = self._optimize_model(model_path)
            if optimized_path:
                model_path = optimized_path
        
        # Create and run detector
        detector = EmotionDetectorPro(model_path, use_dnn_face_detector=use_dnn)
        detector.run()
    
    def train_model(self, config_path: Optional[str] = None):
        """Train advanced emotion detection model"""
        logger.info("Starting advanced model training...")
        
        # Load training config
        if config_path:
            train_config = TrainingConfig.from_yaml(config_path)
        else:
            train_config = TrainingConfig(
                train_dir=os.path.expanduser(self.config['training']['dataset_path'] + '/train'),
                test_dir=os.path.expanduser(self.config['training']['dataset_path'] + '/test'),
                batch_size=self.config['training']['batch_size'],
                epochs=self.config['training']['epochs']
            )
        
        # Create trainer and train
        trainer = AdvancedTrainer(train_config)
        trainer.train()
        
        logger.info("Training completed successfully!")
    
    def optimize_model(self, model_path: str, output_format: str = 'tflite'):
        """Optimize model for deployment"""
        logger.info(f"Optimizing model to {output_format} format...")
        
        output_path = f"exports/emotion_model_optimized.{output_format}"
        
        if output_format == 'tflite':
            ModelOptimizer.convert_to_tflite(model_path, output_path, quantize=True)
        elif output_format == 'onnx':
            ModelOptimizer.convert_to_onnx(model_path, output_path)
        elif output_format == 'coreml':
            ModelOptimizer.convert_to_coreml(model_path, output_path)
        else:
            raise ValueError(f"Unknown format: {output_format}")
        
        logger.info(f"Model optimized and saved to: {output_path}")
        return output_path
    
    def benchmark_model(self, model_path: str):
        """Benchmark model performance"""
        logger.info("Running performance benchmark...")
        
        # Create inference engine
        engine = InferenceEngine(model_path, backend='auto')
        
        # Run benchmark
        results = engine.benchmark(num_runs=100)
        
        # Display results
        print("\n" + "="*50)
        print("PERFORMANCE BENCHMARK RESULTS")
        print("="*50)
        for key, value in results.items():
            if isinstance(value, float):
                print(f"{key:20}: {value:10.2f}")
            else:
                print(f"{key:20}: {value}")
        print("="*50)
    
    def _find_best_model(self) -> str:
        """Find the best available model"""
        model_paths = [
            'models/emotion_model_advanced_final.keras',
            'models/best_model_advanced.keras',
            'best_emotion_model.keras',
            'emotion_model_final.keras',
            'exports/emotion_model_optimized.tflite',
            'exports/emotion_model_optimized.mlmodel'
        ]
        
        for path in model_paths:
            if Path(path).exists():
                return path
        
        raise FileNotFoundError(
            "No trained model found. Please train a model first using --train"
        )
    
    def _optimize_model(self, model_path: str) -> Optional[str]:
        """Auto-optimize model for current platform"""
        import platform
        
        system = platform.system()
        
        try:
            if system == 'Darwin':  # macOS
                # Use Core ML for Mac
                output_path = 'exports/emotion_model_optimized.mlmodel'
                if not Path(output_path).exists():
                    return self.optimize_model(model_path, 'coreml')
                return output_path
            else:
                # Use TFLite for other platforms
                output_path = 'exports/emotion_model_optimized.tflite'
                if not Path(output_path).exists():
                    return self.optimize_model(model_path, 'tflite')
                return output_path
        except Exception as e:
            logger.warning(f"Failed to optimize model: {e}")
            return None


def print_banner():
    """Print professional banner"""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║          PROFESSIONAL EMOTION DETECTION SYSTEM               ║
║                  Advanced Deep Learning                      ║
║                    Version 2.0 Pro                          ║
╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def main():
    """Main entry point"""
    print_banner()
    
    parser = argparse.ArgumentParser(
        description='Professional Emotion Detection System',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        '--detect', 
        action='store_true',
        help='Run real-time emotion detection'
    )
    mode_group.add_argument(
        '--train',
        action='store_true',
        help='Train advanced emotion detection model'
    )
    mode_group.add_argument(
        '--optimize',
        metavar='MODEL_PATH',
        help='Optimize model for deployment'
    )
    mode_group.add_argument(
        '--benchmark',
        metavar='MODEL_PATH',
        help='Benchmark model performance'
    )
    
    # Additional options
    parser.add_argument(
        '--config',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--model',
        help='Path to model file (for detection mode)'
    )
    parser.add_argument(
        '--format',
        choices=['tflite', 'onnx', 'coreml'],
        default='tflite',
        help='Output format for optimization'
    )
    parser.add_argument(
        '--dnn',
        action='store_true',
        help='Use DNN face detector (more accurate)'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    
    args = parser.parse_args()
    
    # Set debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create system
    system = EmotionDetectionSystem(args.config)
    
    try:
        if args.detect:
            # Run detection
            system.run_detection(args.model, args.dnn)
            
        elif args.train:
            # Train model
            system.train_model(args.config)
            
        elif args.optimize:
            # Optimize model
            system.optimize_model(args.optimize, args.format)
            
        elif args.benchmark:
            # Benchmark model
            system.benchmark_model(args.benchmark)
            
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()