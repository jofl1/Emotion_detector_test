#!/usr/bin/env python3
"""
High-Performance Inference Engine for Emotion Detection
Optimized for real-time performance with TensorRT/CoreML
"""

import numpy as np
import tensorflow as tf
import cv2
from typing import List, Tuple, Dict, Optional, Union
from dataclasses import dataclass
import time
import threading
import queue
from abc import ABC, abstractmethod
import onnx
import onnxruntime as ort
import coremltools as ct
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)


@dataclass
class InferenceResult:
    """Structured inference result"""
    emotion_id: int
    emotion_name: str
    confidence: float
    probabilities: np.ndarray
    inference_time_ms: float
    
    def to_dict(self) -> Dict:
        return {
            'emotion': self.emotion_name,
            'confidence': float(self.confidence),
            'probabilities': self.probabilities.tolist(),
            'inference_time_ms': self.inference_time_ms
        }


class InferenceBackend(ABC):
    """Abstract base class for inference backends"""
    
    @abstractmethod
    def predict(self, image: np.ndarray) -> InferenceResult:
        pass
    
    @abstractmethod
    def warmup(self, num_runs: int = 10):
        pass
    
    @abstractmethod
    def get_backend_name(self) -> str:
        pass


class TensorFlowLiteBackend(InferenceBackend):
    """TensorFlow Lite backend for edge deployment"""
    
    def __init__(self, model_path: str, num_threads: int = 4):
        self.emotion_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
        
        # Load TFLite model
        self.interpreter = tf.lite.Interpreter(
            model_path=model_path,
            num_threads=num_threads
        )
        self.interpreter.allocate_tensors()
        
        # Get input/output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Determine input shape and type
        self.input_shape = self.input_details[0]['shape'][1:3]
        self.input_dtype = self.input_details[0]['dtype']
        
        logger.info(f"TFLite model loaded: {model_path}")
        logger.info(f"Input shape: {self.input_shape}, dtype: {self.input_dtype}")
    
    def predict(self, image: np.ndarray) -> InferenceResult:
        """Run inference on image"""
        start_time = time.time()
        
        # Preprocess
        processed = self._preprocess(image)
        
        # Run inference
        self.interpreter.set_tensor(self.input_details[0]['index'], processed)
        self.interpreter.invoke()
        
        # Get output
        probabilities = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        
        # Post-process
        emotion_id = np.argmax(probabilities)
        confidence = probabilities[emotion_id]
        
        inference_time = (time.time() - start_time) * 1000
        
        return InferenceResult(
            emotion_id=int(emotion_id),
            emotion_name=self.emotion_names[emotion_id],
            confidence=float(confidence),
            probabilities=probabilities,
            inference_time_ms=inference_time
        )
    
    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for TFLite model"""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Resize
        image = cv2.resize(image, tuple(self.input_shape))
        
        # Normalize based on input type
        if self.input_dtype == np.uint8:
            image = image.astype(np.uint8)
        else:
            image = image.astype(np.float32) / 255.0
        
        # Add batch and channel dimensions
        image = np.expand_dims(image, axis=-1)
        image = np.expand_dims(image, axis=0)
        
        return image
    
    def warmup(self, num_runs: int = 10):
        """Warmup the model"""
        dummy_input = np.random.rand(1, *self.input_shape, 1).astype(self.input_dtype)
        
        for _ in range(num_runs):
            self.interpreter.set_tensor(self.input_details[0]['index'], dummy_input)
            self.interpreter.invoke()
    
    def get_backend_name(self) -> str:
        return "TensorFlow Lite"


class ONNXRuntimeBackend(InferenceBackend):
    """ONNX Runtime backend for cross-platform deployment"""
    
    def __init__(self, model_path: str, providers: Optional[List[str]] = None):
        self.emotion_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
        
        # Set providers (CPU, CUDA, CoreML, etc.)
        if providers is None:
            providers = ['CPUExecutionProvider']
        
        # Create session
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self.session = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=providers
        )
        
        # Get input/output info
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape[1:3]
        self.output_name = self.session.get_outputs()[0].name
        
        logger.info(f"ONNX model loaded: {model_path}")
        logger.info(f"Providers: {self.session.get_providers()}")
    
    def predict(self, image: np.ndarray) -> InferenceResult:
        """Run inference on image"""
        start_time = time.time()
        
        # Preprocess
        processed = self._preprocess(image)
        
        # Run inference
        outputs = self.session.run(
            [self.output_name],
            {self.input_name: processed}
        )
        
        probabilities = outputs[0][0]
        
        # Post-process
        emotion_id = np.argmax(probabilities)
        confidence = probabilities[emotion_id]
        
        inference_time = (time.time() - start_time) * 1000
        
        return InferenceResult(
            emotion_id=int(emotion_id),
            emotion_name=self.emotion_names[emotion_id],
            confidence=float(confidence),
            probabilities=probabilities,
            inference_time_ms=inference_time
        )
    
    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for ONNX model"""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Resize
        image = cv2.resize(image, tuple(self.input_shape))
        
        # Normalize
        image = image.astype(np.float32) / 255.0
        
        # Add dimensions
        image = np.expand_dims(image, axis=-1)
        image = np.expand_dims(image, axis=0)
        
        return image
    
    def warmup(self, num_runs: int = 10):
        """Warmup the model"""
        dummy_input = np.random.rand(1, *self.input_shape, 1).astype(np.float32)
        
        for _ in range(num_runs):
            self.session.run([self.output_name], {self.input_name: dummy_input})
    
    def get_backend_name(self) -> str:
        return f"ONNX Runtime ({', '.join(self.session.get_providers())})"


class CoreMLBackend(InferenceBackend):
    """Core ML backend for Mac optimization"""
    
    def __init__(self, model_path: str):
        self.emotion_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
        
        # Load Core ML model
        self.model = ct.models.MLModel(model_path)
        
        # Get input shape from model spec
        spec = self.model.get_spec()
        input_description = spec.description.input[0]
        self.input_shape = (48, 48)  # Default for emotion detection
        
        logger.info(f"Core ML model loaded: {model_path}")
    
    def predict(self, image: np.ndarray) -> InferenceResult:
        """Run inference on image"""
        start_time = time.time()
        
        # Preprocess
        processed = self._preprocess(image)
        
        # Create input dictionary
        input_dict = {'input': processed}
        
        # Run inference
        output = self.model.predict(input_dict)
        
        # Extract probabilities
        probabilities = output['output']
        if isinstance(probabilities, dict):
            # Convert dict to array if needed
            probabilities = np.array([probabilities[str(i)] for i in range(7)])
        
        # Post-process
        emotion_id = np.argmax(probabilities)
        confidence = probabilities[emotion_id]
        
        inference_time = (time.time() - start_time) * 1000
        
        return InferenceResult(
            emotion_id=int(emotion_id),
            emotion_name=self.emotion_names[emotion_id],
            confidence=float(confidence),
            probabilities=probabilities,
            inference_time_ms=inference_time
        )
    
    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for Core ML model"""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Resize
        image = cv2.resize(image, self.input_shape)
        
        # Normalize
        image = image.astype(np.float32) / 255.0
        
        # Core ML expects CHW format
        image = np.expand_dims(image, axis=0)
        
        return image
    
    def warmup(self, num_runs: int = 10):
        """Warmup the model"""
        dummy_input = np.random.rand(1, *self.input_shape).astype(np.float32)
        
        for _ in range(num_runs):
            self.model.predict({'input': dummy_input})
    
    def get_backend_name(self) -> str:
        return "Core ML"


class BatchedInferenceEngine:
    """Batched inference engine for processing multiple faces efficiently"""
    
    def __init__(self, backend: InferenceBackend, batch_size: int = 8,
                 max_queue_size: int = 100):
        self.backend = backend
        self.batch_size = batch_size
        self.max_queue_size = max_queue_size
        
        # Queues for batching
        self.input_queue = queue.Queue(maxsize=max_queue_size)
        self.output_dict = {}
        self.processing_thread = threading.Thread(target=self._process_batches, daemon=True)
        self.processing_thread.start()
        
    def _process_batches(self):
        """Background thread for batch processing"""
        while True:
            batch = []
            batch_ids = []
            
            # Collect batch
            try:
                # Wait for first item
                item_id, image = self.input_queue.get(timeout=1)
                batch.append(image)
                batch_ids.append(item_id)
                
                # Try to fill batch
                while len(batch) < self.batch_size:
                    try:
                        item_id, image = self.input_queue.get_nowait()
                        batch.append(image)
                        batch_ids.append(item_id)
                    except queue.Empty:
                        break
                        
            except queue.Empty:
                continue
            
            # Process batch
            try:
                results = self._process_batch(batch)
                
                # Store results
                for item_id, result in zip(batch_ids, results):
                    self.output_dict[item_id] = result
                    
            except Exception as e:
                logger.error(f"Batch processing error: {e}")
    
    def _process_batch(self, images: List[np.ndarray]) -> List[InferenceResult]:
        """Process a batch of images"""
        results = []
        
        # For now, process individually (can be optimized for true batch inference)
        for image in images:
            result = self.backend.predict(image)
            results.append(result)
            
        return results
    
    def predict_async(self, image: np.ndarray, item_id: str) -> None:
        """Add image to processing queue"""
        self.input_queue.put((item_id, image))
    
    def get_result(self, item_id: str, timeout: float = 1.0) -> Optional[InferenceResult]:
        """Get result for item ID"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if item_id in self.output_dict:
                return self.output_dict.pop(item_id)
            time.sleep(0.001)
            
        return None


class ModelOptimizer:
    """Optimize and convert models for different backends"""
    
    @staticmethod
    def convert_to_tflite(model_path: str, output_path: str,
                         quantize: bool = True) -> str:
        """Convert Keras model to TFLite"""
        # Load model
        model = tf.keras.models.load_model(model_path)
        
        # Create converter
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        if quantize:
            # Apply optimizations
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            
            # Representative dataset for quantization
            def representative_dataset():
                for _ in range(100):
                    data = np.random.rand(1, 48, 48, 1).astype(np.float32)
                    yield [data]
                    
            converter.representative_dataset = representative_dataset
            
            # Full integer quantization
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.uint8
            converter.inference_output_type = tf.float32
        
        # Convert
        tflite_model = converter.convert()
        
        # Save
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
            
        logger.info(f"TFLite model saved to {output_path}")
        return output_path
    
    @staticmethod
    def convert_to_onnx(model_path: str, output_path: str) -> str:
        """Convert Keras model to ONNX"""
        import tf2onnx
        
        # Load model
        model = tf.keras.models.load_model(model_path)
        
        # Convert to ONNX
        onnx_model, _ = tf2onnx.convert.from_keras(
            model,
            input_signature=[tf.TensorSpec(shape=[None, 48, 48, 1], dtype=tf.float32)],
            opset=13
        )
        
        # Save
        onnx.save(onnx_model, output_path)
        
        logger.info(f"ONNX model saved to {output_path}")
        return output_path
    
    @staticmethod
    def convert_to_coreml(model_path: str, output_path: str) -> str:
        """Convert Keras model to Core ML"""
        # Load model
        model = tf.keras.models.load_model(model_path)
        
        # Convert to Core ML
        coreml_model = ct.convert(
            model,
            inputs=[ct.TensorType(shape=(1, 48, 48, 1))],
            minimum_deployment_target=ct.target.macOS12
        )
        
        # Add metadata
        coreml_model.author = 'Emotion Detection Pro'
        coreml_model.short_description = 'Advanced emotion detection model'
        coreml_model.input_description['input'] = 'Grayscale face image 48x48'
        coreml_model.output_description['output'] = 'Emotion probabilities'
        
        # Save
        coreml_model.save(output_path)
        
        logger.info(f"Core ML model saved to {output_path}")
        return output_path


class InferenceEngine:
    """Main inference engine with automatic backend selection"""
    
    def __init__(self, model_path: str, backend: str = 'auto',
                 batch_mode: bool = False, num_threads: int = 4):
        self.backend_name = backend
        self.batch_mode = batch_mode
        
        # Select backend
        if backend == 'auto':
            self.backend = self._auto_select_backend(model_path, num_threads)
        elif backend == 'tflite':
            self.backend = TensorFlowLiteBackend(model_path, num_threads)
        elif backend == 'onnx':
            self.backend = ONNXRuntimeBackend(model_path)
        elif backend == 'coreml':
            self.backend = CoreMLBackend(model_path)
        else:
            raise ValueError(f"Unknown backend: {backend}")
        
        # Create batched engine if needed
        if batch_mode:
            self.batch_engine = BatchedInferenceEngine(self.backend)
        
        # Warmup
        logger.info(f"Warming up {self.backend.get_backend_name()}...")
        self.backend.warmup()
        
    def _auto_select_backend(self, model_path: str, num_threads: int) -> InferenceBackend:
        """Automatically select best backend based on platform"""
        import platform
        
        system = platform.system()
        
        if system == 'Darwin':  # macOS
            # Try Core ML first for Mac
            try:
                if model_path.endswith('.mlmodel'):
                    return CoreMLBackend(model_path)
            except:
                pass
        
        # Try TFLite as fallback
        if model_path.endswith('.tflite'):
            return TensorFlowLiteBackend(model_path, num_threads)
        
        # Try ONNX
        if model_path.endswith('.onnx'):
            return ONNXRuntimeBackend(model_path)
        
        # Default to TFLite
        return TensorFlowLiteBackend(model_path, num_threads)
    
    def predict(self, image: np.ndarray) -> InferenceResult:
        """Run inference on single image"""
        return self.backend.predict(image)
    
    def predict_batch(self, images: List[np.ndarray]) -> List[InferenceResult]:
        """Run inference on batch of images"""
        if self.batch_mode:
            # Use batched engine
            results = []
            item_ids = [str(i) for i in range(len(images))]
            
            # Submit all images
            for item_id, image in zip(item_ids, images):
                self.batch_engine.predict_async(image, item_id)
            
            # Collect results
            for item_id in item_ids:
                result = self.batch_engine.get_result(item_id)
                if result:
                    results.append(result)
                    
            return results
        else:
            # Process sequentially
            return [self.backend.predict(image) for image in images]
    
    def benchmark(self, num_runs: int = 100, image_size: Tuple[int, int] = (48, 48)) -> Dict:
        """Benchmark inference performance"""
        # Create dummy image
        dummy_image = np.random.randint(0, 255, (*image_size, 1), dtype=np.uint8)
        
        # Warmup
        for _ in range(10):
            self.backend.predict(dummy_image)
        
        # Benchmark
        times = []
        for _ in range(num_runs):
            start = time.time()
            _ = self.backend.predict(dummy_image)
            times.append((time.time() - start) * 1000)
        
        times = np.array(times)
        
        return {
            'backend': self.backend.get_backend_name(),
            'mean_ms': float(np.mean(times)),
            'std_ms': float(np.std(times)),
            'min_ms': float(np.min(times)),
            'max_ms': float(np.max(times)),
            'p50_ms': float(np.percentile(times, 50)),
            'p95_ms': float(np.percentile(times, 95)),
            'p99_ms': float(np.percentile(times, 99)),
            'fps': float(1000 / np.mean(times))
        }


def main():
    """Demo and benchmarking"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Inference Engine Demo')
    parser.add_argument('model', help='Path to model file')
    parser.add_argument('--backend', default='auto', 
                       choices=['auto', 'tflite', 'onnx', 'coreml'])
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--convert', choices=['tflite', 'onnx', 'coreml'])
    parser.add_argument('--output', help='Output path for conversion')
    
    args = parser.parse_args()
    
    if args.convert:
        # Convert model
        if args.convert == 'tflite':
            ModelOptimizer.convert_to_tflite(args.model, args.output or 'model.tflite')
        elif args.convert == 'onnx':
            ModelOptimizer.convert_to_onnx(args.model, args.output or 'model.onnx')
        elif args.convert == 'coreml':
            ModelOptimizer.convert_to_coreml(args.model, args.output or 'model.mlmodel')
    else:
        # Create engine
        engine = InferenceEngine(args.model, backend=args.backend)
        
        if args.benchmark:
            # Run benchmark
            results = engine.benchmark()
            print("\nBenchmark Results:")
            print("-" * 40)
            for key, value in results.items():
                print(f"{key:15}: {value:10.2f}" if isinstance(value, float) else f"{key:15}: {value}")
        else:
            # Demo inference
            dummy_image = np.random.randint(0, 255, (48, 48), dtype=np.uint8)
            result = engine.predict(dummy_image)
            print(f"\nInference Result:")
            print(f"Emotion: {result.emotion_name}")
            print(f"Confidence: {result.confidence:.2%}")
            print(f"Time: {result.inference_time_ms:.2f}ms")


if __name__ == "__main__":
    main()