#!/usr/bin/env python3
"""
Model Optimization Script
Provides various optimization techniques for the emotion detection model
"""

import tensorflow as tf
import numpy as np
import os
import time
import argparse
from pathlib import Path
import tensorflow_model_optimization as tfmot

class ModelOptimizer:
    def __init__(self, model_path):
        """Initialize with model path"""
        self.model_path = model_path
        self.model = tf.keras.models.load_model(model_path)
        self.optimized_models = {}
        
    def quantize_dynamic(self, save_path='models/model_dynamic_quant.tflite'):
        """Dynamic range quantization (simplest, smallest size reduction)"""
        print("\n=== Dynamic Range Quantization ===")
        
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        tflite_model = converter.convert()
        
        # Save model
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'wb') as f:
            f.write(tflite_model)
        
        # Compare sizes
        original_size = os.path.getsize(self.model_path) / (1024**2)
        optimized_size = os.path.getsize(save_path) / (1024**2)
        reduction = (1 - optimized_size/original_size) * 100
        
        print(f"Original size: {original_size:.2f} MB")
        print(f"Optimized size: {optimized_size:.2f} MB")
        print(f"Size reduction: {reduction:.1f}%")
        
        self.optimized_models['dynamic_quant'] = save_path
        return save_path
    
    def quantize_integer(self, representative_data_gen=None, save_path='models/model_int8_quant.tflite'):
        """Full integer quantization (best for edge devices)"""
        print("\n=== Full Integer Quantization ===")
        
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Configure for full integer quantization
        converter.target_spec.supported_types = [tf.int8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
        
        # Provide representative dataset
        if representative_data_gen is None:
            def representative_dataset():
                for _ in range(100):
                    data = np.random.rand(1, 48, 48, 3).astype(np.float32)
                    yield [data]
            converter.representative_dataset = representative_dataset
        else:
            converter.representative_dataset = representative_data_gen
        
        try:
            tflite_model = converter.convert()
            
            # Save model
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, 'wb') as f:
                f.write(tflite_model)
            
            # Compare sizes
            original_size = os.path.getsize(self.model_path) / (1024**2)
            optimized_size = os.path.getsize(save_path) / (1024**2)
            reduction = (1 - optimized_size/original_size) * 100
            
            print(f"Original size: {original_size:.2f} MB")
            print(f"Optimized size: {optimized_size:.2f} MB")
            print(f"Size reduction: {reduction:.1f}%")
            
            self.optimized_models['int8_quant'] = save_path
            return save_path
            
        except Exception as e:
            print(f"Integer quantization failed: {e}")
            return None
    
    def quantize_float16(self, save_path='models/model_float16_quant.tflite'):
        """Float16 quantization (good balance of size and accuracy)"""
        print("\n=== Float16 Quantization ===")
        
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        
        tflite_model = converter.convert()
        
        # Save model
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'wb') as f:
            f.write(tflite_model)
        
        # Compare sizes
        original_size = os.path.getsize(self.model_path) / (1024**2)
        optimized_size = os.path.getsize(save_path) / (1024**2)
        reduction = (1 - optimized_size/original_size) * 100
        
        print(f"Original size: {original_size:.2f} MB")
        print(f"Optimized size: {optimized_size:.2f} MB")
        print(f"Size reduction: {reduction:.1f}%")
        
        self.optimized_models['float16_quant'] = save_path
        return save_path
    
    def prune_model(self, target_sparsity=0.5, save_path='models/model_pruned.h5'):
        """Magnitude-based weight pruning"""
        print(f"\n=== Weight Pruning (Sparsity: {target_sparsity}) ===")
        
        # Define pruning parameters
        pruning_params = {
            'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
                initial_sparsity=0.0,
                final_sparsity=target_sparsity,
                begin_step=0,
                end_step=1000
            )
        }
        
        # Apply pruning to the model
        pruned_model = tfmot.sparsity.keras.prune_low_magnitude(
            self.model,
            **pruning_params
        )
        
        # Compile pruned model
        pruned_model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Save pruned model
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        pruned_model.save(save_path)
        
        # Strip pruning wrappers and save final model
        final_model = tfmot.sparsity.keras.strip_pruning(pruned_model)
        final_path = save_path.replace('.h5', '_stripped.h5')
        final_model.save(final_path)
        
        print(f"Pruned model saved to: {save_path}")
        print(f"Stripped model saved to: {final_path}")
        
        self.optimized_models['pruned'] = final_path
        return final_path
    
    def create_tflite_with_metadata(self, tflite_path, save_path='models/model_with_metadata.tflite'):
        """Add metadata to TFLite model for easier deployment"""
        print("\n=== Adding Metadata to TFLite Model ===")
        
        from tflite_support import metadata as _metadata
        from tflite_support import metadata_writers as _metadata_writers
        
        # Create metadata
        writer = _metadata_writers.ImageClassifierWriter.create(
            tflite_path,
            ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise'],
            'emotion_labels.txt'
        )
        
        writer.save(save_path)
        print(f"Model with metadata saved to: {save_path}")
        
        return save_path
    
    def optimize_for_edge_tpu(self, save_path='models/model_edgetpu.tflite'):
        """Optimize model for Google Edge TPU"""
        print("\n=== Edge TPU Optimization ===")
        
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.int8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
        
        # Representative dataset for quantization
        def representative_dataset():
            for _ in range(100):
                data = np.random.rand(1, 48, 48, 3).astype(np.float32)
                yield [data]
        
        converter.representative_dataset = representative_dataset
        
        # Edge TPU specific settings
        converter.allow_custom_ops = False
        converter._experimental_new_converter = True
        
        try:
            tflite_model = converter.convert()
            
            # Save model
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, 'wb') as f:
                f.write(tflite_model)
            
            print(f"Edge TPU compatible model saved to: {save_path}")
            print("Note: Run 'edgetpu_compiler {save_path}' to compile for Edge TPU")
            
            self.optimized_models['edgetpu'] = save_path
            return save_path
            
        except Exception as e:
            print(f"Edge TPU optimization failed: {e}")
            return None
    
    def benchmark_optimized_models(self, test_images=None):
        """Benchmark all optimized models"""
        print("\n=== Benchmarking Optimized Models ===")
        
        if test_images is None:
            # Generate random test images
            test_images = np.random.rand(10, 48, 48, 3).astype(np.float32)
        
        results = {}
        
        # Benchmark original model
        print("\nOriginal Keras Model:")
        start = time.time()
        for img in test_images:
            _ = self.model.predict(np.expand_dims(img, axis=0), verbose=0)
        keras_time = time.time() - start
        print(f"  Total time: {keras_time:.3f}s")
        print(f"  Per image: {keras_time/len(test_images)*1000:.1f}ms")
        results['original'] = keras_time/len(test_images)
        
        # Benchmark TFLite models
        for name, model_path in self.optimized_models.items():
            if model_path and model_path.endswith('.tflite') and os.path.exists(model_path):
                print(f"\n{name}:")
                
                # Load TFLite model
                interpreter = tf.lite.Interpreter(model_path=model_path)
                interpreter.allocate_tensors()
                
                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()
                
                # Benchmark
                start = time.time()
                for img in test_images:
                    # Prepare input
                    if input_details[0]['dtype'] == np.uint8:
                        input_data = ((img * 127.5) + 127.5).astype(np.uint8)
                    else:
                        input_data = img.astype(input_details[0]['dtype'])
                    
                    input_data = np.expand_dims(input_data, axis=0)
                    
                    # Run inference
                    interpreter.set_tensor(input_details[0]['index'], input_data)
                    interpreter.invoke()
                    output_data = interpreter.get_tensor(output_details[0]['index'])
                
                tflite_time = time.time() - start
                print(f"  Total time: {tflite_time:.3f}s")
                print(f"  Per image: {tflite_time/len(test_images)*1000:.1f}ms")
                print(f"  Speedup: {keras_time/tflite_time:.2f}x")
                results[name] = tflite_time/len(test_images)
        
        return results
    
    def create_optimized_serving_model(self, save_path='models/serving_model'):
        """Create optimized model for TensorFlow Serving"""
        print("\n=== Creating Optimized Serving Model ===")
        
        # Save in SavedModel format with optimization
        tf.saved_model.save(
            self.model,
            save_path,
            signatures={
                'serving_default': tf.function(
                    lambda x: self.model(x, training=False)
                ).get_concrete_function(
                    tf.TensorSpec(
                        shape=[None, 48, 48, 3],
                        dtype=tf.float32
                    )
                )
            }
        )
        
        print(f"Serving model saved to: {save_path}")
        
        # Optionally convert to TensorFlow Lite for serving
        converter = tf.lite.TFLiteConverter.from_saved_model(save_path)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        
        tflite_path = f"{save_path}_lite.tflite"
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"TFLite serving model saved to: {tflite_path}")
        
        return save_path


def main():
    parser = argparse.ArgumentParser(description='Optimize emotion detection model')
    parser.add_argument('model_path', help='Path to the Keras model')
    parser.add_argument('--all', action='store_true', help='Apply all optimizations')
    parser.add_argument('--dynamic', action='store_true', help='Dynamic range quantization')
    parser.add_argument('--int8', action='store_true', help='Full integer quantization')
    parser.add_argument('--float16', action='store_true', help='Float16 quantization')
    parser.add_argument('--prune', action='store_true', help='Weight pruning')
    parser.add_argument('--edgetpu', action='store_true', help='Edge TPU optimization')
    parser.add_argument('--serving', action='store_true', help='Create serving model')
    parser.add_argument('--benchmark', action='store_true', help='Benchmark optimized models')
    
    args = parser.parse_args()
    
    # Create optimizer
    optimizer = ModelOptimizer(args.model_path)
    
    # Apply optimizations
    if args.all or args.dynamic:
        optimizer.quantize_dynamic()
    
    if args.all or args.int8:
        optimizer.quantize_integer()
    
    if args.all or args.float16:
        optimizer.quantize_float16()
    
    if args.all or args.prune:
        optimizer.prune_model()
    
    if args.all or args.edgetpu:
        optimizer.optimize_for_edge_tpu()
    
    if args.all or args.serving:
        optimizer.create_optimized_serving_model()
    
    # Benchmark if requested
    if args.benchmark:
        optimizer.benchmark_optimized_models()
    
    print("\n=== Optimization Complete ===")
    print("Optimized models saved in 'models/' directory")


if __name__ == "__main__":
    main()