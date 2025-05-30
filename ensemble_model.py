#!/usr/bin/env python3
"""
Ensemble Model for Emotion Detection
Combines multiple models for improved accuracy
"""

import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Average, Maximum, Concatenate, Dense
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB1, ResNet50V2, MobileNetV2
import yaml
import os

# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

class EnsembleEmotionModel:
    def __init__(self, ensemble_method='average'):
        """
        Initialize ensemble model
        
        Args:
            ensemble_method: 'average', 'weighted', 'voting', 'stacking'
        """
        self.ensemble_method = ensemble_method
        self.models = []
        self.model_weights = []
        self.ensemble_model = None
        
    def create_base_model(self, architecture, input_shape=(48, 48, 3), num_classes=7):
        """Create a base model with specified architecture"""
        
        # Select base model
        if architecture == 'EfficientNetB0':
            base = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
        elif architecture == 'EfficientNetB1':
            base = EfficientNetB1(weights='imagenet', include_top=False, input_shape=input_shape)
        elif architecture == 'ResNet50V2':
            base = ResNet50V2(weights='imagenet', include_top=False, input_shape=input_shape)
        elif architecture == 'MobileNetV2':
            base = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
        
        # Freeze base initially
        base.trainable = False
        
        # Build model
        inputs = Input(shape=input_shape)
        x = base(inputs, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        outputs = tf.keras.layers.Dense(num_classes, activation='softmax', dtype='float32')(x)
        
        model = Model(inputs, outputs, name=f"{architecture}_emotion_model")
        return model, base
    
    def add_model(self, model_path=None, model=None, weight=1.0):
        """Add a model to the ensemble"""
        if model_path:
            model = load_model(model_path)
        elif model is None:
            raise ValueError("Either model_path or model must be provided")
        
        self.models.append(model)
        self.model_weights.append(weight)
        print(f"Added model: {model.name if hasattr(model, 'name') else 'Unknown'}")
    
    def create_average_ensemble(self):
        """Create an ensemble that averages predictions"""
        if len(self.models) < 2:
            raise ValueError("Need at least 2 models for ensemble")
        
        # Get input shape from first model
        input_shape = self.models[0].input_shape[1:]
        inputs = Input(shape=input_shape)
        
        # Get predictions from all models
        outputs = []
        for model in self.models:
            # Make model layers non-trainable for ensemble
            for layer in model.layers:
                layer.trainable = False
            outputs.append(model(inputs))
        
        # Average predictions
        if self.ensemble_method == 'average':
            ensemble_output = Average()(outputs)
        elif self.ensemble_method == 'maximum':
            ensemble_output = Maximum()(outputs)
        else:
            raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")
        
        self.ensemble_model = Model(inputs, ensemble_output, name="ensemble_emotion_model")
        return self.ensemble_model
    
    def create_weighted_ensemble(self):
        """Create an ensemble with weighted predictions"""
        if len(self.models) < 2:
            raise ValueError("Need at least 2 models for ensemble")
        
        input_shape = self.models[0].input_shape[1:]
        inputs = Input(shape=input_shape)
        
        # Normalize weights
        total_weight = sum(self.model_weights)
        normalized_weights = [w / total_weight for w in self.model_weights]
        
        # Weighted sum of predictions
        weighted_outputs = []
        for model, weight in zip(self.models, normalized_weights):
            for layer in model.layers:
                layer.trainable = False
            output = model(inputs)
            weighted_output = tf.keras.layers.Lambda(lambda x: x * weight)(output)
            weighted_outputs.append(weighted_output)
        
        ensemble_output = tf.keras.layers.Add()(weighted_outputs)
        
        self.ensemble_model = Model(inputs, ensemble_output, name="weighted_ensemble_model")
        return self.ensemble_model
    
    def create_stacking_ensemble(self, meta_learner_layers=[64, 32]):
        """Create a stacking ensemble with a meta-learner"""
        if len(self.models) < 2:
            raise ValueError("Need at least 2 models for ensemble")
        
        input_shape = self.models[0].input_shape[1:]
        inputs = Input(shape=input_shape)
        
        # Get predictions from all models
        model_outputs = []
        for model in self.models:
            for layer in model.layers:
                layer.trainable = False
            model_outputs.append(model(inputs))
        
        # Concatenate all predictions
        concatenated = Concatenate()(model_outputs)
        
        # Meta-learner
        x = concatenated
        for units in meta_learner_layers:
            x = Dense(units, activation='relu')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Dropout(0.3)(x)
        
        # Final prediction
        num_classes = self.models[0].output_shape[-1]
        ensemble_output = Dense(num_classes, activation='softmax', dtype='float32')(x)
        
        self.ensemble_model = Model(inputs, ensemble_output, name="stacking_ensemble_model")
        return self.ensemble_model
    
    def build_ensemble(self):
        """Build the ensemble based on the specified method"""
        if self.ensemble_method in ['average', 'maximum']:
            return self.create_average_ensemble()
        elif self.ensemble_method == 'weighted':
            return self.create_weighted_ensemble()
        elif self.ensemble_method == 'stacking':
            return self.create_stacking_ensemble()
        else:
            raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")
    
    def predict(self, x, use_tta=False):
        """
        Make predictions with the ensemble
        
        Args:
            x: Input data
            use_tta: Use test-time augmentation
        """
        if self.ensemble_model is None:
            raise ValueError("Ensemble model not built. Call build_ensemble() first.")
        
        if use_tta:
            # Test-time augmentation
            predictions = []
            
            # Original
            predictions.append(self.ensemble_model.predict(x))
            
            # Horizontal flip
            flipped = tf.image.flip_left_right(x)
            predictions.append(self.ensemble_model.predict(flipped))
            
            # Small rotations
            for angle in [-10, 10]:
                rotated = tf.keras.preprocessing.image.apply_affine_transform(
                    x.numpy(), theta=angle, fill_mode='nearest'
                )
                predictions.append(self.ensemble_model.predict(rotated))
            
            # Average predictions
            return np.mean(predictions, axis=0)
        else:
            return self.ensemble_model.predict(x)
    
    def save_ensemble(self, filepath):
        """Save the ensemble model"""
        if self.ensemble_model is None:
            raise ValueError("No ensemble model to save")
        self.ensemble_model.save(filepath)
        print(f"Ensemble model saved to: {filepath}")
    
    def evaluate_models(self, test_data):
        """Evaluate individual models and ensemble"""
        results = {}
        
        # Evaluate individual models
        for i, model in enumerate(self.models):
            loss, accuracy = model.evaluate(test_data)
            model_name = model.name if hasattr(model, 'name') else f"Model_{i}"
            results[model_name] = {'loss': loss, 'accuracy': accuracy}
            print(f"{model_name}: Loss={loss:.4f}, Accuracy={accuracy:.4f}")
        
        # Evaluate ensemble
        if self.ensemble_model:
            # Compile ensemble for evaluation
            self.ensemble_model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            loss, accuracy = self.ensemble_model.evaluate(test_data)
            results['Ensemble'] = {'loss': loss, 'accuracy': accuracy}
            print(f"Ensemble: Loss={loss:.4f}, Accuracy={accuracy:.4f}")
        
        return results


def train_diverse_models(train_data, val_data, num_classes=7):
    """Train multiple diverse models for ensemble"""
    models = []
    
    # Architecture configurations
    architectures = [
        ('EfficientNetB0', 30, 50),
        ('MobileNetV2', 25, 40),
        ('ResNet50V2', 25, 40),
    ]
    
    for arch_name, initial_epochs, fine_tune_epochs in architectures:
        print(f"\n{'='*50}")
        print(f"Training {arch_name} model")
        print('='*50)
        
        # Create model
        ensemble_builder = EnsembleEmotionModel()
        model, base_model = ensemble_builder.create_base_model(
            arch_name, 
            input_shape=(48, 48, 3),
            num_classes=num_classes
        )
        
        # Compile
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Phase 1: Train with frozen base
        print(f"\nPhase 1: Training {arch_name} with frozen base")
        history1 = model.fit(
            train_data,
            epochs=initial_epochs,
            validation_data=val_data,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5)
            ]
        )
        
        # Phase 2: Fine-tuning
        print(f"\nPhase 2: Fine-tuning {arch_name}")
        base_model.trainable = True
        
        # Re-compile with lower learning rate
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        history2 = model.fit(
            train_data,
            epochs=initial_epochs + fine_tune_epochs,
            initial_epoch=len(history1.history['loss']),
            validation_data=val_data,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                tf.keras.callbacks.ModelCheckpoint(
                    f'models/{arch_name}_emotion_model.keras',
                    save_best_only=True,
                    monitor='val_accuracy'
                )
            ]
        )
        
        models.append(model)
        print(f"\n{arch_name} training completed!")
    
    return models


def create_and_evaluate_ensemble(models, test_data, ensemble_method='weighted'):
    """Create and evaluate different ensemble methods"""
    print("\n" + "="*50)
    print(f"Creating {ensemble_method} ensemble")
    print("="*50)
    
    # Create ensemble
    ensemble = EnsembleEmotionModel(ensemble_method=ensemble_method)
    
    # Add models with weights based on individual accuracy
    for model in models:
        # Quick evaluation to get accuracy for weighting
        _, acc = model.evaluate(test_data, verbose=0)
        ensemble.add_model(model=model, weight=acc)
    
    # Build ensemble
    ensemble_model = ensemble.build_ensemble()
    
    # Evaluate
    results = ensemble.evaluate_models(test_data)
    
    # Save best ensemble
    ensemble.save_ensemble(f'models/ensemble_{ensemble_method}_model.keras')
    
    return ensemble, results


if __name__ == "__main__":
    print("Ensemble Model Builder")
    print("This script trains multiple models and combines them for better accuracy")
    print("\nNote: This is a demonstration. In practice, you would:")
    print("1. Load your preprocessed data")
    print("2. Train diverse models")
    print("3. Create and evaluate ensembles")
    print("4. Use the best ensemble for predictions")