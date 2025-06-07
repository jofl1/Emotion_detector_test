#!/usr/bin/env python3
"""
Advanced Emotion Detection Model Training
State-of-the-art techniques for maximum accuracy
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from typing import Tuple, List, Dict, Optional
import matplotlib.pyplot as plt
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import wandb
from dataclasses import dataclass
import yaml
import logging
from pathlib import Path
import albumentations as A
from albumentations.tensorflow import ToTensorV2

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enable mixed precision training
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

# XLA compilation for faster training
tf.config.optimizer.set_jit(True)


@dataclass
class TrainingConfig:
    """Training configuration with best practices"""
    # Data
    img_size: int = 48
    batch_size: int = 64
    num_classes: int = 7
    
    # Model
    architecture: str = "efficientnet_v2"
    dropout_rate: float = 0.3
    l2_reg: float = 1e-4
    
    # Training
    epochs: int = 100
    initial_lr: float = 1e-3
    min_lr: float = 1e-7
    warmup_epochs: int = 5
    
    # Augmentation
    augmentation_strength: float = 0.8
    mixup_alpha: float = 0.2
    cutmix_alpha: float = 1.0
    
    # Advanced
    label_smoothing: float = 0.1
    gradient_clip_norm: float = 1.0
    ema_decay: float = 0.999
    
    # Paths
    train_dir: str = "~/Python/archive/train"
    test_dir: str = "~/Python/archive/test"
    model_dir: str = "./models"
    
    @classmethod
    def from_yaml(cls, path: str) -> 'TrainingConfig':
        """Load config from YAML file"""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)


class AdvancedAugmentation:
    """Advanced augmentation pipeline using Albumentations"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.transform = self._build_transform()
        
    def _build_transform(self) -> A.Compose:
        """Build augmentation pipeline"""
        return A.Compose([
            # Geometric transforms
            A.RandomRotate90(p=0.5),
            A.Flip(p=0.5),
            A.Transpose(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.1,
                rotate_limit=15,
                p=0.8
            ),
            
            # Pixel-level transforms
            A.OneOf([
                A.MotionBlur(p=1.0),
                A.MedianBlur(blur_limit=3, p=1.0),
                A.GaussianBlur(p=1.0),
            ], p=0.3),
            
            A.OneOf([
                A.OpticalDistortion(p=1.0),
                A.GridDistortion(p=1.0),
                A.PiecewiseAffine(p=1.0),
            ], p=0.3),
            
            # Noise and artifacts
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
                A.ISONoise(p=1.0),
                A.MultiplicativeNoise(p=1.0),
            ], p=0.3),
            
            # Brightness and contrast
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.8
            ),
            
            # Advanced transforms
            A.CoarseDropout(
                max_holes=1,
                max_height=16,
                max_width=16,
                min_holes=1,
                min_height=8,
                min_width=8,
                p=0.5
            ),
            
            # Normalize
            A.Normalize(mean=0.5, std=0.5, always_apply=True),
            ToTensorV2()
        ])
    
    def __call__(self, image: np.ndarray) -> tf.Tensor:
        """Apply augmentation"""
        augmented = self.transform(image=image)
        return augmented['image']


class MixupCutmixAugmentation:
    """Mixup and CutMix augmentation for better generalization"""
    
    def __init__(self, config: TrainingConfig):
        self.mixup_alpha = config.mixup_alpha
        self.cutmix_alpha = config.cutmix_alpha
        
    def mixup(self, images: tf.Tensor, labels: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Apply Mixup augmentation"""
        batch_size = tf.shape(images)[0]
        
        # Sample lambda from Beta distribution
        lam = tf.random.uniform([], 0, 1)
        lam = tf.maximum(lam, 1 - lam)
        
        # Shuffle indices
        indices = tf.random.shuffle(tf.range(batch_size))
        
        # Mix images and labels
        mixed_images = lam * images + (1 - lam) * tf.gather(images, indices)
        mixed_labels = lam * labels + (1 - lam) * tf.gather(labels, indices)
        
        return mixed_images, mixed_labels
    
    def cutmix(self, images: tf.Tensor, labels: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Apply CutMix augmentation"""
        batch_size = tf.shape(images)[0]
        img_h, img_w = tf.shape(images)[1], tf.shape(images)[2]
        
        # Sample lambda from Beta distribution
        lam = tf.random.uniform([], 0, 1)
        
        # Calculate cut size
        cut_ratio = tf.sqrt(1 - lam)
        cut_h = tf.cast(tf.cast(img_h, tf.float32) * cut_ratio, tf.int32)
        cut_w = tf.cast(tf.cast(img_w, tf.float32) * cut_ratio, tf.int32)
        
        # Sample cut center
        cx = tf.random.uniform([], cut_w // 2, img_w - cut_w // 2, dtype=tf.int32)
        cy = tf.random.uniform([], cut_h // 2, img_h - cut_h // 2, dtype=tf.int32)
        
        # Create mask
        mask = tf.ones((batch_size, img_h, img_w, 1))
        pad_left = cx - cut_w // 2
        pad_top = cy - cut_h // 2
        pad_right = img_w - (cx + cut_w // 2)
        pad_bottom = img_h - (cy + cut_h // 2)
        
        mask = tf.pad(
            tf.zeros((batch_size, cut_h, cut_w, 1)),
            [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]],
            constant_values=1
        )
        
        # Shuffle indices
        indices = tf.random.shuffle(tf.range(batch_size))
        
        # Apply CutMix
        mixed_images = mask * images + (1 - mask) * tf.gather(images, indices)
        mixed_labels = lam * labels + (1 - lam) * tf.gather(labels, indices)
        
        return mixed_images, mixed_labels


class EfficientNetV2EmotionModel:
    """State-of-the-art EfficientNetV2 model for emotion detection"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model = self._build_model()
        
    def _build_model(self) -> Model:
        """Build EfficientNetV2 model with custom head"""
        # Input layer
        inputs = layers.Input(shape=(self.config.img_size, self.config.img_size, 1))
        
        # Convert grayscale to RGB for EfficientNet
        x = layers.Lambda(lambda x: tf.repeat(x, 3, axis=-1))(inputs)
        
        # Base model
        base_model = tf.keras.applications.EfficientNetV2B0(
            include_top=False,
            weights='imagenet',
            input_tensor=x,
            pooling=None
        )
        
        # Freeze base model initially
        base_model.trainable = False
        
        # Feature extraction
        features = base_model(x, training=False)
        
        # Custom head with attention mechanism
        x = layers.GlobalAveragePooling2D()(features)
        
        # Channel attention
        attention = layers.Dense(x.shape[-1] // 16, activation='relu')(x)
        attention = layers.Dense(x.shape[-1], activation='sigmoid')(attention)
        x = layers.Multiply()([x, attention])
        
        # Classification layers
        x = layers.Dense(
            512,
            activation='relu',
            kernel_regularizer=l2(self.config.l2_reg)
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(self.config.dropout_rate)(x)
        
        x = layers.Dense(
            256,
            activation='relu',
            kernel_regularizer=l2(self.config.l2_reg)
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(self.config.dropout_rate)(x)
        
        # Output layer with mixed precision
        outputs = layers.Dense(
            self.config.num_classes,
            activation='softmax',
            dtype='float32'  # Keep output in float32 for stability
        )(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        self.base_model = base_model
        
        return model
    
    def unfreeze_top_layers(self, num_layers: int = 20):
        """Unfreeze top layers for fine-tuning"""
        self.base_model.trainable = True
        
        # Freeze all layers except top N
        for layer in self.base_model.layers[:-num_layers]:
            layer.trainable = False
            
        logger.info(f"Unfroze top {num_layers} layers for fine-tuning")


class CosineLearningRateSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Cosine annealing with warm restarts"""
    
    def __init__(self, initial_lr: float, warmup_steps: int, total_steps: int,
                 min_lr: float = 1e-7, cycles: int = 1):
        super().__init__()
        self.initial_lr = initial_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.cycles = cycles
        
    def __call__(self, step):
        if step < self.warmup_steps:
            # Linear warmup
            return self.initial_lr * (step / self.warmup_steps)
        else:
            # Cosine annealing
            progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            return self.min_lr + (self.initial_lr - self.min_lr) * \
                   0.5 * (1 + tf.cos(np.pi * progress * self.cycles))


class ExponentialMovingAverage(Callback):
    """EMA callback for better generalization"""
    
    def __init__(self, decay: float = 0.999):
        super().__init__()
        self.decay = decay
        self.ema_weights = None
        
    def on_train_begin(self, logs=None):
        """Initialize EMA weights"""
        self.ema_weights = [tf.Variable(w) for w in self.model.get_weights()]
        
    def on_train_batch_end(self, batch, logs=None):
        """Update EMA weights"""
        for ema_w, w in zip(self.ema_weights, self.model.get_weights()):
            ema_w.assign(self.decay * ema_w + (1 - self.decay) * w)
            
    def on_epoch_end(self, epoch, logs=None):
        """Swap weights for validation"""
        # Save current weights
        current_weights = self.model.get_weights()
        
        # Set EMA weights for validation
        self.model.set_weights([w.numpy() for w in self.ema_weights])
        
        # Evaluate with EMA weights
        val_loss, val_acc = self.model.evaluate(
            self.validation_data[0],
            self.validation_data[1],
            verbose=0
        )
        
        # Restore original weights
        self.model.set_weights(current_weights)
        
        # Log EMA metrics
        logs['ema_val_loss'] = val_loss
        logs['ema_val_accuracy'] = val_acc
        
        logger.info(f"EMA - Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")


class AdvancedTrainer:
    """Advanced training pipeline with all optimizations"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model_builder = EfficientNetV2EmotionModel(config)
        self.augmentation = MixupCutmixAugmentation(config)
        
    def prepare_data(self) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """Prepare optimized data pipelines"""
        # Expand paths
        train_dir = os.path.expanduser(self.config.train_dir)
        test_dir = os.path.expanduser(self.config.test_dir)
        
        # Basic data generators for loading
        datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=0.2
        )
        
        # Load data
        train_gen = datagen.flow_from_directory(
            train_dir,
            target_size=(self.config.img_size, self.config.img_size),
            color_mode='grayscale',
            batch_size=self.config.batch_size,
            class_mode='categorical',
            subset='training',
            shuffle=True
        )
        
        val_gen = datagen.flow_from_directory(
            train_dir,
            target_size=(self.config.img_size, self.config.img_size),
            color_mode='grayscale',
            batch_size=self.config.batch_size,
            class_mode='categorical',
            subset='validation',
            shuffle=False
        )
        
        test_gen = datagen.flow_from_directory(
            test_dir,
            target_size=(self.config.img_size, self.config.img_size),
            color_mode='grayscale',
            batch_size=self.config.batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        # Calculate class weights
        self.class_weights = self._calculate_class_weights(train_gen)
        
        # Convert to tf.data for performance
        train_dataset = self._generator_to_dataset(train_gen, augment=True)
        val_dataset = self._generator_to_dataset(val_gen, augment=False)
        test_dataset = self._generator_to_dataset(test_gen, augment=False)
        
        return train_dataset, val_dataset, test_dataset
    
    def _calculate_class_weights(self, generator) -> Dict[int, float]:
        """Calculate balanced class weights"""
        classes = generator.classes
        weights = class_weight.compute_class_weight(
            class_weight='balanced',
            classes=np.unique(classes),
            y=classes
        )
        return dict(enumerate(weights))
    
    def _generator_to_dataset(self, generator, augment: bool) -> tf.data.Dataset:
        """Convert Keras generator to tf.data.Dataset"""
        def gen():
            while True:
                x, y = next(generator)
                yield x, y
                
        dataset = tf.data.Dataset.from_generator(
            gen,
            output_signature=(
                tf.TensorSpec(shape=(None, self.config.img_size, self.config.img_size, 1),
                             dtype=tf.float32),
                tf.TensorSpec(shape=(None, self.config.num_classes), dtype=tf.float32)
            )
        )
        
        # Apply augmentation if needed
        if augment:
            dataset = dataset.map(
                lambda x, y: tf.py_function(
                    self._augment_batch,
                    [x, y],
                    [tf.float32, tf.float32]
                ),
                num_parallel_calls=tf.data.AUTOTUNE
            )
        
        # Optimize pipeline
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def _augment_batch(self, images: tf.Tensor, labels: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Apply advanced augmentations to batch"""
        # Random choice between Mixup and CutMix
        if tf.random.uniform([]) > 0.5:
            images, labels = self.augmentation.mixup(images, labels)
        else:
            images, labels = self.augmentation.cutmix(images, labels)
            
        return images, labels
    
    def compile_model(self):
        """Compile model with advanced optimizations"""
        # Calculate steps
        steps_per_epoch = 500  # Approximate
        total_steps = steps_per_epoch * self.config.epochs
        warmup_steps = steps_per_epoch * self.config.warmup_epochs
        
        # Learning rate schedule
        lr_schedule = CosineLearningRateSchedule(
            self.config.initial_lr,
            warmup_steps,
            total_steps,
            self.config.min_lr
        )
        
        # Optimizer with gradient clipping
        optimizer = Adam(
            learning_rate=lr_schedule,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7,
            clipnorm=self.config.gradient_clip_norm
        )
        
        # Compile
        self.model_builder.model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.CategoricalCrossentropy(
                label_smoothing=self.config.label_smoothing
            ),
            metrics=[
                'accuracy',
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.AUC(name='auc')
            ]
        )
        
    def create_callbacks(self) -> List[Callback]:
        """Create advanced callbacks"""
        callbacks = [
            # Model checkpointing
            tf.keras.callbacks.ModelCheckpoint(
                os.path.join(self.config.model_dir, 'best_model_advanced.keras'),
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            
            # Early stopping with patience
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=20,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Reduce LR on plateau
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=self.config.min_lr,
                verbose=1
            ),
            
            # EMA
            ExponentialMovingAverage(decay=self.config.ema_decay),
            
            # TensorBoard
            tf.keras.callbacks.TensorBoard(
                log_dir='./logs',
                histogram_freq=1,
                profile_batch='500,520'
            ),
        ]
        
        # Add Weights & Biases if available
        try:
            import wandb
            from wandb.keras import WandbCallback
            callbacks.append(WandbCallback())
        except ImportError:
            logger.warning("Weights & Biases not available")
            
        return callbacks
    
    def train(self):
        """Execute training pipeline"""
        # Create model directory
        Path(self.config.model_dir).mkdir(parents=True, exist_ok=True)
        
        # Prepare data
        logger.info("Preparing datasets...")
        train_dataset, val_dataset, test_dataset = self.prepare_data()
        
        # Compile model
        logger.info("Compiling model...")
        self.compile_model()
        
        # Create callbacks
        callbacks = self.create_callbacks()
        
        # Phase 1: Train with frozen base
        logger.info("Phase 1: Training with frozen base model...")
        history_phase1 = self.model_builder.model.fit(
            train_dataset,
            epochs=self.config.warmup_epochs * 2,
            validation_data=val_dataset,
            callbacks=callbacks,
            class_weight=self.class_weights
        )
        
        # Phase 2: Fine-tuning
        logger.info("Phase 2: Fine-tuning...")
        self.model_builder.unfreeze_top_layers(30)
        
        # Recompile with lower learning rate
        self.config.initial_lr /= 10
        self.compile_model()
        
        history_phase2 = self.model_builder.model.fit(
            train_dataset,
            epochs=self.config.epochs,
            initial_epoch=len(history_phase1.history['loss']),
            validation_data=val_dataset,
            callbacks=callbacks,
            class_weight=self.class_weights
        )
        
        # Evaluate on test set
        logger.info("Evaluating on test set...")
        test_results = self.model_builder.model.evaluate(test_dataset)
        
        # Save final model
        final_path = os.path.join(self.config.model_dir, 'emotion_model_advanced_final.keras')
        self.model_builder.model.save(final_path)
        logger.info(f"Model saved to {final_path}")
        
        # Generate report
        self._generate_report(test_dataset, test_results)
        
        return history_phase1, history_phase2
    
    def _generate_report(self, test_dataset, test_results):
        """Generate comprehensive evaluation report"""
        # Get predictions
        y_true = []
        y_pred = []
        
        for x, y in test_dataset:
            predictions = self.model_builder.model.predict(x)
            y_true.extend(np.argmax(y.numpy(), axis=1))
            y_pred.extend(np.argmax(predictions, axis=1))
        
        # Classification report
        emotion_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
        report = classification_report(y_true, y_pred, target_names=emotion_names)
        
        logger.info("\nClassification Report:")
        logger.info(report)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=emotion_names,
                   yticklabels=emotion_names)
        plt.title('Confusion Matrix - Advanced Model')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.model_dir, 'confusion_matrix_advanced.png'))
        plt.close()
        
        # Save report
        with open(os.path.join(self.config.model_dir, 'evaluation_report.txt'), 'w') as f:
            f.write(f"Test Results:\n")
            f.write(f"Loss: {test_results[0]:.4f}\n")
            f.write(f"Accuracy: {test_results[1]:.4f}\n")
            f.write(f"\n{report}")


def main():
    """Main training script"""
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(description='Advanced Emotion Detection Training')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--wandb', action='store_true', help='Use Weights & Biases')
    args = parser.parse_args()
    
    # Load config
    if args.config:
        config = TrainingConfig.from_yaml(args.config)
    else:
        config = TrainingConfig()
    
    # Initialize W&B if requested
    if args.wandb:
        wandb.init(project="emotion-detection-advanced", config=config.__dict__)
    
    # Create trainer
    trainer = AdvancedTrainer(config)
    
    # Train model
    trainer.train()
    
    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()