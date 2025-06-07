#!/usr/bin/env python3
"""
Improved Emotion Detection Model Training
Optimized for accuracy on FER2013 dataset
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Dense, Flatten, Dropout, 
    BatchNormalization, GlobalAveragePooling2D, Activation,
    SeparableConv2D, DepthwiseConv2D, Input
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau,
    LearningRateScheduler
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.applications import EfficientNetB0
import matplotlib.pyplot as plt
from sklearn.utils import class_weight

# Configuration
IMG_SIZE = 48
BATCH_SIZE = 64
EPOCHS = 100
NUM_CLASSES = 7
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Dataset paths - adjust these to your dataset location
BASE_DIR = os.path.expanduser("~/Python/archive")  # Change this to your dataset path
TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR = os.path.join(BASE_DIR, "test")

def create_advanced_cnn(input_shape=(IMG_SIZE, IMG_SIZE, 1), num_classes=NUM_CLASSES):
    """
    Advanced CNN architecture optimized for FER2013
    Uses techniques proven to improve accuracy:
    - Deeper architecture with residual connections
    - Batch normalization for stable training
    - Dropout for regularization
    - Global average pooling instead of flatten
    """
    model = Sequential([
        # Block 1
        Conv2D(64, (3, 3), padding='same', kernel_regularizer=l2(0.0001), 
               input_shape=input_shape),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(64, (3, 3), padding='same', kernel_regularizer=l2(0.0001)),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Block 2
        Conv2D(128, (3, 3), padding='same', kernel_regularizer=l2(0.0001)),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(128, (3, 3), padding='same', kernel_regularizer=l2(0.0001)),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Block 3
        Conv2D(256, (3, 3), padding='same', kernel_regularizer=l2(0.0001)),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(256, (3, 3), padding='same', kernel_regularizer=l2(0.0001)),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(256, (3, 3), padding='same', kernel_regularizer=l2(0.0001)),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Block 4
        Conv2D(512, (3, 3), padding='same', kernel_regularizer=l2(0.0001)),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(512, (3, 3), padding='same', kernel_regularizer=l2(0.0001)),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(512, (3, 3), padding='same', kernel_regularizer=l2(0.0001)),
        BatchNormalization(),
        Activation('relu'),
        GlobalAveragePooling2D(),
        
        # Classification layers
        Dense(512, kernel_regularizer=l2(0.0001)),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.5),
        
        Dense(256, kernel_regularizer=l2(0.0001)),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.5),
        
        Dense(num_classes, activation='softmax')
    ])
    
    return model

def create_efficientnet_model(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=NUM_CLASSES):
    """
    EfficientNet-based model for better accuracy
    Requires RGB input, so we'll convert grayscale to RGB
    """
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    # Freeze base model initially
    base_model.trainable = False
    
    inputs = Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(256, activation='relu', kernel_regularizer=l2(0.0001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.Model(inputs, outputs)
    return model, base_model

def create_data_generators():
    """
    Create optimized data generators with augmentation
    """
    # Training data augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2
    )
    
    # Validation/Test data - only rescaling
    test_datagen = ImageDataGenerator(
        rescale=1./255
    )
    
    # Create generators
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode='grayscale',
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    validation_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode='grayscale',
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode='grayscale',
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    return train_generator, validation_generator, test_generator

def get_class_weights(generator):
    """
    Calculate class weights for imbalanced dataset
    """
    classes = generator.classes
    weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(classes),
        y=classes
    )
    return dict(enumerate(weights))

def cosine_annealing_with_warmup(epoch, lr):
    """
    Learning rate schedule with warmup and cosine annealing
    """
    warmup_epochs = 10
    max_epochs = EPOCHS
    initial_lr = 0.001
    min_lr = 1e-6
    
    if epoch < warmup_epochs:
        return initial_lr * (epoch + 1) / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / (max_epochs - warmup_epochs)
        return min_lr + (initial_lr - min_lr) * 0.5 * (1 + np.cos(np.pi * progress))

def train_model():
    """
    Main training function
    """
    print("Setting up data generators...")
    train_gen, val_gen, test_gen = create_data_generators()
    
    print(f"Training samples: {train_gen.samples}")
    print(f"Validation samples: {val_gen.samples}")
    print(f"Test samples: {test_gen.samples}")
    
    # Get class weights
    class_weights = get_class_weights(train_gen)
    print(f"Class weights: {class_weights}")
    
    # Create model
    print("\nCreating model...")
    model = create_advanced_cnn()
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("Model architecture:")
    model.summary()
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            'best_emotion_model_improved.keras',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_accuracy',
            patience=20,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        LearningRateScheduler(cosine_annealing_with_warmup, verbose=0)
    ]
    
    # Train model
    print("\nStarting training...")
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_loss, test_accuracy = model.evaluate(test_gen, verbose=1)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Save final model
    model.save('emotion_model_final.keras')
    print("\nModel saved as 'emotion_model_final.keras'")
    
    # Plot training history
    plot_training_history(history)
    
    return model, history

def plot_training_history(history):
    """
    Plot training and validation metrics
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Accuracy
    axes[0].plot(history.history['accuracy'], label='Training')
    axes[0].plot(history.history['val_accuracy'], label='Validation')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    # Loss
    axes[1].plot(history.history['loss'], label='Training')
    axes[1].plot(history.history['val_loss'], label='Validation')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

if __name__ == "__main__":
    # Check if dataset exists
    if not os.path.exists(TRAIN_DIR):
        print(f"Error: Dataset not found at {BASE_DIR}")
        print("Please update BASE_DIR to point to your FER2013 dataset location")
        exit(1)
    
    # Train model
    model, history = train_model()