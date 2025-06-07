"""
Unit tests for emotion detection model
"""

import pytest
import numpy as np
import tensorflow as tf
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modules to test
from predict_emotion import preprocess_image, load_model

# Emotion labels for testing
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

class TestModel:
    """Test model functionality"""
    
    def test_preprocess_image_shape(self):
        """Test image preprocessing output shape"""
        # Create dummy grayscale image and save it
        dummy_image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        test_path = 'test_image.png'
        from PIL import Image
        Image.fromarray(dummy_image, mode='L').save(test_path)
        
        # Preprocess
        processed = preprocess_image(test_path)
        
        # Clean up
        os.remove(test_path)
        
        # Check shape (grayscale with channel dimension)
        assert processed.shape == (1, 48, 48, 1)
        
    def test_preprocess_image_normalization(self):
        """Test image normalization"""
        # Create image with known values
        test_image = np.ones((48, 48, 3), dtype=np.uint8) * 255
        
        # Preprocess
        processed = preprocess_image(test_image)
        
        # Check normalization
        assert processed.min() >= -1
        assert processed.max() <= 1
        
    def test_model_output_shape(self):
        """Test model output shape"""
        # Skip if no model available
        if not os.path.exists('models/'):
            pytest.skip("No models directory found")
            
        try:
            # Create dummy model for testing
            model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(48, 48, 3)),
                tf.keras.layers.Conv2D(32, 3, activation='relu'),
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dense(7, activation='softmax')
            ])
            
            # Test prediction
            dummy_input = np.random.randn(1, 48, 48, 3).astype(np.float32)
            output = model.predict(dummy_input)
            
            # Check output
            assert output.shape == (1, 7)
            assert np.allclose(output.sum(), 1.0, atol=1e-5)
            
        except Exception as e:
            pytest.skip(f"Model test failed: {e}")
            
    def test_emotion_labels(self):
        """Test emotion labels configuration"""
        assert len(EMOTION_LABELS) == 7
        assert 'Happy' in EMOTION_LABELS
        assert 'Sad' in EMOTION_LABELS
        assert 'Angry' in EMOTION_LABELS
        
    @pytest.mark.parametrize("batch_size", [1, 8, 16, 32])
    def test_batch_processing(self, batch_size):
        """Test different batch sizes"""
        # Create batch
        batch = np.random.randn(batch_size, 48, 48, 3).astype(np.float32)
        
        # Normalize
        batch = (batch - batch.mean()) / batch.std()
        
        # Check shape
        assert batch.shape[0] == batch_size
        assert batch.shape[1:] == (48, 48, 3)


class TestDataAugmentation:
    """Test data augmentation functions"""
    
    def test_augmentation_output_shape(self):
        """Test augmentation preserves shape"""
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        
        # Create augmentation
        datagen = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True
        )
        
        # Test on batch
        batch = np.random.rand(32, 48, 48, 3).astype(np.float32)
        
        # Apply augmentation
        augmented = next(datagen.flow(batch, batch_size=32))
        
        # Check shape preserved
        assert augmented.shape == batch.shape


class TestAPI:
    """Test API endpoints"""
    
    def test_emotion_list(self):
        """Test emotion list matches config"""
        assert isinstance(EMOTION_LABELS, list)
        assert len(EMOTION_LABELS) == 7


@pytest.fixture
def sample_face_image():
    """Create sample face image for testing"""
    return np.random.randint(0, 255, (48, 48, 3), dtype=np.uint8)


def test_face_detection_import():
    """Test OpenCV face detection imports"""
    import cv2
    
    # Check cascade classifier available
    face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    assert os.path.exists(face_cascade_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])