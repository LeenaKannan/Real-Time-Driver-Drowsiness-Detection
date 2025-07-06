# tests/test_system.py
import unittest
import cv2
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from inference.drowsiness_detector import DrowsinessDetector

class TestDrowsinessSystem(unittest.TestCase):
    def setUp(self):
        # Create dummy model for testing
        self.create_dummy_model()
        self.detector = DrowsinessDetector('test_model.h5')
    
    def create_dummy_model(self):
        """Create a dummy model for testing"""
        import tensorflow as tf
        
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(128, 128, 3)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(2, activation='softmax')
        ])
        
        model.compile(optimizer='adam', loss='categorical_crossentropy')
        model.save('test_model.h5')
    
    def test_detector_initialization(self):
        """Test detector initialization"""
        self.assertIsNotNone(self.detector)
    
    def test_drowsiness_detection(self):
        """Test drowsiness detection with dummy frame"""
        # Create dummy frame
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        result = self.detector.detect_drowsiness(frame)
        
        self.assertIn('is_drowsy', result)
        self.assertIn('status', result)
        self.assertIn('confidence', result)
    
    def tearDown(self):
        # Clean up
        if os.path.exists('test_model.h5'):
            os.remove('test_model.h5')

if __name__ == '__main__':
    unittest.main()
