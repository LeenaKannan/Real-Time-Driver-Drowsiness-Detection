# tests/test_system.py
import unittest
import cv2
import numpy as np
import tempfile
import os
from unittest.mock import Mock, patch

from src.inference.drowsiness_detector import DrowsinessDetector
from src.hardware.oled_display import OLEDDisplay
from src.hardware.buzzer_controller import BuzzerController

class TestDrowsinessDetection(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        # Create a mock model
        self.mock_model_path = "test_model.h5"
        
        # Create test images
        self.test_open_eye = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        self.test_closed_eye = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        
    def test_ear_calculation(self):
        """Test Eye Aspect Ratio calculation"""
        detector = DrowsinessDetector(self.mock_model_path)
        
        # Test with known eye landmarks
        eye_landmarks = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
        ear = detector.calculate_ear(eye_landmarks)
        
        self.assertIsInstance(ear, float)
        self.assertGreater(ear, 0)
    
    def test_drowsiness_detection_pipeline(self):
        """Test the complete drowsiness detection pipeline"""
        with patch('tensorflow.keras.models.load_model'):
            detector = DrowsinessDetector(self.mock_model_path)
            
            # Create a test frame
            test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            
            # Mock MediaPipe results
            with patch.object(detector.face_mesh, 'process') as mock_process:
                mock_process.return_value.multi_face_landmarks = None
                
                result = detector.detect_drowsiness(test_frame)
                
                self.assertIn('status', result)
                self.assertIn('confidence', result)
                self.assertIn('alert_required', result)
    
    def test_hardware_initialization(self):
        """Test hardware component initialization"""
        # Test OLED display (mocked)
        with patch('board.SCL'), patch('board.SDA'), patch('busio.I2C'):
            with patch('adafruit_ssd1306.SSD1306_I2C'):
                display = OLEDDisplay()
                self.assertIsNotNone(display)
        
        # Test buzzer controller (mocked)
        with patch('RPi.GPIO.setmode'), patch('RPi.GPIO.setup'):
            buzzer = BuzzerController()
            self.assertIsNotNone(buzzer)

class TestModelPerformance(unittest.TestCase):
    def test_model_accuracy(self):
        """Test model accuracy on validation data"""
        # This would test against a known validation set
        # Expected accuracy > 95% based on research results
        pass
    
    def test_inference_speed(self):
        """Test real-time inference performance"""
        # Should process at least 20 FPS on Raspberry Pi 4
        pass

if __name__ == '__main__':
    unittest.main()
