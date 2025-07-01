# src/main_pi.py
import cv2
import time
import yaml
import argparse
from inference.tflite_detector import TFLiteDrowsinessDetector
from hardware.oled_display import OLEDDisplay
from hardware.buzzer_controller import BuzzerController
from utils.logger import setup_logger

class PiDrowsinessSystem:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.logger = setup_logger("pi_drowsiness")
        
        # Initialize TFLite detector
        self.detector = TFLiteDrowsinessDetector(
            model_path="models/drowsiness_model.tflite",
            ear_threshold=self.config['detection']['ear_threshold']
        )
        
        # Hardware components
        try:
            self.oled = OLEDDisplay()
            self.buzzer = BuzzerController(self.config['hardware']['buzzer_pin'])
            self.hardware_available = True
            self.logger.info("Hardware initialized")
        except Exception as e:
            self.logger.warning(f"Hardware init failed: {e}")
            self.hardware_available = False
        
        self.running = False
    
    def start_detection(self):
        """Start real-time detection"""
        self.logger.info("Starting Pi drowsiness detection")
        
        # Initialize camera
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency
        
        if not cap.isOpened():
            self.logger.error("Failed to open camera")
            return
        
        if self.hardware_available:
            self.oled.show_startup_message()
            time.sleep(2)
            self.buzzer.start_alert_system()
        
        self.running = True
        fps_counter = 0
        start_time = time.time()
        
        try:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    continue
                
                # Detect drowsiness
                result = self.detector.detect_drowsiness(frame)
                
                # Update hardware
                if self.hardware_available:
                    self.oled.update_status(result)
                    self.buzzer.set_alert_level(result['status'])
                
                # Log alerts
                if result['alert_required']:
                    self.logger.warning(f"ALERT: {result['status']} - Confidence: {result['confidence']:.3f}")
                
                # FPS calculation
                fps_counter += 1
                if fps_counter % 30 == 0:
                    fps = 30 / (time.time() - start_time)
                    self.logger.info(f"FPS: {fps:.1f}")
                    start_time = time.time()
                
                # Optional: Save frame for debugging
                if result['status'] in ['DROWSY', 'SLEEPING']:
                    timestamp = int(time.time())
                    cv2.imwrite(f"logs/alert_{timestamp}.jpg", frame)
                
                time.sleep(0.01)  # Small delay to prevent overheating
                
        except KeyboardInterrupt:
            self.logger.info("System stopped by user")
        finally:
            self.cleanup(cap)
    
    def cleanup(self, cap):
        """Cleanup resources"""
        self.running = False
        cap.release()
        
        if self.hardware_available:
            self.buzzer.cleanup()
            self.oled.stop_display_thread()
        
        self.logger.info("System cleanup complete")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--headless', action='store_true', help='Run without display')
    
    args = parser.parse_args()
    
    system = PiDrowsinessSystem(args.config)
    system.start_detection()

if __name__ == "__main__":
    main()
