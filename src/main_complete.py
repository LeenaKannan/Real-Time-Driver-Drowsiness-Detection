# src/main_complete.py
import cv2
import time
import yaml
import argparse
import threading
from src.inference.pi_camera_inference import PiCameraInference
from src.hardware.advanced_alert_system import AdvancedAlertSystem
from src.hardware.oled_display import OLEDDisplay
from src.utils.logger import setup_logger

class CompleteDrowsinessSystem:
    def __init__(self, config_path="config.yaml"):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.logger = setup_logger("complete_drowsiness_system")
        
        # Initialize components
        self.camera_inference = PiCameraInference(
            model_path="models/drowsiness_detection_quantized.tflite",
            camera_resolution=(640, 480)
        )
        
        self.alert_system = AdvancedAlertSystem(
            buzzer_pin=self.config['hardware']['buzzer_pin'],
            led_pin=16,
            speaker_enabled=True
        )
        
        try:
            self.oled_display = OLEDDisplay()
            self.display_available = True
        except Exception as e:
            self.logger.warning(f"OLED display not available: {e}")
            self.display_available = False
        
        self.running = False
        self.stats = {
            'total_frames': 0,
            'drowsy_detections': 0,
            'alerts_triggered': 0,
            'start_time': time.time()
        }
    
    def start_system(self):
        """Start the complete drowsiness detection system"""
        self.logger.info("Starting complete drowsiness detection system...")
        
        # Start all components
        self.camera_inference.start_inference()
        self.alert_system.start_alert_system()
        
        if self.display_available:
            self.oled_display.start_display_thread()
        
        self.running = True
        
        # Start main processing loop
        self.main_loop()
    
    def main_loop(self):
        """Main processing loop"""
        last_log_time = time.time()
        
        try:
            while self.running:
                # Get latest inference result
                result = self.camera_inference.get_latest_result()
                
                if result:
                    self.process_result(result)
                    
                    # Log statistics every 30 seconds
                    if time.time() - last_log_time > 30:
                        self.log_statistics()
                        last_log_time = time.time()
                
                time.sleep(0.1)  # Small delay to prevent excessive CPU usage
                
        except KeyboardInterrupt:
            self.logger.info("System stopped by user")
        finally:
            self.cleanup()
    
    def process_result(self, result):
        """Process inference result and trigger appropriate actions"""
        self.stats['total_frames'] += 1
        
        alert_level = result['alert_level']
        confidence = result['confidence']
        
        # Update alert system
        self.alert_system.set_alert_level(alert_level)
        
        # Update OLED display
        if self.display_available:
            display_data = {
                'status': alert_level,
                'confidence': confidence,
                'fps': result['fps'],
                'total_frames': self.stats['total_frames']
            }
            self.oled_display.update_status(display_data)
        
        # Log alerts
        if alert_level in ['WARNING', 'CRITICAL']:
            self.stats['drowsy_detections'] += 1
            
            if alert_level == 'CRITICAL':
                self.stats['alerts_triggered'] += 1
                self.logger.warning(
                    f"CRITICAL ALERT: Drowsiness detected with {confidence:.3f} confidence"
                )
            else:
                self.logger.info(
                    f"WARNING: Potential drowsiness detected with {confidence:.3f} confidence"
                )
    
    def log_statistics(self):
        """Log system statistics"""
        runtime = time.time() - self.stats['start_time']
        avg_fps = self.stats['total_frames'] / runtime if runtime > 0 else 0
        
        self.logger.info(
            f"Statistics - Runtime: {runtime:.1f}s, "
            f"Frames: {self.stats['total_frames']}, "
            f"Avg FPS: {avg_fps:.1f}, "
            f"Drowsy detections: {self.stats['drowsy_detections']}, "
            f"Critical alerts: {self.stats['alerts_triggered']}"
        )
    
    def cleanup(self):
        """Cleanup all system components"""
        self.logger.info("Cleaning up system components...")
        
        self.running = False
        
        # Cleanup components
        self.camera_inference.cleanup()
        self.alert_system.cleanup()
        
        if self.display_available:
            self.oled_display.stop_display_thread()
        
        # Log final statistics
        self.log_statistics()
        self.logger.info("System cleanup complete")

def main():
    parser = argparse.ArgumentParser(description="Complete Drowsiness Detection System")
    parser.add_argument('--config', default='config.yaml', help='Configuration file path')
    parser.add_argument('--test', action='store_true', help='Run system test')
    args = parser.parse_args()
    
    system = CompleteDrowsinessSystem(args.config)
    
    if args.test:
        # Test mode - run for 60 seconds
        print("Running system test for 60 seconds...")
        system.start_system()
        time.sleep(60)
        system.cleanup()
    else:
        # Normal operation
        system.start_system()

if __name__ == "__main__":
    main()
