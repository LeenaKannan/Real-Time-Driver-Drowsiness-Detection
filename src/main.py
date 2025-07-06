# src/main.py
import cv2
import time
import yaml
import argparse
import threading
import signal
import sys
from pathlib import Path

from src.inference.drowsiness_detector import DrowsinessDetector
from src.hardware.buzzer_controller import BuzzerController
from src.hardware.oled_display import OLEDDisplay
from src.utils.logger import setup_logger
from src.utils.config import Config

class DrowsinessDetectionSystem:
    def __init__(self, config_path="config.yaml"):
        self.config = Config(config_path)
        self.logger = setup_logger("drowsiness_system")
        
        # Initialize components
        self.detector = DrowsinessDetector(
            model_path=self.config.get('model.path'),
            model_type=self.config.get('model.type', 'cnn')
        )
        
        try:
            self.buzzer = BuzzerController(
                buzzer_pin=self.config.get('hardware.buzzer_pin', 18)
            )
            self.buzzer.start_alert_system()
            
            self.oled = OLEDDisplay(
                width=self.config.get('hardware.oled_width', 128),
                height=self.config.get('hardware.oled_height', 64)
            )
            self.oled.start_display_thread()
            self.hardware_available = True
        except Exception as e:
            self.logger.warning(f"Hardware not available: {e}")
            self.hardware_available = False
        
        self.running = False
        self.stats = {
            'total_frames': 0,
            'drowsy_detections': 0,
            'alerts_triggered': 0,
            'start_time': time.time()
        }
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def signal_handler(self, signum, frame):
        self.logger.info("Received shutdown signal")
        self.stop()
    
    def start(self, video_source=0):
        """Start the drowsiness detection system"""
        self.logger.info("Starting drowsiness detection system...")
        self.running = True
        
        # Initialize video capture
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            self.logger.error(f"Cannot open video source: {video_source}")
            return False
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.get('video.width', 640))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.get('video.height', 480))
        cap.set(cv2.CAP_PROP_FPS, self.config.get('video.fps', 30))
        
        fps_counter = 0
        fps_start_time = time.time()
        
        try:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    self.logger.warning("Failed to read frame")
                    continue
                
                # Process frame
                result = self.detector.detect_drowsiness(frame)
                self.process_result(result, frame)
                
                # Calculate FPS
                fps_counter += 1
                if fps_counter % 30 == 0:
                    current_fps = 30 / (time.time() - fps_start_time)
                    fps_start_time = time.time()
                    self.logger.info(f"Current FPS: {current_fps:.1f}")
                
                # Display frame (optional for debugging)
                if not self.hardware_available:
                    self.draw_debug_info(frame, result)
                    cv2.imshow('Drowsiness Detection', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
        except Exception as e:
            self.logger.error(f"Error in main loop: {e}")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.cleanup()
    
    def process_result(self, result, frame):
        """Process detection result and trigger alerts"""
        self.stats['total_frames'] += 1
        
        if result['is_drowsy']:
            self.stats['drowsy_detections'] += 1
            
            if self.hardware_available:
                self.buzzer.set_alert_level(result['status'])
                self.oled.update_status({
                    'status': result['status'],
                    'confidence': result['confidence'],
                    'ear': result.get('ear', 0),
                    'mar': result.get('mar', 0)
                })
            
            if result['status'] == 'SLEEPING':
                self.stats['alerts_triggered'] += 1
                self.logger.warning(f"CRITICAL ALERT: {result['status']} - Confidence: {result['confidence']:.3f}")
    
    def draw_debug_info(self, frame, result):
        """Draw debug information on frame"""
        status_color = (0, 255, 0) if not result['is_drowsy'] else (0, 0, 255)
        cv2.putText(frame, f"Status: {result['status']}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        cv2.putText(frame, f"Confidence: {result['confidence']:.3f}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        if 'ear' in result:
            cv2.putText(frame, f"EAR: {result['ear']:.3f}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    def stop(self):
        """Stop the system"""
        self.running = False
        self.logger.info("Stopping drowsiness detection system...")
    
    def cleanup(self):
        """Cleanup resources"""
        if self.hardware_available:
            self.buzzer.cleanup()
            self.oled.stop_display_thread()
        
        # Log final statistics
        runtime = time.time() - self.stats['start_time']
        self.logger.info(f"Final stats - Runtime: {runtime:.1f}s, "
                        f"Frames: {self.stats['total_frames']}, "
                        f"Drowsy detections: {self.stats['drowsy_detections']}, "
                        f"Alerts: {self.stats['alerts_triggered']}")

def main():
    parser = argparse.ArgumentParser(description="Driver Drowsiness Detection System")
    parser.add_argument('--config', default='config.yaml', help='Configuration file')
    parser.add_argument('--video', default=0, help='Video source (0 for camera, path for video file)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    system = DrowsinessDetectionSystem(args.config)
    
    try:
        system.start(args.video)
    except KeyboardInterrupt:
        system.stop()

if __name__ == "__main__":
    main()
