# src/main.py
import cv2
import time
import threading
import argparse
import yaml
import logging
from pathlib import Path

from inference.drowsiness_detector import DrowsinessDetector
from hardware.oled_display import OLEDDisplay
from hardware.buzzer_controller import BuzzerController
from utils.logger import setup_logger
from utils.video_utils import VideoProcessor

class DrowsinessDetectionSystem:
    def __init__(self, config_path="config.yaml"):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Setup logging
        self.logger = setup_logger("drowsiness_system")
        
        # Initialize components
        self.detector = DrowsinessDetector(
            model_path=self.config['model']['path'],
            ear_threshold=self.config['detection']['ear_threshold'],
            yawn_threshold=self.config['detection']['yawn_threshold']
        )
        
        # Hardware components (only on Raspberry Pi)
        self.oled_display = None
        self.buzzer = None
        
        try:
            self.oled_display = OLEDDisplay()
            self.buzzer = BuzzerController(
                pin=self.config['hardware']['buzzer_pin']
            )
            self.hardware_available = True
            self.logger.info("Hardware components initialized successfully")
        except Exception as e:
            self.logger.warning(f"Hardware initialization failed: {e}")
            self.hardware_available = False
        
        # Video processing
        self.video_processor = VideoProcessor()
        self.running = False
        
        # Statistics
        self.stats = {
            'total_frames': 0,
            'drowsy_detections': 0,
            'alert_count': 0,
            'start_time': time.time()
        }
    
    def start_system(self, video_source=0, demo_mode=False):
        """Start the drowsiness detection system"""
        self.logger.info("Starting Drowsiness Detection System")
        
        # Initialize video capture
        if demo_mode and isinstance(video_source, str):
            cap = cv2.VideoCapture(video_source)
        else:
            cap = cv2.VideoCapture(video_source)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
        
        if not cap.isOpened():
            self.logger.error("Failed to open video source")
            return
        
        # Start hardware components
        if self.hardware_available:
            self.oled_display.show_startup_message()
            time.sleep(2)
            self.oled_display.start_display_thread()
            self.buzzer.start_alert_system()
        
        self.running = True
        fps_counter = 0
        fps_start_time = time.time()
        
        try:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    if demo_mode:
                        # Loop the demo video
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    else:
                        self.logger.error("Failed to read frame")
                        break
                
                # Process frame
                detection_result = self.detector.detect_drowsiness(frame)
                self.stats['total_frames'] += 1
                
                # Update hardware
                if self.hardware_available:
                    self.oled_display.update_status(detection_result)
                    self.buzzer.set_alert_level(detection_result['status'])
                
                # Handle alerts
                if detection_result['alert_required']:
                    self.stats['alert_count'] += 1
                    if detection_result['status'] in ['DROWSY', 'SLEEPING']:
                        self.stats['drowsy_detections'] += 1
                
                # Draw visualization
                self._draw_visualization(frame, detection_result)
                
                # Display frame
                cv2.imshow('Drowsiness Detection System', frame)
                
                # Calculate FPS
                fps_counter += 1
                if fps_counter % 30 == 0:
                    fps = 30 / (time.time() - fps_start_time)
                    self.logger.info(f"FPS: {fps:.2f}")
                    fps_start_time = time.time()
                
                # Check for exit
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    self._save_statistics()
                elif key == ord('t') and self.hardware_available:
                    self.buzzer.test_buzzer()
        
        except KeyboardInterrupt:
            self.logger.info("System interrupted by user")
        
        finally:
            self._cleanup(cap)
    
    def _draw_visualization(self, frame, detection_result):
        """Draw detection visualization on frame"""
        h, w = frame.shape[:2]
        
        # Status box
        status = detection_result['status']
        confidence = detection_result['confidence']
        
        # Color based on status
        if status == "AWAKE":
            color = (0, 255, 0)  # Green
        elif status == "DROWSY" or status == "YAWNING":
            color = (0, 165, 255)  # Orange
        elif status == "SLEEPING":
            color = (0, 0, 255)  # Red
        else:
            color = (128, 128, 128)  # Gray
        
        # Draw status box
        cv2.rectangle(frame, (10, 10), (300, 120), color, 2)
        cv2.rectangle(frame, (10, 10), (300, 40), color, -1)
        
        # Status text
        cv2.putText(frame, f"STATUS: {status}", (15, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Metrics
        cv2.putText(frame, f"Confidence: {confidence:.3f}", (15, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.putText(frame, f"EAR: {detection_result['ear']:.3f}", (15, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.putText(frame, f"MAR: {detection_result['mar']:.3f}", (15, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Alert indicator
        if detection_result['alert_required']:
            cv2.rectangle(frame, (w-150, 10), (w-10, 60), (0, 0, 255), -1)
            cv2.putText(frame, "ALERT!", (w-130, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Statistics
        runtime = time.time() - self.stats['start_time']
        cv2.putText(frame, f"Runtime: {runtime:.0f}s", (10, h-60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Frames: {self.stats['total_frames']}", (10, h-40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Alerts: {self.stats['alert_count']}", (10, h-20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def _save_statistics(self):
        """Save system statistics"""
        runtime = time.time() - self.stats['start_time']
        stats_data = {
            'runtime_seconds': runtime,
            'total_frames': self.stats['total_frames'],
            'drowsy_detections': self.stats['drowsy_detections'],
            'alert_count': self.stats['alert_count'],
            'avg_fps': self.stats['total_frames'] / runtime if runtime > 0 else 0,
            'drowsiness_rate': self.stats['drowsy_detections'] / self.stats['total_frames'] if self.stats['total_frames'] > 0 else 0
        }
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        stats_file = f"logs/session_stats_{timestamp}.yaml"
        
        with open(stats_file, 'w') as f:
            yaml.dump(stats_data, f)
        
        self.logger.info(f"Statistics saved to {stats_file}")
    
    def _cleanup(self, cap):
        """Clean up resources"""
        self.logger.info("Cleaning up system resources")
        
        self.running = False
        cap.release()
        cv2.destroyAllWindows()
        
        if self.hardware_available:
            self.oled_display.stop_display_thread()
            self.buzzer.cleanup()
        
        self._save_statistics()

def main():
    parser = argparse.ArgumentParser(description='Driver Drowsiness Detection System')
    parser.add_argument('--config', default='config.yaml', help='Configuration file path')
    parser.add_argument('--video', default=0, help='Video source (camera index or file path)')
    parser.add_argument('--demo', action='store_true', help='Run in demo mode with video file')
    parser.add_argument('--test-hardware', action='store_true', help='Test hardware components')
    
    args = parser.parse_args()
    
    # Initialize system
    system = DrowsinessDetectionSystem(args.config)
    
    if args.test_hardware:
        if system.hardware_available:
            system.buzzer.test_buzzer()
            system.oled_display.show_startup_message()
            time.sleep(3)
        else:
            print("Hardware not available for testing")
        return
    
    # Convert video argument
    video_source = args.video
    if video_source != '0' and not args.demo:
        try:
            video_source = int(video_source)
        except ValueError:
            pass  # Keep as string for file path
    
    # Start system
    system.start_system(video_source, args.demo)

if __name__ == "__main__":
    main()
