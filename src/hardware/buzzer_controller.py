# src/hardware/buzzer_controller.py
import RPi.GPIO as GPIO
import time
import threading
from enum import Enum

class AlertLevel(Enum):
    NONE = 0
    DROWSY = 1
    SLEEPING = 2
    YAWNING = 3

class BuzzerController:
    def __init__(self, buzzer_pin=18):
        self.buzzer_pin = buzzer_pin
        self.current_alert = AlertLevel.NONE
        self.alert_thread = None
        self.running = False
        
        # Setup GPIO
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.buzzer_pin, GPIO.OUT)
        GPIO.output(self.buzzer_pin, GPIO.LOW)
        
        # Alert patterns (on_time, off_time, repetitions)
        self.alert_patterns = {
            AlertLevel.DROWSY: (0.2, 0.3, 3),      # Short beeps
            AlertLevel.SLEEPING: (0.5, 0.2, 5),    # Long urgent beeps
            AlertLevel.YAWNING: (0.1, 0.1, 2)      # Quick double beep
        }
    
    def start_alert_system(self):
        """Start the alert monitoring thread"""
        self.running = True
        self.alert_thread = threading.Thread(target=self._alert_loop)
        self.alert_thread.daemon = True
        self.alert_thread.start()
    
    def stop_alert_system(self):
        """Stop the alert system"""
        self.running = False
        if self.alert_thread:
            self.alert_thread.join()
        GPIO.output(self.buzzer_pin, GPIO.LOW)
    
    def set_alert_level(self, status):
        """Set the current alert level based on drowsiness status"""
        if status == "DROWSY":
            self.current_alert = AlertLevel.DROWSY
        elif status == "SLEEPING":
            self.current_alert = AlertLevel.SLEEPING
        elif status == "YAWNING":
            self.current_alert = AlertLevel.YAWNING
        else:
            self.current_alert = AlertLevel.NONE
    
    def _alert_loop(self):
        """Main alert monitoring loop"""
        last_alert_time = 0
        alert_cooldown = 2.0  # Minimum time between alert sequences
        
        while self.running:
            current_time = time.time()
            
            if (self.current_alert != AlertLevel.NONE and 
                current_time - last_alert_time > alert_cooldown):
                
                self._play_alert_pattern(self.current_alert)
                last_alert_time = current_time
            
            time.sleep(0.1)
    
    def _play_alert_pattern(self, alert_level):
        """Play the alert pattern for the given level"""
        if alert_level == AlertLevel.NONE:
            return
        
        on_time, off_time, repetitions = self.alert_patterns[alert_level]
        
        for _ in range(repetitions):
            GPIO.output(self.buzzer_pin, GPIO.HIGH)
            time.sleep(on_time)
            GPIO.output(self.buzzer_pin, GPIO.LOW)
            time.sleep(off_time)
    
    def test_buzzer(self):
        """Test the buzzer functionality"""
        print("Testing buzzer...")
        for alert_level in [AlertLevel.DROWSY, AlertLevel.YAWNING, AlertLevel.SLEEPING]:
            print(f"Testing {alert_level.name} pattern")
            self._play_alert_pattern(alert_level)
            time.sleep(1)
        print("Buzzer test complete")
    
    def cleanup(self):
        """Clean up GPIO resources"""
        self.stop_alert_system()
        GPIO.cleanup()
