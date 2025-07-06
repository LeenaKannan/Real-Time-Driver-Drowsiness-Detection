# src/hardware/advanced_alert_system.py
import RPi.GPIO as GPIO
import time
import threading
from enum import Enum
import pygame
import os

class AlertLevel(Enum):
    NORMAL = 0
    WARNING = 1
    CRITICAL = 2

class AdvancedAlertSystem:
    def __init__(self, buzzer_pin=18, led_pin=16, speaker_enabled=True):
        self.buzzer_pin = buzzer_pin
        self.led_pin = led_pin
        self.speaker_enabled = speaker_enabled
        self.current_alert = AlertLevel.NORMAL
        self.alert_thread = None
        self.running = False
        
        # Setup GPIO
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.buzzer_pin, GPIO.OUT)
        GPIO.setup(self.led_pin, GPIO.OUT)
        GPIO.output(self.buzzer_pin, GPIO.LOW)
        GPIO.output(self.led_pin, GPIO.LOW)
        
        # Setup pygame for audio alerts
        if self.speaker_enabled:
            try:
                pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
                self.create_alert_sounds()
            except:
                print("Speaker initialization failed, using buzzer only")
                self.speaker_enabled = False
        
        # Alert patterns: (buzzer_pattern, led_pattern, audio_file)
        self.alert_patterns = {
            AlertLevel.WARNING: {
                'buzzer': [(0.2, 0.3)] * 2,  # Short beeps
                'led': [(0.5, 0.5)] * 4,     # Slow blink
                'audio': 'warning.wav'
            },
            AlertLevel.CRITICAL: {
                'buzzer': [(0.1, 0.1)] * 10,  # Rapid beeps
                'led': [(0.1, 0.1)] * 20,     # Rapid blink
                'audio': 'critical.wav'
            }
        }
        
    def create_alert_sounds(self):
        """Create alert sound files"""
        # This would typically load pre-recorded audio files
        # For now, we'll use pygame to generate simple tones
        pass
    
    def start_alert_system(self):
        """Start the alert monitoring system"""
        self.running = True
        self.alert_thread = threading.Thread(target=self._alert_loop, daemon=True)
        self.alert_thread.start()
        print("Alert system started")
    
    def stop_alert_system(self):
        """Stop the alert system"""
        self.running = False
        if self.alert_thread:
            self.alert_thread.join()
        GPIO.output(self.buzzer_pin, GPIO.LOW)
        GPIO.output(self.led_pin, GPIO.LOW)
        print("Alert system stopped")
    
    def set_alert_level(self, alert_level):
        """Set current alert level"""
        if isinstance(alert_level, str):
            alert_mapping = {
                'NORMAL': AlertLevel.NORMAL,
                'WARNING': AlertLevel.WARNING,
                'CRITICAL': AlertLevel.CRITICAL
            }
            self.current_alert = alert_mapping.get(alert_level, AlertLevel.NORMAL)
        else:
            self.current_alert = alert_level
    
    def _alert_loop(self):
        """Main alert processing loop"""
        last_alert_time = 0
        alert_cooldown = 1.0  # Minimum time between alert cycles
        
        while self.running:
            current_time = time.time()
            
            if (self.current_alert != AlertLevel.NORMAL and 
                current_time - last_alert_time > alert_cooldown):
                
                self._execute_alert_pattern(self.current_alert)
                last_alert_time = current_time
            
            time.sleep(0.1)
    
    def _execute_alert_pattern(self, alert_level):
        """Execute alert pattern for given level"""
        if alert_level == AlertLevel.NORMAL:
            return
            
        pattern = self.alert_patterns[alert_level]
        
        # Start buzzer and LED patterns in parallel
        buzzer_thread = threading.Thread(
            target=self._buzzer_pattern, 
            args=(pattern['buzzer'],),
            daemon=True
        )
        led_thread = threading.Thread(
            target=self._led_pattern, 
            args=(pattern['led'],),
            daemon=True
        )
        
        buzzer_thread.start()
        led_thread.start()
        
        # Play audio alert if enabled
        if self.speaker_enabled:
            self._play_audio_alert(pattern['audio'])
    
    def _buzzer_pattern(self, pattern):
        """Execute buzzer pattern"""
        for on_time, off_time in pattern:
            if not self.running:
                break
            GPIO.output(self.buzzer_pin, GPIO.HIGH)
            time.sleep(on_time)
            GPIO.output(self.buzzer_pin, GPIO.LOW)
            time.sleep(off_time)
    
    def _led_pattern(self, pattern):
        """Execute LED pattern"""
        for on_time, off_time in pattern:
            if not self.running:
                break
            GPIO.output(self.led_pin, GPIO.HIGH)
            time.sleep(on_time)
            GPIO.output(self.led_pin, GPIO.LOW)
            time.sleep(off_time)
    
    def _play_audio_alert(self, audio_file):
        """Play audio alert"""
        try:
            if os.path.exists(f"sounds/{audio_file}"):
                pygame.mixer.music.load(f"sounds/{audio_file}")
                pygame.mixer.music.play()
        except:
            pass
    
    def test_all_alerts(self):
        """Test all alert levels"""
        print("Testing alert system...")
        
        for alert_level in [AlertLevel.WARNING, AlertLevel.CRITICAL]:
            print(f"Testing {alert_level.name} alert...")
            self.set_alert_level(alert_level)
            time.sleep(3)
            self.set_alert_level(AlertLevel.NORMAL)
            time.sleep(1)
        
        print("Alert system test complete")
    
    def cleanup(self):
        """Cleanup GPIO resources"""
        self.stop_alert_system()
        GPIO.cleanup()
        if self.speaker_enabled:
            pygame.mixer.quit()

if __name__ == "__main__":
    # Test the alert system
    alert_system = AdvancedAlertSystem()
    alert_system.start_alert_system()
    
    try:
        alert_system.test_all_alerts()
        time.sleep(5)
    except KeyboardInterrupt:
        pass
    finally:
        alert_system.cleanup()
