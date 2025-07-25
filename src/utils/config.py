import yaml
import os
from pathlib import Path

class Config:
    def __init__(self, config_path="config.yaml"):
        self.config_path = config_path
        self.config = self.load_config()
    
    def load_config(self):
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            return self.get_default_config()
    
    def get_default_config(self):
        return {
            'model': {
                'path': "models/drowsiness_detection_model.h5",
                'type': "cnn"
            },
            'detection': {
                'ear_threshold': 0.25,
                'yawn_threshold': 0.6,
                'drowsy_frame_threshold': 20,
                'sleep_frame_threshold': 60
            },
            'hardware': {
                'buzzer_pin': 18,
                'oled_width': 128,
                'oled_height': 64,
                'i2c_address': '0x3C'
            },
            'video': {
                'width': 640,
                'height': 480,
                'fps': 30
            },
            'logging': {
                'level': "INFO",
                'file': "logs/drowsiness_system.log"
            },
            'alerts': {
                'cooldown_seconds': 2.0,
                'patterns': {
                    'drowsy': [0.2, 0.3, 3],
                    'sleeping': [0.5, 0.2, 5],
                    'yawning': [0.1, 0.1, 2]
                }
            }
        }
    
    def get(self, key, default=None):
        keys = key.split('.')
        value = self.config
        for k in keys:
            value = value.get(k, {})
        return value if value != {} else default