# src/hardware/oled_display.py
import board
import busio
import adafruit_ssd1306
from PIL import Image, ImageDraw, ImageFont
import time
import threading

class OLEDDisplay:
    def __init__(self, width=128, height=64):
        self.width = width
        self.height = height
        
        # Initialize I2C and display
        i2c = busio.I2C(board.SCL, board.SDA)
        self.oled = adafruit_ssd1306.SSD1306_I2C(width, height, i2c)
        
        # Clear display
        self.oled.fill(0)
        self.oled.show()
        
        # Create image and drawing context
        self.image = Image.new("1", (width, height))
        self.draw = ImageDraw.Draw(self.image)
        
        # Load font
        try:
            self.font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
            self.small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
        except:
            self.font = ImageFont.load_default()
            self.small_font = ImageFont.load_default()
        
        self.current_status = "INITIALIZING"
        self.display_thread = None
        self.running = False
        
    def start_display_thread(self):
        """Start the display update thread"""
        self.running = True
        self.display_thread = threading.Thread(target=self._display_loop)
        self.display_thread.daemon = True
        self.display_thread.start()
    
    def stop_display_thread(self):
        """Stop the display update thread"""
        self.running = False
        if self.display_thread:
            self.display_thread.join()
    
    def _display_loop(self):
        """Main display update loop"""
        while self.running:
            self._update_display()
            time.sleep(0.1)  # Update at 10 FPS
    
    def update_status(self, status_data):
        """Update the display with new status data"""
        self.current_status = status_data.get('status', 'UNKNOWN')
        self.confidence = status_data.get('confidence', 0.0)
        self.ear = status_data.get('ear', 0.0)
        self.mar = status_data.get('mar', 0.0)
    
    def _update_display(self):
        """Update the OLED display"""
        # Clear the image
        self.draw.rectangle((0, 0, self.width, self.height), outline=0, fill=0)
        
        # Status header
        status_color = 1
        if self.current_status == "DROWSY":
            status_color = 1  # Blinking effect could be added
        elif self.current_status == "SLEEPING":
            status_color = 1
        
        # Draw status
        self.draw.text((2, 2), f"STATUS: {self.current_status}", font=self.font, fill=status_color)
        
        # Draw metrics
        if hasattr(self, 'confidence'):
            self.draw.text((2, 18), f"Confidence: {self.confidence:.2f}", font=self.small_font, fill=1)
        if hasattr(self, 'ear'):
            self.draw.text((2, 30), f"EAR: {self.ear:.3f}", font=self.small_font, fill=1)
        if hasattr(self, 'mar'):
            self.draw.text((2, 42), f"MAR: {self.mar:.3f}", font=self.small_font, fill=1)
        
        # Draw timestamp
        current_time = time.strftime("%H:%M:%S")
        self.draw.text((2, 54), current_time, font=self.small_font, fill=1)
        
        # Alert indicator
        if self.current_status in ["DROWSY", "SLEEPING", "YAWNING"]:
            # Draw alert box
            self.draw.rectangle((90, 18, 126, 54), outline=1, fill=0)
            self.draw.text((95, 30), "ALERT!", font=self.small_font, fill=1)
        
        # Display the image
        self.oled.image(self.image)
        self.oled.show()
    
    def show_startup_message(self):
        """Show startup message"""
        self.draw.rectangle((0, 0, self.width, self.height), outline=0, fill=0)
        self.draw.text((10, 20), "Drowsiness", font=self.font, fill=1)
        self.draw.text((10, 35), "Detection", font=self.font, fill=1)
        self.draw.text((10, 50), "System Ready", font=self.small_font, fill=1)
        self.oled.image(self.image)
        self.oled.show()
