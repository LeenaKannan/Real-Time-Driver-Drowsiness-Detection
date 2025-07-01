import unittest
from unittest.mock import patch, MagicMock, call
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.hardware.oled_display import OLEDDisplay
from src.hardware.buzzer_controller import BuzzerController
from src.hardware.gpio_manager import GPIOManager

class TestOLEDDisplay(unittest.TestCase):
    
    @patch('board.SCL')
    @patch('board.SDA')
    @patch('busio.I2C')
    @patch('adafruit_ssd1306.SSD1306_I2C')
    def setUp(self, mock_ssd1306, mock_i2c, mock_sda, mock_scl):
        self.mock_display = MagicMock()
        mock_ssd1306.return_value = self.mock_display
        self.oled = OLEDDisplay()
    
    def test_initialization(self):
        self.assertIsNotNone(self.oled)
        self.assertEqual(self.oled.width, 128)
        self.assertEqual(self.oled.height, 64)
    
    def test_clear_display(self):
        self.oled.clear()
        self.mock_display.fill.assert_called_with(0)
        self.mock_display.show.assert_called_once()
    
    def test_display_text(self):
        with patch('PIL.ImageDraw.Draw') as mock_draw:
            mock_draw_instance = MagicMock()
            mock_draw.return_value = mock_draw_instance
            
            self.oled.display_text("Alert", "Driver Drowsy")
            mock_draw_instance.text.assert_called()
            self.mock_display.image.assert_called_once()
            self.mock_display.show.assert_called_once()
    
    def test_display_status(self):
        with patch('PIL.ImageDraw.Draw') as mock_draw:
            mock_draw_instance = MagicMock()
            mock_draw.return_value = mock_draw_instance
            
            self.oled.display_status("ALERT", 0.85, 25)
            mock_draw_instance.text.assert_called()
            self.mock_display.image.assert_called_once()
            self.mock_display.show.assert_called_once()
    
    def test_show_startup_message(self):
        with patch('PIL.ImageDraw.Draw') as mock_draw:
            mock_draw_instance = MagicMock()
            mock_draw.return_value = mock_draw_instance
            
            self.oled.show_startup_message()
            mock_draw_instance.text.assert_called()
            self.mock_display.image.assert_called_once()
            self.mock_display.show.assert_called_once()

class TestBuzzerController(unittest.TestCase):
    
    @patch('RPi.GPIO.setup')
    @patch('RPi.GPIO.setmode')
    @patch('RPi.GPIO.PWM')
    def setUp(self, mock_pwm, mock_setmode, mock_setup):
        self.mock_pwm_instance = MagicMock()
        mock_pwm.return_value = self.mock_pwm_instance
        self.buzzer = BuzzerController(pin=18)
    
    def test_initialization(self):
        self.assertEqual(self.buzzer.pin, 18)
        self.assertIsNotNone(self.buzzer.pwm)
    
    def test_single_beep(self):
        with patch('time.sleep') as mock_sleep:
            self.buzzer.beep(duration=0.5, frequency=1000)
            self.mock_pwm_instance.start.assert_called_with(50)
            self.mock_pwm_instance.ChangeFrequency.assert_called_with(1000)
            self.mock_pwm_instance.stop.assert_called_once()
            mock_sleep.assert_called_with(0.5)
    
    def test_alert_pattern(self):
        with patch('time.sleep') as mock_sleep:
            self.buzzer.alert_pattern()
            self.assertEqual(self.mock_pwm_instance.start.call_count, 3)
            self.assertEqual(self.mock_pwm_instance.stop.call_count, 3)
    
    def test_warning_pattern(self):
        with patch('time.sleep') as mock_sleep:
            self.buzzer.warning_pattern()
            self.assertEqual(self.mock_pwm_instance.start.call_count, 2)
            self.assertEqual(self.mock_pwm_instance.stop.call_count, 2)
    
    def test_stop_buzzer(self):
        self.buzzer.stop()
        self.mock_pwm_instance.stop.assert_called_once()
    
    @patch('RPi.GPIO.cleanup')
    def test_cleanup(self, mock_cleanup):
        self.buzzer.cleanup()
        self.mock_pwm_instance.stop.assert_called_once()
        mock_cleanup.assert_called_once()

class TestGPIOManager(unittest.TestCase):
    
    @patch('RPi.GPIO.setmode')
    @patch('RPi.GPIO.setup')
    def setUp(self, mock_setup, mock_setmode):
        self.gpio_manager = GPIOManager()
    
    @patch('RPi.GPIO.setup')
    def test_setup_pin_output(self, mock_setup):
        self.gpio_manager.setup_pin(18, 'OUTPUT')
        mock_setup.assert_called_with(18, self.gpio_manager.GPIO.OUT)
    
    @patch('RPi.GPIO.setup')
    def test_setup_pin_input(self, mock_setup):
        self.gpio_manager.setup_pin(24, 'INPUT')
        mock_setup.assert_called_with(24, self.gpio_manager.GPIO.IN)
    
    @patch('RPi.GPIO.output')
    def test_set_pin_high(self, mock_output):
        self.gpio_manager.set_pin(18, True)
        mock_output.assert_called_with(18, self.gpio_manager.GPIO.HIGH)
    
    @patch('RPi.GPIO.output')
    def test_set_pin_low(self, mock_output):
        self.gpio_manager.set_pin(18, False)
        mock_output.assert_called_with(18, self.gpio_manager.GPIO.LOW)
    
    @patch('RPi.GPIO.input')
    def test_read_pin(self, mock_input):
        mock_input.return_value = True
        result = self.gpio_manager.read_pin(24)
        mock_input.assert_called_with(24)
        self.assertTrue(result)
    
    @patch('RPi.GPIO.PWM')
    def test_setup_pwm(self, mock_pwm):
        mock_pwm_instance = MagicMock()
        mock_pwm.return_value = mock_pwm_instance
        
        pwm = self.gpio_manager.setup_pwm(18, 1000)
        mock_pwm.assert_called_with(18, 1000)
        self.assertEqual(pwm, mock_pwm_instance)
    
    @patch('RPi.GPIO.cleanup')
    def test_cleanup(self, mock_cleanup):
        self.gpio_manager.cleanup()
        mock_cleanup.assert_called_once()

class TestHardwareIntegration(unittest.TestCase):
    
    @patch('hardware.oled_display.OLEDDisplay')
    @patch('hardware.buzzer_controller.BuzzerController')
    @patch('hardware.gpio_manager.GPIOManager')
    def setUp(self, mock_gpio, mock_buzzer, mock_oled):
        self.mock_oled = mock_oled.return_value
        self.mock_buzzer = mock_buzzer.return_value
        self.mock_gpio = mock_gpio.return_value
    
    def test_drowsiness_alert_sequence(self):
        self.mock_oled.display_status("DROWSY", 0.9, 30)
        self.mock_buzzer.alert_pattern()
        
        self.mock_oled.display_status.assert_called_with("DROWSY", 0.9, 30)
        self.mock_buzzer.alert_pattern.assert_called_once()
    
    def test_warning_sequence(self):
        self.mock_oled.display_status("WARNING", 0.7, 25)
        self.mock_buzzer.warning_pattern()
        
        self.mock_oled.display_status.assert_called_with("WARNING", 0.7, 25)
        self.mock_buzzer.warning_pattern.assert_called_once()
    
    def test_normal_status_display(self):
        self.mock_oled.display_status("NORMAL", 0.3, 23)
        self.mock_oled.display_status.assert_called_with("NORMAL", 0.3, 23)

class TestHardwareErrorHandling(unittest.TestCase):
    
    @patch('board.SCL')
    @patch('board.SDA')
    @patch('busio.I2C')
    @patch('adafruit_ssd1306.SSD1306_I2C')
    def test_oled_connection_error(self, mock_ssd1306, mock_i2c, mock_sda, mock_scl):
        mock_ssd1306.side_effect = Exception("I2C connection failed")
        
        with self.assertRaises(Exception):
            OLEDDisplay()
    
    @patch('RPi.GPIO.setup')
    @patch('RPi.GPIO.setmode')
    @patch('RPi.GPIO.PWM')
    def test_buzzer_gpio_error(self, mock_pwm, mock_setmode, mock_setup):
        mock_setup.side_effect = Exception("GPIO setup failed")
        
        with self.assertRaises(Exception):
            BuzzerController(pin=18)
    
    @patch('RPi.GPIO.setmode')
    def test_gpio_manager_initialization_error(self, mock_setmode):
        mock_setmode.side_effect = Exception("GPIO mode setup failed")
        
        with self.assertRaises(Exception):
            GPIOManager()

if __name__ == '__main__':
    unittest.main()