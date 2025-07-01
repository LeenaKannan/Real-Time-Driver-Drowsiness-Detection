try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False
    print("RPi.GPIO not available - running in simulation mode")

class GPIOManager:
    def __init__(self):
        self.gpio_available = GPIO_AVAILABLE
        self.pins_setup = []
        
        if self.gpio_available:
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
    
    def setup_output_pin(self, pin):
        if self.gpio_available:
            GPIO.setup(pin, GPIO.OUT)
            GPIO.output(pin, GPIO.LOW)
            self.pins_setup.append(pin)
        else:
            print(f"[SIM] Setup output pin {pin}")
    
    def setup_input_pin(self, pin, pull_up_down=None):
        if self.gpio_available:
            if pull_up_down:
                GPIO.setup(pin, GPIO.IN, pull_up_down=pull_up_down)
            else:
                GPIO.setup(pin, GPIO.IN)
            self.pins_setup.append(pin)
        else:
            print(f"[SIM] Setup input pin {pin}")

    def setup_pin(self, pin, mode):
        if mode == 'OUTPUT':
            self.setup_output_pin(pin)
        elif mode == 'INPUT':
            self.setup_input_pin(pin)
        else:
            print(f"[SIM] Unknown mode {mode} for pin {pin}")
    
    def digital_write(self, pin, value):
        if self.gpio_available:
            GPIO.output(pin, GPIO.HIGH if value else GPIO.LOW)
        else:
            print(f"[SIM] Pin {pin} set to {'HIGH' if value else 'LOW'}")
    
    def digital_read(self, pin):
        if self.gpio_available:
            return GPIO.input(pin)
        else:
            print(f"[SIM] Reading pin {pin}")
            return False
    
    def cleanup(self):
        if self.gpio_available:
            for pin in self.pins_setup:
                GPIO.output(pin, GPIO.LOW)
            GPIO.cleanup()
        else:
            print("[SIM] GPIO cleanup")
        self.pins_setup = []

   
