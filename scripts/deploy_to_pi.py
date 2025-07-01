# scripts/deploy_to_pi.py
#!/usr/bin/env python3
import os
import subprocess
import sys
import shutil
from pathlib import Path

class PiDeployer:
    def __init__(self):
        self.pi_user = "pi"
        self.pi_host = None
        self.project_name = "drowsiness-detection"
        
    def setup_ssh_connection(self):
        """Setup SSH connection to Pi"""
        self.pi_host = input("Enter Raspberry Pi IP address: ")
        
        # Test connection
        result = subprocess.run(
            ["ssh", f"{self.pi_user}@{self.pi_host}", "echo 'Connected'"],
            capture_output=True, text=True
        )
        
        if result.returncode != 0:
            print("Failed to connect to Raspberry Pi")
            sys.exit(1)
        
        print("✓ Connected to Raspberry Pi")
    
    def transfer_files(self):
        """Transfer project files to Pi"""
        print("Transferring files to Raspberry Pi...")
        
        # Create project directory on Pi
        subprocess.run([
            "ssh", f"{self.pi_user}@{self.pi_host}",
            f"mkdir -p /home/{self.pi_user}/{self.project_name}"
        ])
        
        # Transfer essential files
        files_to_transfer = [
            "src/",
            "models/drowsiness_model.tflite",
            "config.yaml",
            "requirements_pi.txt",
            "scripts/install_dependencies.sh"
        ]
        
        for file_path in files_to_transfer:
            if os.path.exists(file_path):
                subprocess.run([
                    "scp", "-r", file_path,
                    f"{self.pi_user}@{self.pi_host}:/home/{self.pi_user}/{self.project_name}/"
                ])
        
        print("✓ Files transferred")
    
    def setup_pi_environment(self):
        """Setup Python environment on Pi"""
        print("Setting up environment on Raspberry Pi...")
        
        commands = [
            "cd /home/pi/drowsiness-detection",
            "python3 -m venv venv",
            "source venv/bin/activate",
            "pip install --upgrade pip",
            "pip install -r requirements_pi.txt",
            "sudo raspi-config nonint do_camera 0",
            "sudo raspi-config nonint do_i2c 0"
        ]
        
        subprocess.run([
            "ssh", f"{self.pi_user}@{self.pi_host}",
            " && ".join(commands)
        ])
        
        print("✓ Environment setup complete")
    
    def create_systemd_service(self):
        """Create systemd service for auto-start"""
        service_content = f"""[Unit]
Description=Driver Drowsiness Detection System
After=network.target

[Service]
Type=simple
User={self.pi_user}
WorkingDirectory=/home/{self.pi_user}/{self.project_name}
Environment=PATH=/home/{self.pi_user}/{self.project_name}/venv/bin
ExecStart=/home/{self.pi_user}/{self.project_name}/venv/bin/python src/main_pi.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
"""
        
        # Write service file
        with open("/tmp/drowsiness-detection.service", "w") as f:
            f.write(service_content)
        
        # Transfer and install service
        subprocess.run([
            "scp", "/tmp/drowsiness-detection.service",
            f"{self.pi_user}@{self.pi_host}:/tmp/"
        ])
        
        subprocess.run([
            "ssh", f"{self.pi_user}@{self.pi_host}",
            "sudo mv /tmp/drowsiness-detection.service /etc/systemd/system/ && sudo systemctl daemon-reload"
        ])
        
        print("✓ Systemd service created")
    
    def deploy(self):
        """Main deployment function"""
        print("🚀 Starting deployment to Raspberry Pi...")
        
        self.setup_ssh_connection()
        self.transfer_files()
        self.setup_pi_environment()
        self.create_systemd_service()
        
        print("\n✅ Deployment complete!")
        print("\nTo start the system:")
        print(f"ssh {self.pi_user}@{self.pi_host}")
        print("cd drowsiness-detection")
        print("source venv/bin/activate")
        print("python src/main_pi.py")
        print("\nTo enable auto-start:")
        print("sudo systemctl enable drowsiness-detection")

if __name__ == "__main__":
    deployer = PiDeployer()
    deployer.deploy()
