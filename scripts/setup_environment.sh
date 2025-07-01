#!/bin/bash
# scripts/setup_environment.sh

echo "Setting up Driver Drowsiness Detection System..."

# Update system
sudo apt update && sudo apt upgrade -y

# Install system dependencies
sudo apt install -y python3-pip python3-venv python3-dev
sudo apt install -y libopencv-dev python3-opencv
sudo apt install -y i2c-tools libi2c-dev
sudo apt install -y libatlas-base-dev libhdf5-dev
sudo apt install -y portaudio19-dev python3-pyaudio

# Enable I2C
sudo raspi-config nonint do_i2c 0

# Create virtual environment
python3 -m venv drowsiness_env
source drowsiness_env/bin/activate

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Create necessary directories
mkdir -p logs models data/raw data/processed scripts

# Set permissions
chmod +x scripts/*.sh

# Install systemd service
sudo cp scripts/drowsiness-detection.service /etc/systemd/system/
sudo systemctl daemon-reload

echo "Setup complete! Activate environment with: source drowsiness_env/bin/activate"