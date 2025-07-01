#!/bin/bash

set -e

echo "Installing Driver Drowsiness Detection System Dependencies..."

sudo apt-get update
sudo apt-get upgrade -y

echo "Installing system packages..."
sudo apt-get install -y python3-pip python3-venv python3-dev
sudo apt-get install -y cmake build-essential pkg-config
sudo apt-get install -y libjpeg-dev libtiff5-dev libjasper-dev libpng-dev
sudo apt-get install -y libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
sudo apt-get install -y libxvidcore-dev libx264-dev
sudo apt-get install -y libfontconfig1-dev libcairo2-dev
sudo apt-get install -y libgdk-pixbuf2.0-dev libpango1.0-dev
sudo apt-get install -y libgtk2.0-dev libgtk-3-dev
sudo apt-get install -y libatlas-base-dev gfortran
sudo apt-get install -y libhdf5-dev libhdf5-serial-dev libhdf5-103
sudo apt-get install -y libqtgui4 libqtwebkit4 libqt4-test
sudo apt-get install -y i2c-tools
sudo apt-get install -y git

echo "Installing GPIO and I2C libraries..."
sudo apt-get install -y python3-rpi.gpio
sudo raspi-config nonint do_i2c 0

echo "Installing Python packages..."
pip3 install --upgrade pip
pip3 install numpy==1.21.6
pip3 install opencv-python==4.6.0.66
pip3 install mediapipe==0.8.11
pip3 install tensorflow==2.9.2
pip3 install Pillow==9.5.0
pip3 install PyYAML==6.0
pip3 install scipy==1.9.3
pip3 install scikit-learn==1.1.3
pip3 install matplotlib==3.5.3
pip3 install seaborn==0.12.2
pip3 install adafruit-circuitpython-ssd1306
pip3 install adafruit-circuitpython-busdevice
pip3 install board
pip3 install busio
pip3 install digitalio

echo "Setting up camera..."
sudo raspi-config nonint do_camera 0

echo "Adding user to gpio and i2c groups..."
sudo usermod -a -G gpio $USER
sudo usermod -a -G i2c $USER

echo "Creating virtual environment..."
python3 -m venv ~/drowsiness_env
source ~/drowsiness_env/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

echo "Setting up permissions..."
sudo chmod +x scripts/setup_environment.sh
sudo chmod +x scripts/autostart_service.sh

echo "Enabling I2C and Camera..."
echo "dtparam=i2c_arm=on" | sudo tee -a /boot/config.txt
echo "dtparam=camera=on" | sudo tee -a /boot/config.txt
echo "gpu_mem=128" | sudo tee -a /boot/config.txt

echo "Creating models directory..."
mkdir -p models
mkdir -p data/processed
mkdir -p data/raw

echo "Installation completed successfully!"
echo "Please reboot the system to enable I2C and camera:"
echo "sudo reboot"