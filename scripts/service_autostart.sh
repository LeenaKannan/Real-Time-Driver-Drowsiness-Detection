#!/bin/bash
# scripts/setup_autostart.sh
echo "Setting up auto-start service..."

# Copy service file
sudo cp drowsiness-detection.service /etc/systemd/system/

# Reload systemd
sudo systemctl daemon-reload

# Enable service
sudo systemctl enable drowsiness-detection

# Start service
sudo systemctl start drowsiness-detection

echo "Auto-start setup complete!"
echo "Service status:"
sudo systemctl status drowsiness-detection
