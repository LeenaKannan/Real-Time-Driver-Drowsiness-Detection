#!/bin/bash

set -e

SERVICE_NAME="drowsiness-detection"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PYTHON_PATH="$HOME/drowsiness_env/bin/python"
MAIN_SCRIPT="$PROJECT_DIR/src/main.py"

echo "Setting up autostart service for Driver Drowsiness Detection..."

if [ ! -f "$MAIN_SCRIPT" ]; then
    echo "Error: main.py not found at $MAIN_SCRIPT"
    exit 1
fi

echo "Creating systemd service file..."
sudo tee "$SERVICE_FILE" > /dev/null <<EOF
[Unit]
Description=Driver Drowsiness Detection System
After=network.target
Wants=network.target

[Service]
Type=simple
User=pi
Group=pi
WorkingDirectory=$PROJECT_DIR
Environment=PYTHONPATH=$PROJECT_DIR/src
ExecStart=$PYTHON_PATH $MAIN_SCRIPT
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

echo "Reloading systemd daemon..."
sudo systemctl daemon-reload

echo "Enabling service..."
sudo systemctl enable "$SERVICE_NAME"

echo "Starting service..."
sudo systemctl start "$SERVICE_NAME"

echo "Checking service status..."
sudo systemctl status "$SERVICE_NAME" --no-pager

echo ""
echo "Service commands:"
echo "  Start:   sudo systemctl start $SERVICE_NAME"
echo "  Stop:    sudo systemctl stop $SERVICE_NAME"
echo "  Restart: sudo systemctl restart $SERVICE_NAME"
echo "  Status:  sudo systemctl status $SERVICE_NAME"
echo "  Logs:    sudo journalctl -u $SERVICE_NAME -f"
echo "  Disable: sudo systemctl disable $SERVICE_NAME"

echo ""
echo "Autostart service setup completed!"
echo "The drowsiness detection system will now start automatically on boot."