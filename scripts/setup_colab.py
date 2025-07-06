# scripts/setup_colab.py
"""
Run this first in Google Colab to set up the environment
"""

# Install required packages
!pip install -q tensorflow==2.13.0
!pip install -q opencv-python-headless==4.8.0.74
!pip install -q mediapipe==0.10.3
!pip install -q scikit-learn==1.3.0
!pip install -q matplotlib==3.7.2
!pip install -q tqdm

# Clone or upload your project
import os
import shutil

# Create project structure
project_dirs = [
    'src/models',
    'src/utils',
    'data/raw',
    'data/processed',
    'models',
    'logs'
]

for directory in project_dirs:
    os.makedirs(directory, exist_ok=True)

print("✅ Environment setup complete!")
print("📁 Project structure created")
print("🚀 Ready for training!")
